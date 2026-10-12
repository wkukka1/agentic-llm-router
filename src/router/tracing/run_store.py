"""A small SQLite store for ``AgenticRouter`` runs: prompts, model responses,
and **expected vs. actual cost** per call, per request and per plan.

Wire it in as the router's result sink -- results are written on the
background bookkeeping thread, never on the request path::

    from router.tracing.run_store import sqlite_sink

    agent = AgenticRouter(router, clients, result_sink=sqlite_sink())   # data/runs/agentic_runs.sqlite
    agent.run(...); agent.stream(...)                                   # both are recorded

or record a result yourself with :func:`connect` + :func:`record_result`.

Tables
------
``runs``   one row per request (``AgenticResult``): the prompt, conversation
           history, answer, mode, triage reason, chosen model, stage timings,
           and summed expected vs. actual cost of its model calls.
``calls``  one row per node of the request's call tree (``SubCall``): prompt,
           model, the model that actually served it (``provider_model``),
           answer / error, expected cost (and where the estimate came from)
           vs. actual cost, tokens, cache reads, latency. ``is_leaf = 1`` rows
           are real model calls; branch rows group an orchestrated sub-tree.
``plans``  one row per plan revision the plan orchestrator ran: the
           forecaster's estimated cost / quality / latency vs. what executing
           it actually spent, and its outcome (``completed``, ``step_failed``,
           ``over_budget``).

Views
-----
``v_call_costs``          per leaf call: expected, actual, actual - expected, actual / expected
``v_run_costs``           per request: the same, summed over its calls
``v_model_cost_accuracy`` per routed model x serving model x estimate source:
                          calls, total expected vs. actual, mean ratio, errors, latency
``v_plan_costs``          per plan revision: forecast vs. actual

``calls.cost_source`` says what ``actual_cost`` is: ``actual`` (the
provider's usage x prices), ``estimate`` (the call ran but couldn't be
priced, so it is charged its expected cost), ``unknown`` (neither), or
``failed``. The ratio / error columns of the views only use ``actual`` rows --
an estimate compared with itself would read as perfect accuracy -- while
totals include estimates (``runs.n_estimated`` counts them per request).

An expected cost's ``expected_cost_source`` matters when reading the
comparison: ``profile`` is per-token prices x estimated tokens, the same
basis the actual cost uses; ``router_prior`` is the router's training-data
mean cost for the *pool* model, which a different serving model (e.g. a
Claude tier) is not expected to match. See
:func:`router.llm.cost.estimate_call_cost`.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Union

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from ..agentic.result import AgenticResult, SubCall

__all__ = ["DEFAULT_PATH", "SCHEMA", "connect", "record_result", "sqlite_sink"]

#: repo-relative default database (``data/runs/`` is git-ignored)
DEFAULT_PATH = "data/runs/agentic_runs.sqlite"

SCHEMA_VERSION = 2

TABLES = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    recorded_at       TEXT NOT NULL,              -- UTC, ISO 8601
    router            TEXT,                       -- label given to the sink
    conversation_id   TEXT,
    turn_index        INTEGER,
    prompt            TEXT NOT NULL,              -- the user turn answered
    history_json      TEXT,                       -- earlier messages, JSON list
    mode              TEXT NOT NULL,              -- single | orchestrated
    answer            TEXT,
    triage_reason     TEXT,
    selected_model_id TEXT,
    predicted_quality REAL,
    orchestrator      TEXT,
    expected_cost     REAL,                       -- sum over leaf calls; NULL if any is unknown
    actual_cost       REAL,                       -- AgenticResult.cost; NULL if any is unknown
    n_calls           INTEGER,
    n_errors          INTEGER,
    n_estimated       INTEGER,                    -- leaf calls charged their estimate
    route_s           REAL,
    triage_s          REAL,
    dispatch_s        REAL,
    ttft_s            REAL,
    total_s           REAL
);

CREATE TABLE IF NOT EXISTS calls (
    call_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    seq                  INTEGER NOT NULL,        -- pre-order position in the call tree
    parent_seq           INTEGER,                 -- NULL for a top-level call
    depth                INTEGER,
    is_leaf              INTEGER NOT NULL,        -- 1 = a real model call
    mode                 TEXT,
    prompt               TEXT,
    model_id             TEXT,                    -- the routed (pool) model
    provider_model       TEXT,                    -- the model that served it
    predicted_quality    REAL,
    answer               TEXT,
    error                TEXT,
    expected_cost        REAL,
    expected_cost_source TEXT,                    -- profile | router_prior | unknown
    actual_cost          REAL,
    cost_source          TEXT,                    -- actual | estimate | unknown | failed
    input_tokens         INTEGER,
    output_tokens        INTEGER,
    cache_read_tokens    INTEGER,
    cache_write_tokens   INTEGER,
    latency_s            REAL,
    ttft_s               REAL,
    stop_reason          TEXT,
    request_id           TEXT,
    step_id              TEXT                     -- plan step, for plan-orchestrated calls
);

CREATE TABLE IF NOT EXISTS plans (
    plan_row_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    plan_id           TEXT,
    revision          INTEGER,
    n_steps           INTEGER,
    budget            REAL,                       -- NULL = unlimited
    estimated_cost    REAL,                       -- forecast for the steps still to run
    estimated_quality REAL,
    estimated_latency REAL,
    utility           REAL,
    actual_cost       REAL,                       -- what executing this revision spent
    actual_latency    REAL,
    outcome           TEXT
);

CREATE INDEX IF NOT EXISTS ix_calls_run   ON calls(run_id);
CREATE INDEX IF NOT EXISTS ix_calls_model ON calls(model_id);
CREATE INDEX IF NOT EXISTS ix_runs_conv   ON runs(conversation_id);
CREATE INDEX IF NOT EXISTS ix_plans_run   ON plans(run_id);

"""

#: columns added after schema version 1 -- added in place to older files
_ADDED_COLUMNS = (
    ("runs", "n_estimated", "INTEGER"),
    ("calls", "cost_source", "TEXT"),
)

# views are recreated on every connect, so a view change reaches existing files
VIEWS = """
DROP VIEW IF EXISTS v_call_costs;
CREATE VIEW v_call_costs AS
SELECT c.call_id, c.run_id, r.recorded_at, c.model_id, c.provider_model,
       c.expected_cost_source, c.expected_cost, c.actual_cost, c.cost_source,
       CASE WHEN c.cost_source = 'actual' THEN c.actual_cost - c.expected_cost END AS cost_error,
       CASE WHEN c.cost_source = 'actual' AND c.expected_cost > 0
            THEN c.actual_cost / c.expected_cost END AS cost_ratio,
       c.input_tokens, c.output_tokens, c.cache_read_tokens, c.latency_s, c.error
FROM calls c JOIN runs r USING (run_id)
WHERE c.is_leaf = 1;

DROP VIEW IF EXISTS v_run_costs;
CREATE VIEW v_run_costs AS
SELECT run_id, recorded_at, conversation_id, turn_index, mode, selected_model_id,
       n_calls, n_errors, n_estimated, expected_cost, actual_cost,
       CASE WHEN n_estimated = 0 THEN actual_cost - expected_cost END AS cost_error,
       CASE WHEN n_estimated = 0 AND expected_cost > 0
            THEN actual_cost / expected_cost END AS cost_ratio,
       total_s
FROM runs;

DROP VIEW IF EXISTS v_model_cost_accuracy;
CREATE VIEW v_model_cost_accuracy AS
SELECT model_id, provider_model, expected_cost_source,
       COUNT(*)                                     AS n_calls,
       SUM(expected_cost)                           AS total_expected,
       SUM(actual_cost)                             AS total_actual,
       AVG(CASE WHEN cost_source = 'actual' AND expected_cost > 0
                THEN actual_cost / expected_cost END) AS mean_cost_ratio,
       SUM(CASE WHEN cost_source = 'estimate' THEN 1 ELSE 0 END) AS n_estimated,
       SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS n_errors,
       AVG(latency_s)                               AS mean_latency_s
FROM calls
WHERE is_leaf = 1
GROUP BY model_id, provider_model, expected_cost_source;

DROP VIEW IF EXISTS v_plan_costs;
CREATE VIEW v_plan_costs AS
SELECT p.plan_row_id, p.run_id, p.plan_id, p.revision, p.n_steps, p.budget, p.outcome,
       p.estimated_cost, p.actual_cost,
       p.actual_cost - p.estimated_cost AS cost_error,
       CASE WHEN p.estimated_cost > 0 THEN p.actual_cost / p.estimated_cost END AS cost_ratio,
       p.estimated_latency, p.actual_latency, p.estimated_quality
FROM plans p;
"""

SCHEMA = TABLES + VIEWS


def _num(x) -> Optional[float]:
    """A finite float, else ``None`` (NaN / inf / missing are stored as NULL)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _int(x) -> Optional[int]:
    return None if x is None else int(x)


def connect(path: Union[str, Path, None] = None) -> sqlite3.Connection:
    """Open (creating if needed) the run database at ``path`` (default
    :data:`DEFAULT_PATH` under the repo root) with the schema applied."""
    from ..config import REPO_ROOT

    path = Path(path) if path is not None else REPO_ROOT / DEFAULT_PATH
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")      # readers don't block the writer
    conn.executescript(TABLES)
    for table, column, kind in _ADDED_COLUMNS:          # files written by an older version
        have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
    conn.executescript(VIEWS)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return conn


def _walk(steps) -> Iterator[tuple[int, Optional[int], "SubCall"]]:
    """``(seq, parent_seq, node)`` for every node of the call tree, pre-order."""
    seq = 0

    def visit(node, parent):
        nonlocal seq
        me = seq
        seq += 1
        yield me, parent, node
        for child in node.children:
            yield from visit(child, me)

    for step in steps:
        yield from visit(step, None)


def record_result(conn: sqlite3.Connection, result: "AgenticResult", *,
                  router: Optional[str] = None, run_id: Optional[str] = None) -> str:
    """Write one ``AgenticResult`` (its run row, every call, every plan) in
    one transaction; returns the ``run_id``."""
    run_id = run_id or uuid.uuid4().hex
    nodes = list(_walk(result.steps))
    leaves = [leaf for step in result.steps for leaf in step.leaves()]
    expected = [_num(n.meta.get("expected_cost")) for n in leaves]
    t = result.timings or {}
    with conn:
        conn.execute(
            "INSERT INTO runs (run_id, recorded_at, router, conversation_id, turn_index, prompt, "
            "history_json, mode, answer, triage_reason, selected_model_id, predicted_quality, "
            "orchestrator, expected_cost, actual_cost, n_calls, n_errors, n_estimated, route_s, "
            "triage_s, dispatch_s, ttft_s, total_s) VALUES (" + ",".join("?" * 23) + ")",
            (
                run_id, datetime.now(timezone.utc).isoformat(timespec="milliseconds"), router,
                result.conversation_id, _int(result.turn_index), result.prompt,
                json.dumps(result.history or [], ensure_ascii=False), result.mode, result.answer,
                result.triage_reason, result.selected_model_id, _num(result.predicted_quality),
                result.orchestrator,
                None if (not expected or any(e is None for e in expected)) else sum(expected),
                _num(result.cost), len(leaves), sum(1 for n in leaves if n.error is not None and not n.meta.get("superseded")),
                sum(1 for n in leaves if n.meta.get("cost_source") == "estimate"),
                _num(t.get("route")), _num(t.get("triage")), _num(t.get("dispatch")),
                _num(t.get("ttft")), _num(t.get("total")),
            ),
        )
        conn.executemany(
            "INSERT INTO calls (run_id, seq, parent_seq, depth, is_leaf, mode, prompt, model_id, "
            "provider_model, predicted_quality, answer, error, expected_cost, expected_cost_source, "
            "actual_cost, cost_source, input_tokens, output_tokens, cache_read_tokens, "
            "cache_write_tokens, latency_s, ttft_s, stop_reason, request_id, step_id) "
            "VALUES (" + ",".join("?" * 25) + ")",
            [
                (
                    run_id, seq, parent, n.depth, 0 if n.children else 1, n.mode, n.prompt,
                    n.model_id, n.meta.get("provider_model"), _num(n.predicted_quality),
                    n.answer, n.error, _num(n.meta.get("expected_cost")),
                    n.meta.get("expected_cost_source"), _num(n.cost), n.meta.get("cost_source"),
                    _int(n.meta.get("input_tokens")), _int(n.meta.get("output_tokens")),
                    _int(n.meta.get("cache_read_input_tokens")),
                    _int(n.meta.get("cache_creation_input_tokens")),
                    _num(n.meta.get("latency")), _num(n.meta.get("ttft")),
                    n.meta.get("stop_reason"), n.meta.get("request_id"), n.meta.get("step_id"),
                )
                for seq, parent, n in nodes
            ],
        )
        conn.executemany(
            "INSERT INTO plans (run_id, plan_id, revision, n_steps, budget, estimated_cost, "
            "estimated_quality, estimated_latency, utility, actual_cost, actual_latency, outcome) "
            "VALUES (" + ",".join("?" * 12) + ")",
            [
                (
                    run_id, p.get("plan_id"), _int(p.get("revision")), _int(p.get("n_steps")),
                    _num(p.get("budget")), _num(p.get("estimated_cost")),
                    _num(p.get("estimated_quality")), _num(p.get("estimated_latency")),
                    _num(p.get("utility")), _num(p.get("actual_cost")),
                    _num(p.get("actual_latency")), p.get("outcome"),
                )
                for p in (result.plans or [])
            ],
        )
    return run_id


def sqlite_sink(path: Union[str, Path, None] = None, *,
                router: Optional[str] = None) -> Callable[["AgenticResult"], None]:
    """A ``result_sink`` for :class:`~router.agentic.router.AgenticRouter`
    that appends every result to the database at ``path`` (default
    :data:`DEFAULT_PATH`). The connection opens on first use and writes are
    serialized, so one sink may be shared by several routers. ``router`` is
    a free-text label stored on each run (e.g. the run name)."""
    state: dict = {}
    lock = threading.Lock()

    def sink(result: "AgenticResult") -> None:
        with lock:
            if "conn" not in state:
                state["conn"] = connect(path)
            record_result(state["conn"], result, router=router)

    sink.path = path                     # type: ignore[attr-defined]
    return sink
