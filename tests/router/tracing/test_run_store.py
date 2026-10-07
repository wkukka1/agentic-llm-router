"""The SQLite run store (router.tracing.run_store) and AgenticRouter's
result_sink / expected-cost bookkeeping that feeds it."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from router.agentic import AgenticRouter, ClientRegistry, LLMClient, LLMResponse
from router.agentic.router import flush_bookkeeping
from router.execution.routed import plan_orchestrator
from router.llm.adapters.fakes import CallableAdapter
from router.llm.cost import estimate_call_cost
from router.llm.profile import LLMProfile
from router.routing import RouterModel
from router.tracing.run_store import SCHEMA_VERSION, connect, record_result, sqlite_sink

POOL = ["small", "big", "coder"]
COMPOUND = "Explain ownership, and then write a borrow example."


class TextRouter(RouterModel):
    kind = "store_text"
    can_route_text = True

    def __init__(self, costs=(0.001, 0.01, 0.005)):
        super().__init__(POOL, name="t")
        self._costs = None if costs is None else np.asarray(costs, float)

    @property
    def default_model_costs(self):
        return self._costs

    def predict_scores(self, query_ids):
        return pd.DataFrame(0.5, index=[str(q) for q in query_ids], columns=POOL)

    def predict_scores_text(self, texts, *, encoder=None):
        return pd.DataFrame([[0.7, 0.9, 0.8]] * len(texts), index=range(len(texts)), columns=POOL)


PRICES = {"input_cost_per_token": 1e-6, "output_cost_per_token": 5e-6}


def _priced_clients(*, fail=()):
    def make(m):
        def fn(prompt, **kw):
            if m in fail:
                raise RuntimeError(f"{m} down")
            return LLMResponse(f"{m} says hi", input_tokens=40, output_tokens=10,
                               meta={"stop_reason": "end_turn", "request_id": f"req-{m}"})
        profile = LLMProfile(name=m, provider="test", model_id=f"served-{m}", **PRICES)
        return LLMClient(CallableAdapter(f"served-{m}", fn), profile)
    return ClientRegistry({m: make(m) for m in POOL})


def _flush():
    """Wait for everything already submitted to the bookkeeping thread."""
    flush_bookkeeping(timeout=5)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "runs.sqlite"


def _rows(db, sql, *args):
    conn = connect(db)
    conn.row_factory = __import__("sqlite3").Row
    return [dict(r) for r in conn.execute(sql, args)]


# --------------------------------------------------------------------------- #
# expected cost on every call                                                 #
# --------------------------------------------------------------------------- #
def test_estimate_call_cost_sources():
    priced = LLMProfile(name="m", provider="x", model_id="m", **PRICES)
    cost, src = estimate_call_cost(priced, 400)
    assert src == "profile" and cost == pytest.approx(100 * 1e-6 + 256 * 5e-6)
    assert estimate_call_cost(None, 400, fallback=0.02) == (0.02, "router_prior")
    assert estimate_call_cost(None, 400) == (None, "unknown")


def test_calls_carry_expected_cost_and_serving_model():
    res = AgenticRouter(TextRouter(), _priced_clients()).run("What is 2+2?")
    meta = res.steps[0].meta
    assert meta["expected_cost_source"] == "profile" and meta["expected_cost"] > 0
    assert meta["provider_model"] == "served-big"
    assert res.cost == pytest.approx(40e-6 + 50e-6)            # actual, priced by LLMClient


def test_unpriced_clients_fall_back_to_the_router_prior():
    res = AgenticRouter(TextRouter()).run("What is 2+2?")      # echo clients, no prices
    assert res.steps[0].meta["expected_cost"] == pytest.approx(0.01)
    assert res.steps[0].meta["expected_cost_source"] == "router_prior"


# --------------------------------------------------------------------------- #
# the store                                                                   #
# --------------------------------------------------------------------------- #
def test_single_run_is_recorded_with_expected_vs_actual(db):
    agent = AgenticRouter(TextRouter(), _priced_clients(), result_sink=sqlite_sink(db, router="t"))
    res = agent.run("What is 2+2?")
    _flush()
    run = _rows(db, "SELECT * FROM runs")[0]
    assert run["prompt"] == "What is 2+2?" and run["answer"] == res.answer and run["router"] == "t"
    assert run["mode"] == "single" and run["n_calls"] == 1 and run["n_errors"] == 0
    assert run["actual_cost"] == pytest.approx(res.cost)
    assert run["expected_cost"] == pytest.approx(res.steps[0].meta["expected_cost"])
    assert run["route_s"] is not None and run["total_s"] >= run["route_s"]
    call = _rows(db, "SELECT * FROM v_call_costs")[0]
    assert call["model_id"] == "big" and call["provider_model"] == "served-big"
    assert call["cost_error"] == pytest.approx(call["actual_cost"] - call["expected_cost"])
    assert call["input_tokens"] == 40
    raw = _rows(db, "SELECT stop_reason, request_id, answer FROM calls")[0]
    assert raw == {"stop_reason": "end_turn", "request_id": "req-big", "answer": "big says hi"}


def test_conversation_history_and_ids_are_stored(db):
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
            {"role": "user", "content": "What is 2+2?"}]
    AgenticRouter(TextRouter(), _priced_clients(), result_sink=sqlite_sink(db)).run(msgs, conversation_id="c9")
    _flush()
    run = _rows(db, "SELECT conversation_id, turn_index, history_json FROM runs")[0]
    assert run["conversation_id"] == "c9" and run["turn_index"] == 2
    assert json.loads(run["history_json"]) == msgs[:2]


def test_plan_orchestrated_run_records_plans_and_steps(db):
    agent = AgenticRouter(TextRouter(), _priced_clients(), orchestrator=plan_orchestrator(budget=1.0),
                          result_sink=sqlite_sink(db))
    res = agent.run(COMPOUND)
    _flush()
    assert res.mode == "orchestrated" and len(res.plans) == 1
    plan = _rows(db, "SELECT * FROM v_plan_costs")[0]
    assert plan["outcome"] == "completed" and plan["n_steps"] == 2 and plan["budget"] == 1.0
    assert plan["estimated_cost"] > 0 and plan["actual_cost"] == pytest.approx(res.cost)
    calls = _rows(db, "SELECT step_id, expected_cost_source FROM calls WHERE is_leaf = 1")
    assert len(calls) == 2 and all(c["step_id"] and c["expected_cost_source"] == "profile" for c in calls)


def test_replanned_and_refused_plans_are_recorded(db):
    agent = AgenticRouter(TextRouter(), _priced_clients(fail={"big"}),
                          orchestrator=plan_orchestrator(), result_sink=sqlite_sink(db))
    agent.run(COMPOUND)
    refused = AgenticRouter(TextRouter(), _priced_clients(),
                            orchestrator=plan_orchestrator(budget=1e-9), result_sink=sqlite_sink(db))
    refused.run(COMPOUND)
    _flush()
    plans = _rows(db, "SELECT revision, outcome, actual_cost FROM plans ORDER BY plan_row_id")
    assert [(p["revision"], p["outcome"]) for p in plans] == [(0, "step_failed"), (1, "completed"),
                                                              (0, "over_budget")]
    assert plans[-1]["actual_cost"] is None
    errors = _rows(db, "SELECT error FROM calls WHERE error IS NOT NULL")
    # the replanned run's failed attempts were replaced by their retries; only
    # the refused run's over-budget marker is an error
    assert [e["error"] for e in errors] == ["over_budget"]


def test_recursive_orchestration_stores_the_call_tree(db):
    agent = AgenticRouter(TextRouter(), _priced_clients(), result_sink=sqlite_sink(db))
    agent.run(COMPOUND)
    _flush()
    calls = _rows(db, "SELECT seq, parent_seq, is_leaf, depth FROM calls ORDER BY seq")
    assert [c["is_leaf"] for c in calls] == [1, 1] and all(c["depth"] == 1 for c in calls)
    model = _rows(db, "SELECT * FROM v_model_cost_accuracy")[0]
    assert model["n_calls"] == 2 and model["total_actual"] == pytest.approx(2 * 90e-6)


def test_stream_is_recorded_with_ttft(db):
    agent = AgenticRouter(TextRouter(), _priced_clients(), result_sink=sqlite_sink(db))
    gen = agent.stream("What is 2+2?")
    for _ in gen:
        pass
    _flush()
    run = _rows(db, "SELECT ttft_s, expected_cost, actual_cost FROM runs")[0]
    assert run["ttft_s"] is not None and run["expected_cost"] > 0 and run["actual_cost"] > 0


def test_nan_and_inf_are_stored_as_null(tmp_path):
    from router.agentic.result import AgenticResult

    conn = connect(tmp_path / "x.sqlite")
    record_result(conn, AgenticResult(prompt="p", mode="single", answer="a", triage_reason="r",
                                      predicted_quality=float("nan"),
                                      plans=[{"plan_id": "x", "budget": float("inf")}]))
    assert conn.execute("SELECT predicted_quality FROM runs").fetchone() == (None,)
    assert conn.execute("SELECT budget FROM plans").fetchone() == (None,)


def test_a_failing_sink_is_logged_not_raised(caplog):
    def broken(result):
        raise ValueError("disk full")

    agent = AgenticRouter(TextRouter(), result_sink=broken)
    with caplog.at_level(logging.ERROR, logger="router.agentic.router"):
        assert agent.run("What is 2+2?").mode == "single"
        _flush()
    assert "result sink failed" in caplog.text


# --------------------------------------------------------------------------- #
# unknown actual cost -> the labelled estimate                                #
# --------------------------------------------------------------------------- #
def _unpriced_clients():
    """Real-looking calls that report no usage and have no prices."""
    return ClientRegistry({m: LLMClient(CallableAdapter(m, lambda p, **kw: LLMResponse("ok")))
                           for m in POOL})


def test_unpriced_call_is_charged_its_estimate():
    res = AgenticRouter(TextRouter(), _unpriced_clients()).run("What is 2+2?")
    leaf = res.steps[0]
    assert leaf.meta["cost_source"] == "estimate" and leaf.cost == pytest.approx(0.01)
    assert res.cost == pytest.approx(0.01)                        # not None, not 0


def test_priced_and_failed_calls_are_labelled():
    ok = AgenticRouter(TextRouter(), _priced_clients()).run("What is 2+2?")
    assert ok.steps[0].meta["cost_source"] == "actual"
    bad = AgenticRouter(TextRouter(), _priced_clients(fail={"big"})).run(COMPOUND)
    assert {s.meta["cost_source"] for s in bad.steps} == {"failed"}


def test_plan_ledger_commits_estimates_for_unpriced_steps():
    agent = AgenticRouter(TextRouter(), _unpriced_clients(), orchestrator=plan_orchestrator())
    res = agent.run(COMPOUND)
    assert res.cost == pytest.approx(0.02)                        # two steps x router prior 0.01
    assert res.plans[0]["actual_cost"] == pytest.approx(0.02)


def test_plan_with_no_cost_signal_at_all_warns_and_commits_zero():
    agent = AgenticRouter(TextRouter(costs=None), _unpriced_clients(), orchestrator=plan_orchestrator())
    with pytest.warns(UserWarning, match="cost is unknown"):
        res = agent.run(COMPOUND)
    assert res.cost is None and res.plans[0]["actual_cost"] == 0.0


def test_estimated_costs_are_kept_out_of_accuracy_ratios(db):
    sink = sqlite_sink(db)
    AgenticRouter(TextRouter(), _unpriced_clients(), result_sink=sink).run("What is 2+2?")
    AgenticRouter(TextRouter(), _priced_clients(), result_sink=sink).run("What is 3+3?")
    _flush()
    calls = _rows(db, "SELECT cost_source, cost_ratio FROM v_call_costs ORDER BY call_id")
    assert calls[0] == {"cost_source": "estimate", "cost_ratio": None}
    assert calls[1]["cost_source"] == "actual" and calls[1]["cost_ratio"] is not None
    runs = _rows(db, "SELECT n_estimated, cost_ratio FROM v_run_costs ORDER BY recorded_at")
    assert runs[0] == {"n_estimated": 1, "cost_ratio": None} and runs[1]["n_estimated"] == 0


def test_connect_adds_new_columns_to_an_older_file(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = connect(path)
    for v in ("v_call_costs", "v_run_costs", "v_model_cost_accuracy", "v_plan_costs"):
        conn.execute(f"DROP VIEW {v}")
    conn.execute("ALTER TABLE calls DROP COLUMN cost_source")    # what a v1 file looks like
    conn.execute("ALTER TABLE runs DROP COLUMN n_estimated")
    conn.commit()
    conn.close()
    conn = connect(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(calls)")}
    assert "cost_source" in cols and conn.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
    res = AgenticRouter(TextRouter(), _priced_clients()).run("What is 2+2?")
    record_result(conn, res)
    assert conn.execute("SELECT cost_source FROM v_call_costs").fetchone() == ("actual",)
