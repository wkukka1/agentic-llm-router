"""End-to-end: the real agentic stack, from configs/agentic.yaml to the run database.

Loads the trained ``nirt-2d-projected`` router and its BERT encoder once, then
drives ``AgenticRouter`` through the paths a deployment uses -- settings file
-> warm-up -> single prompt -> multi-turn conversation -> streaming -> plan
orchestrator (parallel steps, budget refusal) -> SQLite run store -> report
script. Model calls use echo clients, so this needs no network or API key.

Needs the built data (``data/processed/``: the run, the ``irt`` profile store,
``nirt_observations.parquet``) and the BERT weights; it skips when they're
absent. Marked ``real_data`` (deselect with ``-m 'not real_data'``)::

    pytest tests/e2e -v                        # ~15-30 s, mostly the BERT load

The last test calls Claude for real (a few cents) and only runs with
``ANTHROPIC_LIVE_TEST=1`` plus credentials.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from router.config import REPO_ROOT

RUN = "nirt-2d-projected"
RUN_DIR = REPO_ROOT / "data" / "processed" / "nirt_runs" / RUN
COMPOUND = "Explain ownership in Rust, and then write a short borrow-checker example."
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "multiturn_small.jsonl"

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not (RUN_DIR / "model.pt").exists(), reason=f"router run {RUN} not built"),
]


# --------------------------------------------------------------------------- #
# shared, expensive setup: one router + encoder for the whole module          #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def router():
    from router.routing import NIRTRouter

    r = NIRTRouter.from_run(RUN)
    try:
        r.warmup()
    except (FileNotFoundError, OSError) as exc:          # stores / weights not available
        pytest.skip(f"router can't warm up here: {exc}")
    return r


@pytest.fixture
def db(tmp_path):
    return tmp_path / "e2e_runs.sqlite"


@pytest.fixture(autouse=True)
def fake_shared_chat(monkeypatch):
    """from_settings always builds a real chat model for plan.model (decompose
    + judge + synthesize), even for a request that stays single-shot -- stand
    in for it so this module needs no langchain-ollama install or network,
    matching its own "no network or API key" claim above. The live-Claude test
    (ANTHROPIC_LIVE_TEST=1) uses router.llm clients directly, not this."""
    import router.llm.adapters.langchain_adapter as lca

    monkeypatch.setattr(lca, "chat_model", lambda model, **kw: object())


def _agent(router, db=None, **overrides):
    """AgenticRouter from the shipped settings file, with ``overrides`` applied.
    A ``plan=`` override merges onto the shipped plan settings (preserving
    ``plan.model``) rather than replacing the whole section."""
    from router.agentic import AgenticRouter
    from router.agentic.schemas import AgenticRouterSettings, load_settings

    base = load_settings().model_dump()
    plan_override = overrides.pop("plan", {})
    data = {**base, "warmup": False, "plan": {**base["plan"], **plan_override}, **overrides}
    if db is not None:
        data["result_sink"] = str(db)
    return AgenticRouter.from_settings(AgenticRouterSettings.model_validate(data), router=router)


def _query(db, sql, *args):
    from router.agentic.router import flush_bookkeeping

    flush_bookkeeping(timeout=30)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def _conversation(conv_id):
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        conv = json.loads(line)
        if conv["conversation_id"] == conv_id:
            return conv
    raise KeyError(conv_id)


def _messages(conv):
    msgs = []
    for t in conv["turns"][:-1]:
        msgs += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["assistant"]}]
    return msgs + [{"role": "user", "content": conv["turns"][-1]["user"]}]


# --------------------------------------------------------------------------- #
# 1. settings file -> warm router -> single prompt                            #
# --------------------------------------------------------------------------- #
def test_settings_file_builds_a_warm_router_that_answers(router, db):
    agent = _agent(router, db)
    res = agent.run("What is the capital of France?")

    assert res.mode in ("single", "orchestrated")
    assert res.selected_model_id in router.model_ids
    assert res.answer
    # warm path: routing must not pay any cold start (it was 37 s before warmup existed)
    assert res.timings["route"] < 2.0, res.timings
    leaf = res.steps[0] if res.mode == "single" else res.steps[0].leaves().__next__()
    assert leaf.meta["expected_cost_source"] in ("profile", "router_prior")
    assert leaf.meta["cost_source"] == "actual"            # echo clients report 0.0

    run = _query(db, "SELECT * FROM runs")[0]
    assert run["prompt"] == "What is the capital of France?" and run["router"] == RUN
    assert run["answer"] == res.answer and run["n_calls"] >= 1


# --------------------------------------------------------------------------- #
# 2. multi-turn conversation                                                  #
# --------------------------------------------------------------------------- #
def test_conversation_is_routed_on_all_user_turns_and_answered_with_history(router, db):
    from router.agentic.conversation import as_messages, routing_text

    conv = _conversation("fixture:four-turn")
    msgs = _messages(conv)
    agent = _agent(router, db)
    res = agent.run(msgs, conversation_id=conv["conversation_id"])

    assert res.turn_index == 4 and res.conversation_id == "fixture:four-turn"
    assert res.prompt == msgs[-1]["content"]
    assert len(res.history) == 6                             # 3 earlier user + 3 assistant turns
    # the router scored the whole conversation, rendered as in training
    expected = agent.route_decision(routing_text(as_messages(msgs))).selected_model_ids[0]
    assert res.selected_model_id == expected or res.mode == "orchestrated"
    # every routed call carried the earlier turns (echo answers quote what they were sent)
    for leaf in (n for s in res.steps for n in s.leaves()):
        assert "planning a trip to Japan" in leaf.answer

    run = _query(db, "SELECT conversation_id, turn_index, history_json FROM runs")[0]
    assert run["turn_index"] == 4 and len(json.loads(run["history_json"])) == 6


# --------------------------------------------------------------------------- #
# 3. streaming                                                                #
# --------------------------------------------------------------------------- #
def test_streaming_yields_the_answer_then_the_result(router, db):
    agent = _agent(router, db)
    seen = []
    gen = agent.stream("What is 2+2?", on_result=seen.append)
    chunks = []
    try:
        while True:
            chunks.append(next(gen))
    except StopIteration as stop:
        res = stop.value

    assert chunks and "".join(chunks) == res.answer
    assert 0 < res.timings["ttft"] <= res.timings["total"]
    assert _query(db, "SELECT COUNT(*) AS n FROM runs")[0]["n"] == 1
    assert seen == [res]                                     # on_result ran (after the flush)


# --------------------------------------------------------------------------- #
# 4. plan orchestrator: batched routing, parallel steps, budget               #
# --------------------------------------------------------------------------- #
def test_plan_orchestrator_end_to_end(router, db):
    agent = _agent(router, db, plan={"budget": 1.0, "max_workers": 4})
    res = agent.run(COMPOUND)

    assert res.mode == "orchestrated" and res.orchestrator == "plan"
    assert len(res.steps) >= 2 and not res.errors()
    assert all(s.meta.get("step_id") for s in res.steps)
    assert len(res.plans) == 1 and res.plans[0]["outcome"] == "completed"
    assert res.plans[0]["estimated_cost"] > 0                # router-prior forecast

    plan = _query(db, "SELECT * FROM v_plan_costs")[0]
    assert plan["outcome"] == "completed" and plan["n_steps"] == len(res.steps)
    calls = _query(db, "SELECT step_id, expected_cost, cost_source FROM calls WHERE is_leaf = 1")
    assert len(calls) == len(res.steps)
    assert all(c["step_id"] and c["expected_cost"] > 0 and c["cost_source"] == "actual" for c in calls)


def test_plan_over_budget_is_refused_before_any_call(router, db):
    agent = _agent(router, db, plan={"budget": 1e-12})
    res = agent.run(COMPOUND)

    assert res.mode == "orchestrated" and "over budget" in res.answer
    assert [e.error for e in res.errors()] == ["over_budget"]
    assert _query(db, "SELECT outcome, actual_cost FROM plans")[0] == {"outcome": "over_budget",
                                                                       "actual_cost": None}


# --------------------------------------------------------------------------- #
# 5. the report script reads what the store wrote                             #
# --------------------------------------------------------------------------- #
def test_runs_report_reads_the_database(router, db):
    agent = _agent(router, db)
    agent.run("What is the capital of France?")
    agent.run(COMPOUND)
    _query(db, "SELECT 1")                                   # flush the background writes

    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "agentic" / "runs_report.py"), "--db", str(db)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert out.returncode == 0, out.stderr
    assert "== totals" in out.stdout and "expected vs actual cost by model" in out.stdout
    assert "plan revisions" in out.stdout


# --------------------------------------------------------------------------- #
# 6. optional: real Claude calls                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.environ.get("ANTHROPIC_LIVE_TEST") != "1",
                    reason="live API test; set ANTHROPIC_LIVE_TEST=1 (and credentials) to run")
def test_live_claude_single_and_streamed(router, db):
    agent = _agent(router, db, clients="anthropic")
    res = agent.run("Reply with the single word: pong")
    leaf = res.steps[0] if res.mode == "single" else next(res.steps[0].leaves())
    assert "pong" in res.answer.lower()
    assert leaf.meta["provider_model"].startswith("claude-")
    assert leaf.meta["cost_source"] == "actual" and leaf.cost > 0
    assert leaf.meta["expected_cost_source"] == "profile"

    chunks = list(agent.stream([{"role": "user", "content": "Count from 1 to 5."}]))
    assert "".join(chunks).strip()

    rows = _query(db, "SELECT cost_source, cost_ratio FROM v_call_costs")
    assert all(r["cost_source"] == "actual" and r["cost_ratio"] is not None for r in rows)
