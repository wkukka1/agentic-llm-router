"""The agentic routing layer (router.agentic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from router.agentic import (
    AgenticRouter,
    ClientRegistry,
    CallableClient,
    EchoClient,
    HeuristicTriage,
    NaiveDecomposer,
    RecursiveOrchestrator,
    TriageDecision,
    guess_provider,
)
from router.agentic.triage import best_from_result
from router.routing import Router

POOL = ["fast-small", "big-strong", "coder"]


class FakeTextRouter(Router):
    """Text-capable router: quality 0.9 for every model, unless the prompt
    contains 'hard' (0.2); 'coder' gets +0.05 when the prompt mentions code."""

    kind = "fake_text"
    can_route_text = True

    def __init__(self, model_ids=POOL):
        super().__init__(model_ids, name="fake")

    def _row(self, text: str) -> list[float]:
        base = 0.2 if "hard" in text.lower() else 0.9
        return [base + (0.05 if (m == "coder" and "code" in text.lower()) else 0.0)
                for m in self._model_ids]

    def predict_scores(self, query_ids):
        ids = [str(q) for q in query_ids]
        return pd.DataFrame([[0.5] * len(self._model_ids) for _ in ids],
                            index=ids, columns=self._model_ids)

    def predict_scores_text(self, texts, *, encoder=None):
        return pd.DataFrame([self._row(t) for t in texts],
                            index=list(range(len(texts))), columns=self._model_ids)


# --------------------------------------------------------------------------- #
# LLM clients                                                                    #
# --------------------------------------------------------------------------- #
def test_echo_client_is_deterministic():
    b = EchoClient("m1")
    r1, r2 = b.invoke("hello world"), b.invoke("hello world")
    assert r1.text == r2.text
    assert r1.model_id == "m1" and "m1" in r1.text


def test_callable_client_wraps_fn():
    b = CallableClient("m", lambda p, **kw: p.upper())
    assert b.invoke("hi").text == "HI"


def test_registry_synthesises_unknown_ids():
    reg = ClientRegistry({"known": EchoClient("known", prefix="k")})
    assert reg.get("known").invoke("x").text.startswith("[k::known]")
    # unseen id -> default EchoClient, cached
    made = reg.get("surprise")
    assert isinstance(made, EchoClient) and reg.get("surprise") is made


def test_guess_provider():
    assert guess_provider("gpt-4o") == "openai"
    assert guess_provider("claude-3-5-sonnet") == "anthropic"
    assert guess_provider("some-random-model") is None


# --------------------------------------------------------------------------- #
# triage                                                                      #
# --------------------------------------------------------------------------- #
def test_heuristic_triage_single_when_strong():
    t = HeuristicTriage(quality_threshold=0.55)
    d = t("What is the capital of France?", model_id="big-strong", predicted_quality=0.9)
    assert d.mode == "single" and not d.orchestrate


def test_heuristic_triage_orchestrates_on_weakness():
    t = HeuristicTriage(quality_threshold=0.55)
    d = t("Prove the Riemann hypothesis.", model_id="big-strong", predicted_quality=0.2)
    assert d.orchestrate


def test_heuristic_triage_orchestrates_on_multipart():
    t = HeuristicTriage(quality_threshold=0.55)
    d = t("Explain X and then implement Y.", model_id="m", predicted_quality=0.95)
    assert d.orchestrate


def test_best_from_result():
    r = FakeTextRouter().route_text(["a simple question"])
    mid, q, scores = best_from_result(r)
    assert mid in POOL and q == pytest.approx(0.9)
    assert set(scores) == set(POOL)


# --------------------------------------------------------------------------- #
# decomposition                                                               #
# --------------------------------------------------------------------------- #
def test_naive_decomposer_splits_and_then():
    parts = NaiveDecomposer()("Explain quicksort and then implement it in Rust "
                              "and then analyse the complexity")
    assert len(parts) == 3
    assert parts[0].lower().startswith("explain quicksort")


def test_naive_decomposer_declines_single_task():
    assert NaiveDecomposer()("What is 2 + 2?") == ["What is 2 + 2?"]


# --------------------------------------------------------------------------- #
# AgenticRouter end to end (echo clients, recursive orchestrator)            #
# --------------------------------------------------------------------------- #
def _agent(**kw):
    return AgenticRouter(FakeTextRouter(), ClientRegistry.echo(POOL), **kw)


def test_single_prompt_goes_straight_to_routed_model():
    res = _agent().run("What is the capital of France?")
    assert res.mode == "single"
    assert res.selected_model_id in POOL
    assert res.selected_model_id in res.answer          # echo marker
    assert len(res.steps) == 1 and res.steps[0].mode == "single"


def test_compound_prompt_is_orchestrated_and_synthesised():
    res = _agent().run("Explain quicksort and then implement it in Rust "
                       "and then analyse the complexity")
    assert res.mode == "orchestrated"
    assert res.orchestrator == "recursive"
    assert len(res.steps) == 3
    assert all(s.mode == "single" for s in res.steps)
    # every sub-answer threads into the final synthesis
    for s in res.steps:
        assert s.answer in res.answer
    assert set(res.models_used()) <= set(POOL)


def test_weak_but_indivisible_prompt_collapses_to_single():
    # "hard" -> predicted quality 0.2 so triage wants to decompose, but there is
    # nothing to split -> it falls back to a single routed call.
    res = _agent().run("Do the hard thing now")
    assert res.mode == "single"
    assert res.selected_model_id in POOL


def test_recursion_is_depth_capped():
    forced = lambda prompt, **kw: TriageDecision("orchestrate", "big-strong", 0.1, "forced", {})
    agent = _agent(triage=forced, decomposer=lambda p: ["first subtask here", "second subtask here"],
                   max_depth=1)
    res = agent.run("anything")
    assert res.mode == "orchestrated"
    # depth 1 hits the cap -> children answered directly, no deeper nesting
    assert len(res.steps) == 2
    assert all(not c.children for c in res.steps)


def test_matrix_router_needs_a_resolver():
    from router.routing import MatrixRouter

    mr = MatrixRouter(pd.DataFrame([[0.9, 0.1]], index=["q1"], columns=["a", "b"]))
    ar = AgenticRouter(mr, ClientRegistry.echo(["a", "b"]))
    with pytest.raises(TypeError, match="cannot score raw text"):
        ar.run("hello")
    # with a resolver mapping text -> an existing query_id it works
    ar2 = AgenticRouter(mr, ClientRegistry.echo(["a", "b"]), query_resolver=lambda t: "q1")
    assert ar2.run("hello").selected_model_id == "a"


def test_cost_is_summed_from_client_responses():
    reg = ClientRegistry({m: CallableClient(m, lambda p, **kw: "ok") for m in POOL})
    # CallableClient returns no cost -> total stays 0.0, no crash
    res = AgenticRouter(FakeTextRouter(), reg).run("a and then b and then c")
    assert res.cost == 0.0


# --------------------------------------------------------------------------- #
# LangChain orchestrator (needs langchain)                                    #
# --------------------------------------------------------------------------- #
def test_langchain_orchestrator_tools_build_and_recurse():
    pytest.importorskip("langchain_core")
    from router.agentic.orchestrator import build_router_tools

    agent = _agent()
    steps: list = []
    tools = build_router_tools(agent, depth=0, steps=steps)
    names = {t.name for t in tools}
    assert names == {"route_query", "answer_with_model", "route_and_answer"}

    rq = next(t for t in tools if t.name == "route_query")
    out = rq.invoke({"query": "translate this sentence"})
    assert "best=" in out and "predicted_quality=" in out

    ra = next(t for t in tools if t.name == "route_and_answer")
    ans = ra.invoke({"query": "a short question"})
    assert len(steps) == 1 and steps[0].answer == ans
