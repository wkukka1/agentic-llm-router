"""The agentic routing layer (router.agentic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from router.agentic import (
    AgenticRouter,
    ClientRegistry,
    HeuristicTriage,
    LLMClient,
    LLMDecomposer,
    NaiveDecomposer,
    RecursiveOrchestrator,
    TriageDecision,
    guess_provider,
)
from router.agentic.triage import best_from_result
from router.llm.adapters.fakes import CallableAdapter, EchoAdapter
from router.routing import RouterModel

POOL = ["fast-small", "big-strong", "coder"]


class FakeTextRouter(RouterModel):
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
    b = LLMClient(EchoAdapter("m1"))
    r1, r2 = b.complete("hello world"), b.complete("hello world")
    assert r1.content == r2.content
    assert "m1" in r1.content


def test_callable_client_wraps_fn():
    b = LLMClient(CallableAdapter("m", lambda p, **kw: p.upper()))
    assert b.complete("hi").content == "HI"


def test_registry_synthesises_unknown_ids():
    reg = ClientRegistry({"known": LLMClient(EchoAdapter("known", prefix="k"))})
    assert reg.get("known").complete("x").content.startswith("[k::known]")
    # unseen id -> default echo client, cached
    with pytest.warns(UserWarning, match="no client registered"):
        made = reg.get("surprise")
    assert isinstance(made, LLMClient) and reg.get("surprise") is made


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


def test_naive_decomposer_keeps_a_short_leading_instruction():
    """NAIVEDECOMPOSER-001: "Fix bug" is under min_part_chars but is a real sub-task."""
    assert NaiveDecomposer()("Fix bug. Add tests for the module. Update the docs.") == [
        "Fix bug. Add tests for the module.",
        "Update the docs.",
    ]


def test_naive_decomposer_keeps_a_short_trailing_part():
    assert NaiveDecomposer()("Explain quicksort in detail. Write it in Rust. Thanks.") == [
        "Explain quicksort in detail.",
        "Write it in Rust. Thanks.",
    ]


def test_naive_decomposer_declines_when_every_part_is_short():
    assert NaiveDecomposer()("Do A. Do B.") == ["Do A. Do B."]


def test_naive_decomposer_folds_parts_past_max_parts_into_the_last():
    """NAIVEDECOMPOSER-001: parts beyond max_parts used to be cut off."""
    prompt = "Please handle every item\n" + "\n".join(f"{i}. Handle item number {i} carefully" for i in range(1, 9))
    parts = NaiveDecomposer()(prompt)
    assert len(parts) == 6
    assert parts[:5] == ["Please handle every item."] + [f"Handle item number {i} carefully." for i in range(1, 5)]
    assert parts[5] == " ".join(f"Handle item number {i} carefully." for i in range(5, 9))


def test_naive_decomposer_max_parts_one_declines_rather_than_truncating():
    assert NaiveDecomposer(max_parts=1)("Explain quicksort. Then write it in Rust.") == [
        "Explain quicksort. Then write it in Rust."
    ]


def test_llm_decomposer_folds_lines_past_max_parts_into_the_last():
    """NAIVEDECOMPOSER-001: LLMDecomposer._parse truncated the same way."""
    class Chat:
        def invoke(self, _prompt):
            return "1. first task\n2. second task\n3. third task\n4. fourth task"

    assert LLMDecomposer(Chat(), max_parts=2)("anything") == [
        "first task",
        "second task third task fourth task",
    ]


class _ReplyChat:
    """Chat-model stand-in whose ``invoke`` returns a fixed reply."""

    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, _prompt):
        return self.reply


def test_llm_decomposer_keeps_leading_decimal_number():
    """LLMDECOMPOSER-001: a task starting with a decimal is not a list marker."""
    reply = "3.14 times 2 equals what?\n2.5 GHz is how many MHz?\n1.2.3 is which version?"
    assert LLMDecomposer(_ReplyChat(reply))("q") == [
        "3.14 times 2 equals what?",
        "2.5 GHz is how many MHz?",
        "1.2.3 is which version?",
    ]


def test_llm_decomposer_still_strips_list_markers():
    reply = "1. summarise the text\n2) translate it\n- proofread it\n* publish it\n## done soon"
    assert LLMDecomposer(_ReplyChat(reply))("q") == [
        "summarise the text",
        "translate it",
        "proofread it",
        "publish it",
        "done soon",
    ]


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
    assert res.models_used()
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
