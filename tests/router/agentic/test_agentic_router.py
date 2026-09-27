"""The agentic routing layer (router.agentic)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from router.agentic import (
    AgenticRouter,
    ClientRegistry,
    HeuristicTriage,
    LLMClient,
    LLMDecomposer,
    LLMResponse,
    LLMSynthesizer,
    LLMTriage,
    NaiveDecomposer,
    RecursiveOrchestrator,
    TriageDecision,
    guess_provider,
)
from router.agentic.triage import best_from_result
from router.llm.adapters.fakes import CallableAdapter, EchoAdapter
from router.routing import MatrixRouter, RouterModel

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


def test_route_query_tool_error_is_returned_not_raised():
    """LANGCHAINTOOLORCHESTRATOR-001: handle_tool_error=True only catches
    ToolException -- route_query's LookupError must be converted so a bad
    prompt becomes an observation, not an executor-aborting crash."""
    pytest.importorskip("langchain_core")
    from router.agentic.orchestrator import build_router_tools

    class _FailingAgent:
        router = SimpleNamespace(model_ids=["a"])

        def route_decision(self, q):
            raise LookupError("no usable score for this prompt")

    tools = build_router_tools(_FailingAgent(), 0, [])
    route_query = next(t for t in tools if t.name == "route_query")

    result = route_query.invoke({"query": "x"})

    assert "no usable score for this prompt" in result


def test_route_and_answer_tool_error_is_returned_not_raised():
    """Same bug, via agent._solve (what route_and_answer calls)."""
    pytest.importorskip("langchain_core")
    from router.agentic.orchestrator import build_router_tools

    class _FailingAgent:
        router = SimpleNamespace(model_ids=["a"])

        def _solve(self, prompt, depth):
            raise TypeError("router can't score raw text")

    steps: list = []
    tools = build_router_tools(_FailingAgent(), 0, steps)
    route_and_answer = next(t for t in tools if t.name == "route_and_answer")

    result = route_and_answer.invoke({"query": "x"})

    assert "router can't score raw text" in result
    assert steps == []  # the failed call must not be recorded as a completed SubCall


def test_route_query_reraises_an_undocumented_exception_type():
    """An exception outside the documented LookupError/TypeError failure
    modes (e.g. an AttributeError from a real bug) must not be silently
    swallowed as tool output."""
    pytest.importorskip("langchain_core")
    from router.agentic.orchestrator import build_router_tools

    class _BrokenAgent:
        router = SimpleNamespace(model_ids=["a"])

        def route_decision(self, q):
            raise AttributeError("something else broke")

    tools = build_router_tools(_BrokenAgent(), 0, [])
    route_query = next(t for t in tools if t.name == "route_query")

    with pytest.raises(AttributeError, match="something else broke"):
        route_query.invoke({"query": "x"})


# --------------------------------------------------------------------------- #
# known_bugs/AgenticRouter fixes (AGENTICROUTER-001..004)                     #
# --------------------------------------------------------------------------- #
def _agent_with(reg, **kw):
    return AgenticRouter(FakeTextRouter(), reg, **kw)


def _costed_registry(model_ids, cost=1.0):
    return ClientRegistry({m: LLMClient(CallableAdapter(
        m, lambda p, m=m, cost=cost, **kw: LLMResponse(f"{m}:{p}", cost=cost)))
        for m in model_ids})


class _FixedChat:
    """LangChain-chat-model stand-in whose ``invoke`` returns a fixed reply."""

    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, _prompt):
        return self.reply


def test_result_predicted_quality_matches_the_model_that_answered():
    """AGENTICROUTER-001: with lam driving the pick toward a cheaper, weaker
    model, predicted_quality must be that model's own score, not the pool
    best -- previously the result claimed the pool-best model's quality next
    to the cheaper model that actually answered."""
    scores = pd.DataFrame([[0.90, 0.80]], index=["q"], columns=["big", "cheap"])
    router = MatrixRouter(scores, model_costs=[1.0, 0.0])
    agent = AgenticRouter(router, ClientRegistry.echo(["big", "cheap"]),
                          query_resolver=lambda p: "q", lam=0.5)
    res = agent.run("hello")
    assert res.selected_model_id == "cheap"
    assert res.predicted_quality == pytest.approx(0.80)
    assert res.predicted_quality == res.steps[0].predicted_quality


def test_unpriced_successful_call_makes_cost_unknown_not_zero():
    """AGENTICROUTER-002: a call that actually completed but whose client
    couldn't price it must make AgenticResult.cost None -- not silently 0.0,
    which would misreport a real, billed call as free."""
    reg = ClientRegistry({m: LLMClient(CallableAdapter(m, lambda p, **kw: LLMResponse(p, cost=None)))
                          for m in POOL})
    res = _agent_with(reg).run("What is the capital of France?")
    assert res.mode == "single"
    assert res.cost is None


def test_one_unpriced_leaf_makes_the_whole_orchestrated_total_unknown():
    """AGENTICROUTER-002: an orchestrated run with one known-cost leaf and one
    unpriced (but successful) leaf must report an unknown total, not silently
    sum the known leaf and treat the unpriced one as free."""
    def maybe_priced(p, **kw):
        return LLMResponse(p, cost=None if "second" in p else 1.0)

    reg = ClientRegistry({m: LLMClient(CallableAdapter(m, maybe_priced)) for m in POOL})
    res = _agent_with(reg).run("first part here and then second part here")
    assert res.mode == "orchestrated"
    assert res.cost is None


def test_failed_leaf_still_allows_a_known_total():
    """A failed call (error set) is $0, not unknown -- it didn't complete, so
    nothing was billed. Distinguishes 'failed' from 'succeeded but unpriced'
    (both used to collapse to the same misleading 0.0 total)."""
    def boom(p, **kw):
        if "second" in p:
            raise TimeoutError("provider timeout")
        return LLMResponse(p, cost=1.0)

    reg = ClientRegistry({m: LLMClient(CallableAdapter(m, boom)) for m in POOL})
    res = _agent_with(reg).run("first part here and then second part here")
    assert res.mode == "orchestrated" and len(res.errors()) == 1
    assert res.cost == pytest.approx(1.0)


def test_llm_triage_counts_against_max_calls():
    """AGENTICROUTER-003: LLMTriage's own LLM call must count against
    max_calls -- previously only the routed model call did, so a triage call
    was effectively free and uncapped."""
    triage = LLMTriage(_FixedChat("SINGLE"))
    agent = _agent_with(ClientRegistry.echo(POOL), triage=triage, max_calls=1)
    # the triage call alone spends the whole budget; the routed call that
    # necessarily follows must now be refused
    with pytest.raises(RuntimeError, match="max_calls=1"):
        agent.run("What is the capital of France?")


def test_llm_decomposer_and_synthesizer_count_against_max_calls():
    """AGENTICROUTER-003: LLMDecomposer/LLMSynthesizer calls must count too.
    Budget of exactly 4 (1 decompose + 2 routed sub-calls + 1 synthesize)
    succeeds; one less refuses the synthesizer's call."""
    forced = lambda prompt, **kw: TriageDecision(  # noqa: E731
        "orchestrate", kw["model_id"], 0.1, "forced", kw.get("scores", {}))
    decomposer = LLMDecomposer(_FixedChat("first part here\nsecond part here"))
    synthesizer = LLMSynthesizer(_FixedChat("combined answer"))
    reg = _costed_registry(POOL)

    ok = AgenticRouter(FakeTextRouter(), reg, triage=forced, decomposer=decomposer,
                       synthesizer=synthesizer, max_calls=4, max_depth=1)
    res = ok.run("anything")
    assert res.mode == "orchestrated" and len(res.steps) == 2

    tight = AgenticRouter(FakeTextRouter(), reg, triage=forced, decomposer=decomposer,
                          synthesizer=synthesizer, max_calls=3, max_depth=1)
    with pytest.raises(RuntimeError, match="max_calls=3"):
        tight.run("anything")


def test_reentrant_run_on_the_same_instance_keeps_its_own_call_budget():
    """AGENTICROUTER-004: per-run state used to live on the AgenticRouter
    instance, so a run() nested inside another (e.g. a custom triage/tool
    calling back into the same agent) leaked its call count and triage cache
    into the outer run. A nested run() on the same instance must not spend
    the outer call's budget."""
    reg = ClientRegistry.echo(POOL)
    seen: dict = {}

    def nested_triage(prompt, *, model_id, predicted_quality, scores=None):
        if "outer" not in seen:
            seen["outer"] = True
            # re-enter run() on the SAME instance before the outer call
            # finishes -- simulates two overlapping run() calls without
            # needing real threads.
            inner = agent.run("a fully unrelated inner prompt")
            assert inner.mode == "single"
        return TriageDecision("single", model_id, predicted_quality, "forced", scores or {})

    agent = AgenticRouter(FakeTextRouter(), reg, triage=nested_triage, max_calls=1)
    res = agent.run("the outer prompt")
    assert res.mode == "single"
    assert res.selected_model_id in POOL
