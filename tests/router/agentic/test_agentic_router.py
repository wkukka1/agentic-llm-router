"""The agentic routing layer (router.agentic)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from router.agentic.result import CallBudgetExceeded
from router.agentic import (
    AgenticResult,
    AgenticRouter,
    ClientRegistry,
    HeuristicTriage,
    LLMClient,
    LLMDecomposer,
    LLMResponse,
    LLMSynthesizer,
    LLMTriage,
    NaiveDecomposer,
    SubCall,
    TriageDecision,
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
def test_registry_synthesises_unknown_ids():
    reg = ClientRegistry({"known": LLMClient(EchoAdapter("known", prefix="k"))})
    assert reg.get("known").complete("x").content.startswith("[k::known]")
    # unseen id alongside a real client -> refuse the echo placeholder (CLIENTREGISTRY-001)
    with pytest.raises(LookupError, match="no client registered"):
        reg.get("surprise")


def test_registry_built_with_no_clients_still_synthesises_unknown_ids():
    # no explicit clients given at all (e.g. AgenticRouter(clients=None), the
    # all-echo demo config) -> every id is expected to be synthesised, so
    # this must not raise
    reg = ClientRegistry()
    made = reg.get("surprise")
    assert isinstance(made, LLMClient) and reg.get("surprise") is made
    assert "surprise" in reg.synthesized


def test_can_serve_true_for_a_registered_id():
    reg = ClientRegistry({"known": LLMClient(EchoAdapter("known"))})
    assert reg.can_serve("known") is True


def test_can_serve_true_for_anything_in_a_pure_echo_registry():
    # no real clients at all -> the echo factory answers anything safely, so
    # nothing is excluded, and the probe must not synthesise a client (the
    # whole point of can_serve is a side-effect-free check)
    reg = ClientRegistry()
    assert reg.can_serve("surprise") is True
    assert "surprise" not in reg and "surprise" not in reg.synthesized


def test_can_serve_false_for_an_unregistered_id_alongside_real_clients():
    # the exact CLIENTREGISTRY-001 shape: once the registry holds a real
    # client, an unregistered id must not be considered servable either --
    # and the probe itself must not mutate the registry
    reg = ClientRegistry({"known": LLMClient(EchoAdapter("known"))})
    assert reg.can_serve("surprise") is False
    assert "surprise" not in reg and "surprise" not in reg.synthesized


def test_can_serve_true_for_a_non_echo_default_factory():
    # a registry with a real, dynamic default_factory (e.g. .ollama()/.langchain(),
    # or a caller's own factory) must keep resolving unregistered ids -- that's
    # the intended behaviour, not a bug this fix should restrict
    reg = ClientRegistry({"known": LLMClient(EchoAdapter("known"))},
                         default_factory=lambda mid: LLMClient(EchoAdapter(mid)))
    assert reg.can_serve("surprise") is True


def test_eligible_mask_matches_can_serve_per_id_in_order():
    reg = ClientRegistry({"a": LLMClient(EchoAdapter("a")), "c": LLMClient(EchoAdapter("c"))})
    mask = reg.eligible_mask(["a", "b", "c"])
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask, [True, False, True])


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


class _FixedChat:
    """LangChain-chat-model stand-in whose ``invoke`` returns a fixed reply."""

    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, _prompt):
        return self.reply


def test_llm_decomposer_keeps_leading_decimal_number():
    """LLMDECOMPOSER-001: a task starting with a decimal is not a list marker."""
    reply = "3.14 times 2 equals what?\n2.5 GHz is how many MHz?\n1.2.3 is which version?"
    assert LLMDecomposer(_FixedChat(reply))("q") == [
        "3.14 times 2 equals what?",
        "2.5 GHz is how many MHz?",
        "1.2.3 is which version?",
    ]


def test_llm_decomposer_drops_exact_and_case_or_punctuation_repeats():
    reply = "Do task A.\ndo task a\nDo  task A!\nDo task B."
    assert LLMDecomposer(_FixedChat(reply))("q") == ["Do task A.", "Do task B."]


def test_llm_decomposer_dedupes_before_folding_past_max_parts():
    reply = "Task A\nTask A\nTask B\nTask C"
    assert LLMDecomposer(_FixedChat(reply), max_parts=2)("q") == ["Task A", "Task B Task C"]


def test_llm_decomposer_still_strips_list_markers():
    reply = "1. summarise the text\n2) translate it\n- proofread it\n* publish it\n## done soon"
    assert LLMDecomposer(_FixedChat(reply))("q") == [
        "summarise the text",
        "translate it",
        "proofread it",
        "publish it",
        "done soon",
    ]


# --------------------------------------------------------------------------- #
# AgenticRouter end to end (echo clients, plan orchestrator)                 #
# --------------------------------------------------------------------------- #
def _agent(**kw):
    return AgenticRouter(FakeTextRouter(), ClientRegistry.echo(POOL), **kw)


def test_single_prompt_goes_straight_to_routed_model():
    res = _agent().run("What is the capital of France?")
    assert res.mode == "single"
    assert res.selected_model_id in POOL
    assert res.selected_model_id in res.answer          # echo marker
    assert len(res.steps) == 1 and res.steps[0].mode == "single"


def test_run_result_carries_the_full_triage_score_spread():
    res = _agent().run("What is the capital of France?")
    assert set(res.triage_scores) == set(POOL)          # every pool model scored, not just the winner
    assert res.triage_scores[res.selected_model_id] == pytest.approx(res.predicted_quality)


def test_route_decision_restricts_to_servable_models():
    # FakeTextRouter ties fast-small/big-strong/coder at 0.9 for a plain
    # prompt (argmax -> lowest column index, "fast-small"); with "fast-small"
    # deliberately unregistered, route_decision must never select it, even
    # though it ties for the best raw score.
    reg = ClientRegistry({"big-strong": LLMClient(EchoAdapter("big-strong")),
                          "coder": LLMClient(EchoAdapter("coder"))})
    agent = AgenticRouter(FakeTextRouter(), reg)
    rr = agent.route_decision("a simple question")
    assert rr.selected_model_ids[0] == "big-strong"
    res = agent.run("a simple question")
    assert res.mode == "single" and res.selected_model_id == "big-strong"


def test_compound_prompt_is_orchestrated_and_synthesised():
    res = _agent().run("Explain quicksort and then implement it in Rust "
                       "and then analyse the complexity")
    assert res.mode == "orchestrated"
    assert res.orchestrator == "plan"
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
# known_bugs/AgenticRouter fixes (AGENTICROUTER-001..004)                     #
# --------------------------------------------------------------------------- #
def _agent_with(reg, **kw):
    return AgenticRouter(FakeTextRouter(), reg, **kw)


def _costed_registry(model_ids, cost=1.0):
    return ClientRegistry({m: LLMClient(CallableAdapter(
        m, lambda p, m=m, cost=cost, **kw: LLMResponse(f"{m}:{p}", cost=cost)))
        for m in model_ids})


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


# --------------------------------------------------------------------------- #
# routed helpers: decomposer / synthesizer pick their pool model by threshold #
# --------------------------------------------------------------------------- #
_HELPER_SCORES = pd.DataFrame([[0.95, 0.80, 0.50]], index=["q"], columns=["big", "mid", "small"])
_HELPER_COSTS = [3.0, 1.0, 0.1]


def _helper_agent(reg=None, **kw):
    router = MatrixRouter(_HELPER_SCORES, model_costs=_HELPER_COSTS)
    reg = reg or ClientRegistry.echo(["big", "mid", "small"])
    return AgenticRouter(router, reg, query_resolver=lambda p: "q", **kw)


def test_helper_model_is_the_cheapest_above_the_threshold():
    model_id, q = _helper_agent().pick_helper_model("anything", min_quality=0.75)
    assert (model_id, q) == ("mid", pytest.approx(0.80))


def test_helper_model_falls_back_to_the_top_score_when_nothing_clears_it():
    model_id, q = _helper_agent().pick_helper_model("anything", min_quality=0.99)
    assert (model_id, q) == ("big", pytest.approx(0.95))


def test_helper_model_skips_an_unregistered_top_scorer():
    # "big" (0.95, the top score) is deliberately unregistered -- the
    # fallback-to-top-score path must skip it and land on "mid" (0.80),
    # never pick a model clients can't actually call.
    reg = ClientRegistry({m: LLMClient(CallableAdapter(m, lambda p, **kw: LLMResponse(f"{m}:{p}")))
                          for m in ["mid", "small"]})
    model_id, q = _helper_agent(reg).pick_helper_model("anything", min_quality=0.99)
    assert (model_id, q) == ("mid", pytest.approx(0.80))


def test_routed_decomposer_and_synthesizer_run_on_the_picked_pool_model():
    def reply(m):
        def fn(p, **kw):
            if "Break the request below" in p:
                return LLMResponse("Task A\nTask B", cost=0.5)
            if "Sub-answers" in p:
                return LLMResponse("FINAL", cost=0.5)
            return LLMResponse(f"{m}:{p}", cost=1.0)
        return fn

    reg = ClientRegistry({m: LLMClient(CallableAdapter(m, reply(m))) for m in ["big", "mid", "small"]})
    agent = _helper_agent(reg)
    agent.decomposer = LLMDecomposer(chat_for=lambda r: agent.helper_call("decompose", r, 0.75))
    agent.synthesizer = LLMSynthesizer(chat_for=lambda r: agent.helper_call("synthesize", r, 0.75))

    res = agent.run("First, do one thing. Then do another thing.")

    assert res.mode == "orchestrated"
    assert [s.prompt for s in res.steps] == ["Task A", "Task B"]
    assert res.answer == "FINAL"
    assert [(h["role"], h["model_id"]) for h in res.helper_calls] == [
        ("decompose", "mid"), ("synthesize", "mid")]
    assert all(h["predicted_quality"] == pytest.approx(0.80) for h in res.helper_calls)


# --------------------------------------------------------------------------- #
# decomposition policy: AgenticRouter.decompose_plan gates LLM decomposer calls #
# --------------------------------------------------------------------------- #
class _WeakRouter(FakeTextRouter):
    """Every prompt scores 0.3 (< the 0.55 triage threshold), so triage
    always asks to orchestrate -- the case that caused runaway re-splitting."""

    def _row(self, text: str) -> list[float]:
        return [0.3] * len(self._model_ids)


class _CountingDecomposeChat:
    """Records the request of every decompose call; replies from ``replies``
    (request -> newline-separated sub-tasks), else echoes the request."""

    def __init__(self, replies=None):
        self.replies = replies or {}
        self.requests: list[str] = []

    def invoke(self, text):
        request = text.rsplit("Request:", 1)[1].strip()
        self.requests.append(request)
        return self.replies.get(request, request)


def _decomposing_agent(replies=None):
    chat = _CountingDecomposeChat(replies)
    agent = AgenticRouter(_WeakRouter(), ClientRegistry.echo(POOL), decomposer=LLMDecomposer(chat))
    return agent, chat


def test_root_atomic_sentence_skips_the_llm_decomposer():
    agent, chat = _decomposing_agent()
    res = agent.run("What is the capital of France?")
    assert chat.requests == []
    assert res.mode == "single"


def test_root_comma_separated_request_is_decomposed():
    prompt = "Explain why the cache misses, fix the lookup, and write a test for it"
    agent, chat = _decomposing_agent({prompt: "Explain why the cache misses.\n"
                                              "Fix the cache lookup.\n"
                                              "Write a test for the cache lookup fix."})
    res = agent.run(prompt)
    assert chat.requests == [prompt]
    assert res.mode == "orchestrated" and len(res.steps) == 3


def test_root_obvious_multipart_request_is_decomposed():
    # one sentence, no comma -- only the "and then" marker shows it is multi-part
    prompt = "Explain quicksort and then implement it in Rust"
    agent, chat = _decomposing_agent({prompt: "Explain quicksort.\nImplement quicksort in Rust."})
    res = agent.run(prompt)
    assert chat.requests == [prompt]
    assert res.mode == "orchestrated"


def test_atomic_generated_subtasks_are_not_decomposed_again():
    """The plan orchestrator's steps are leaves (planned fully up front, never
    re-triaged or recursively re-decomposed -- see routed.py's module
    docstring); a generated sub-task that is itself multi-part is answered as
    written, not split further. (RecursiveOrchestrator used to recurse into
    a multi-part sub-task via nested agent._solve calls; that behavior no
    longer exists now that it's been removed.)"""
    prompt = "First, explain quicksort and then implement it in Rust"
    agent, chat = _decomposing_agent({prompt: "Explain quicksort.\nImplement quicksort in Rust."})
    res = agent.run(prompt)
    assert len(chat.requests) == 1                     # weak sub-tasks did not re-enter the LLM
    assert [s.prompt for s in res.steps] == ["Explain quicksort.", "Implement quicksort in Rust."]
    assert all(not s.children for s in res.steps)


def test_unpriced_successful_call_makes_cost_unknown_not_zero():
    """AGENTICROUTER-002: a call that actually completed but whose client
    couldn't price it must make AgenticResult.cost None -- not silently 0.0,
    which would misreport a real, billed call as free."""
    reg = ClientRegistry({m: LLMClient(CallableAdapter(m, lambda p, **kw: LLMResponse(p, cost=None)))
                          for m in POOL})
    res = _agent_with(reg).run("What is the capital of France?")
    assert res.mode == "single"
    assert res.cost is None


def test_run_alpha_overrides_lam_for_just_that_call():
    """run(prompt, alpha=) picks by the IRT-Router reward blend instead of the
    agent's static lam, without mutating the agent for later calls."""
    scores = pd.DataFrame([[0.90, 0.80]], index=["q"], columns=["big", "cheap"])
    router = MatrixRouter(scores, model_costs=[1.0, 0.0])
    agent = AgenticRouter(router, ClientRegistry.echo(["big", "cheap"]),
                          query_resolver=lambda p: "q")  # lam=0.0 (default): quality-only

    assert agent.run("hello").selected_model_id == "big"
    assert agent.run("hello", alpha=1.0).selected_model_id == "big"
    assert agent.run("hello", alpha=0.05).selected_model_id == "cheap"
    # the override doesn't stick: the next call without alpha is back to lam=0.0
    assert agent.run("hello").selected_model_id == "big"


def test_run_alpha_holds_through_orchestrated_sub_calls():
    """One request's alpha must drive every sub-task's re-triage/routing too,
    not just the root decision -- it's read from the per-run context-local
    state (_RunState.alpha), not threaded through each recursive call."""
    scores = pd.DataFrame([[0.90, 0.80]], index=["0"], columns=["big", "cheap"])
    router = MatrixRouter(scores, model_costs=[1.0, 0.0])

    class _TextRouter(MatrixRouter):
        can_route_text = True

        def predict_scores_text(self, texts, *, encoder=None):
            return pd.DataFrame([[0.90, 0.80]] * len(texts), columns=["big", "cheap"])

    text_router = _TextRouter(scores, model_costs=[1.0, 0.0])
    agent = AgenticRouter(text_router, ClientRegistry.echo(["big", "cheap"]))

    res = agent.run("first part here and then second part here", alpha=0.05)
    assert res.mode == "orchestrated"
    assert res.models_used() == ["cheap"]  # every sub-call picked cheap, not big


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
    with pytest.raises(CallBudgetExceeded, match="max_calls=1"):
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
    # decomposable (an atomic prompt would now skip the decomposer's call)
    res = ok.run("Do one thing, then another")
    assert res.mode == "orchestrated" and len(res.steps) == 2

    tight = AgenticRouter(FakeTextRouter(), reg, triage=forced, decomposer=decomposer,
                          synthesizer=synthesizer, max_calls=3, max_depth=1)
    with pytest.raises(CallBudgetExceeded, match="max_calls=3"):
        tight.run("Do one thing, then another")


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


# --------------------------------------------------------------------------- #
# warm-up                                                                     #
# --------------------------------------------------------------------------- #
def test_warmup_delegates_to_the_router_and_reports_stage_times():
    router = FakeTextRouter()
    calls = []
    router.warmup = lambda: calls.append("router") or {"route_text": 0.01}
    timings = AgenticRouter(router).warmup()
    assert calls == ["router"]
    assert timings["route_text"] == 0.01 and "total" in timings


# --------------------------------------------------------------------------- #
# multi-turn input                                                            #
# --------------------------------------------------------------------------- #
import json
from pathlib import Path

from router.agentic.conversation import (
    as_messages,
    history_before_latest,
    latest_user_turn,
    routing_text,
    with_history,
)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "multiturn_small.jsonl"


def _fixture():
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line]


def _messages(conv, upto=None):
    turns = conv["turns"][:upto]
    msgs = []
    for t in turns[:-1]:
        msgs += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["assistant"]}]
    return msgs + [{"role": "user", "content": turns[-1]["user"]}]


def test_routing_text_matches_training_render_prompt():
    """The router was trained on Arena conversations rendered by
    ``render_prompt`` over the ``str(list)`` of user turns; live routing must
    produce the identical string."""
    from training.data.normalize import render_prompt

    for conv in _fixture():
        users = [t["user"] for t in conv["turns"]]
        assert routing_text(as_messages(_messages(conv))) == render_prompt(json.dumps(users))


def test_conversation_helpers():
    msgs = as_messages([{"role": "system", "content": "be terse"},
                        {"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"},
                        {"role": "user", "content": "c"}])
    assert routing_text(msgs) == "a\n\nc"
    assert latest_user_turn(msgs) == "c"
    assert [m["role"] for m in history_before_latest(msgs)] == ["system", "user", "assistant"]
    assert with_history((), "x") == "x"
    assert with_history(history_before_latest(msgs), "x")[-1] == {"role": "user", "content": "x"}
    assert as_messages("hi") == [{"role": "user", "content": "hi"}]
    with pytest.raises(ValueError):
        as_messages([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])


class RecordingTextRouter(FakeTextRouter):
    def __init__(self):
        super().__init__()
        self.texts: list[str] = []

    def predict_scores_text(self, texts, *, encoder=None):
        self.texts.extend(texts)
        return super().predict_scores_text(texts, encoder=encoder)


def _recording_clients(sent):
    def fn(prompt, **kw):
        sent.append(prompt)
        return "ok"
    return ClientRegistry({m: LLMClient(CallableAdapter(m, fn)) for m in POOL})


def test_multiturn_run_scores_all_user_turns_and_sends_history():
    conv = _fixture()[1]                      # two turns
    msgs = _messages(conv)
    sent, router = [], RecordingTextRouter()
    res = AgenticRouter(router, _recording_clients(sent)).run(msgs, conversation_id=conv["conversation_id"])

    assert res.mode == "single"
    assert router.texts[0] == routing_text(as_messages(msgs))     # scored on every user turn
    assert sent == [msgs]                                          # model got history + latest turn
    assert res.prompt == latest_user_turn(as_messages(msgs))
    assert res.conversation_id == conv["conversation_id"] and res.turn_index == 2


def test_multiturn_triage_judges_only_the_latest_turn():
    """A multi-part *earlier* turn must not force decomposition of a simple
    latest turn (triage reads the latest turn), though it is still scored."""
    msgs = [{"role": "user", "content": "First, explain X and then explain Y step by step."},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "Thanks. What is 2+2?"}]
    res = AgenticRouter(FakeTextRouter(), _recording_clients([])).run(msgs)
    assert res.mode == "single"


def test_single_string_run_is_unchanged_and_reports_timings():
    sent = []
    res = AgenticRouter(FakeTextRouter(), _recording_clients(sent)).run("What is 2+2?")
    assert sent == ["What is 2+2?"]                                # plain str, no message list
    assert res.turn_index == 1 and res.conversation_id is None
    assert {"route", "triage", "dispatch", "total"} <= set(res.timings)


def test_conversation_object_input():
    from decompose.conversation import Conversation, Message, MessageRole

    conv = Conversation("c1", [Message(MessageRole.USER, "hello"),
                               Message(MessageRole.ASSISTANT, "hi"),
                               Message(MessageRole.USER, "What is 2+2?")])
    sent = []
    res = AgenticRouter(FakeTextRouter(), _recording_clients(sent)).run(conv)
    assert sent[0][-1] == {"role": "user", "content": "What is 2+2?"} and len(sent[0]) == 3
    assert res.conversation_id == "c1"


def test_single_call_meta_carries_the_sent_prompt_including_history():
    conv = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"},
           {"role": "user", "content": "What is 2+2?"}]
    sent = []
    res = AgenticRouter(FakeTextRouter(), _recording_clients(sent)).run(conv)
    assert res.mode == "single"
    step = res.steps[0]
    assert step.prompt == "What is 2+2?"                 # bare latest turn
    assert step.meta["sent_prompt"] == sent[0]            # the real list sent to the client
    assert step.meta["sent_prompt"] != step.prompt        # history made them differ


def test_orchestrated_subcalls_carry_the_history():
    msgs = [{"role": "user", "content": "I am learning Rust."},
            {"role": "assistant", "content": "Great."},
            {"role": "user", "content": "Explain ownership, and then write a borrow example."}]
    sent = []
    res = AgenticRouter(FakeTextRouter(), _recording_clients(sent)).run(msgs)
    assert res.mode == "orchestrated" and len(sent) >= 2
    for prompt in sent:
        assert isinstance(prompt, list) and prompt[0]["content"] == "I am learning Rust."


# --------------------------------------------------------------------------- #
# ClientRegistry: streaming + the Anthropic serve map                         #
# --------------------------------------------------------------------------- #
SERVE_MAP = {
    "max_tokens": 64,
    "models": {"claude-opus-5": {"input_cost_per_token": 5e-6, "output_cost_per_token": 25e-6},
               "claude-haiku-4-5": {"input_cost_per_token": 1e-6, "output_cost_per_token": 5e-6}},
    "tiers": {"strong": "claude-opus-5", "small": "claude-haiku-4-5"},
    "default_tier": "small",
    "pool": {"big-strong": "strong", "fast-small": "small"},
}


def test_anthropic_registry_maps_pool_ids_through_tiers():
    reg = ClientRegistry.anthropic(SERVE_MAP, pool=POOL)
    strong, small, coder = (reg.get(m) for m in ("big-strong", "fast-small", "coder"))
    assert strong._adapter.model == "claude-opus-5"
    assert small._adapter.model == "claude-haiku-4-5"
    assert coder._adapter.model == "claude-haiku-4-5"           # default tier
    assert strong._adapter.client_kwargs["max_tokens"] == 64
    assert strong.profile.output_cost_per_token == 25e-6
    assert small._adapter is coder._adapter                     # one SDK client per Claude model


def test_anthropic_registry_without_default_tier_rejects_unknown_ids():
    reg = ClientRegistry.anthropic({**SERVE_MAP, "default_tier": None}, pool=["big-strong"])
    with pytest.raises(KeyError, match="mystery"):
        reg.get("mystery")


def test_registry_stream_goes_through_the_llm_client():
    reg = ClientRegistry.echo(POOL)
    gen = reg.stream("coder", "hi")
    chunks = list(gen)
    assert chunks and "coder" in chunks[0]


# --------------------------------------------------------------------------- #
# LLM triage gating                                                           #
# --------------------------------------------------------------------------- #
class _CountingChat(_FixedChat):
    def __init__(self, reply):
        super().__init__(reply)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return super().invoke(prompt)


def test_gated_llm_triage_skips_clear_cut_prompts():
    chat = _CountingChat("DECOMPOSE")
    triage = LLMTriage(chat, margin=0.1)                  # heuristic threshold 0.55
    d = triage("What is 2+2?", model_id="m", predicted_quality=0.9)
    assert chat.calls == 0 and d.mode == "single" and "skipped" in d.reason
    d = triage("Do A and then do B.", model_id="m", predicted_quality=0.58)
    assert chat.calls == 0 and d.mode == "orchestrate"    # multi-part: heuristic is sure


def test_gated_llm_triage_asks_inside_the_margin_or_without_a_score():
    chat = _CountingChat("DECOMPOSE")
    triage = LLMTriage(chat, margin=0.1)
    assert triage("Explain X.", model_id="m", predicted_quality=0.6).mode == "orchestrate"
    assert triage("Explain Y.", model_id="m", predicted_quality=float("nan")).mode == "orchestrate"
    assert chat.calls == 2


def test_default_llm_triage_always_asks():
    chat = _CountingChat("SINGLE")
    triage = LLMTriage(chat)
    triage("What is 2+2?", model_id="m", predicted_quality=0.99)
    triage("Do A and then do B.", model_id="m", predicted_quality=0.99)
    assert chat.calls == 2


def test_skipped_triage_does_not_spend_max_calls():
    triage = LLMTriage(_CountingChat("SINGLE"), margin=0.1)
    agent = _agent_with(ClientRegistry.echo(POOL), triage=triage, max_calls=1)
    assert agent.run("What is the capital of France?").mode == "single"


# --------------------------------------------------------------------------- #
# streaming                                                                   #
# --------------------------------------------------------------------------- #
import threading


def _drain(gen):
    chunks = []
    try:
        while True:
            chunks.append(next(gen))
    except StopIteration as stop:
        return chunks, stop.value


def _chunking_clients(sent=None):
    """Adapters whose stream yields the answer word by word."""
    from router.llm.adapters.base import ProviderAdapter

    class _Words(ProviderAdapter):
        def complete(self, prompt, **kw):
            return LLMResponse(content=f"{self.model} says hi", cost=0.5)

        def stream(self, prompt, **kw):
            if sent is not None:
                sent.append(prompt)
            for w in ("hello ", "from ", self.model):
                yield w
            return LLMResponse(content=f"hello from {self.model}", cost=0.25,
                               input_tokens=7, output_tokens=3)

    return ClientRegistry({m: LLMClient(_Words(m)) for m in POOL})


def test_stream_single_yields_chunks_then_returns_the_result():
    sent = []
    agent = AgenticRouter(FakeTextRouter(), _chunking_clients(sent))
    chunks, res = _drain(agent.stream("What is 2+2?"))
    assert len(chunks) == 3 and "".join(chunks) == res.answer
    assert res.mode == "single" and res.cost == 0.25 and sent == ["What is 2+2?"]
    assert res.steps[0].meta["output_tokens"] == 3
    assert {"route", "triage", "ttft", "dispatch", "total"} <= set(res.timings)
    assert res.timings["ttft"] <= res.timings["total"]


def test_stream_sends_conversation_history():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
            {"role": "user", "content": "What is 2+2?"}]
    sent = []
    _, res = _drain(AgenticRouter(FakeTextRouter(), _chunking_clients(sent)).stream(msgs))
    assert sent == [msgs] and res.turn_index == 2


def test_stream_orchestrated_yields_the_synthesized_answer():
    agent = AgenticRouter(FakeTextRouter(), _chunking_clients())
    chunks, res = _drain(agent.stream("Explain ownership, and then write a borrow example."))
    assert res.mode == "orchestrated" and chunks == [res.answer]


def test_stream_on_result_runs_after_the_stream_off_the_caller_thread():
    seen, done = [], threading.Event()

    def sink(result):
        seen.append((result, threading.current_thread().name))
        done.set()

    agent = AgenticRouter(FakeTextRouter(), _chunking_clients())
    _, res = _drain(agent.stream("What is 2+2?", on_result=sink))
    assert done.wait(5)
    assert seen[0][0] is res and seen[0][1] != threading.current_thread().name


def test_interleaved_streams_keep_separate_run_state():
    """Two streams consumed alternately on one agent each get their own
    call budget (the per-run state must not leak between generators)."""
    agent = AgenticRouter(FakeTextRouter(), _chunking_clients(), max_calls=1)
    a, b = agent.stream("What is 2+2?"), agent.stream("What is 3+3?")
    out_a, out_b = [next(a)], [next(b)]
    ra, rb = _drain(a), _drain(b)
    assert ra[1].mode == rb[1].mode == "single"


def test_routed_helper_call_forwards_generation_kwargs():
    log = []
    reg = ClientRegistry({m: LLMClient(CallableAdapter(
        m, lambda p, **kw: (log.append(kw), LLMResponse("ok"))[1])) for m in ["big", "mid", "small"]})
    _helper_agent(reg).helper_call("synthesize", "anything", 0.75)("text", max_tokens=99)
    assert log == [{"max_tokens": 99}]


# --------------------------------------------------------------------------- #
# AgenticResult.debug_trace() / run(..., debug=...)                          #
# --------------------------------------------------------------------------- #
def test_debug_trace_includes_triage_scores_step_comparison_sent_prompt_and_escalation():
    sub = SubCall(
        prompt="sub prompt", model_id="m1", predicted_quality=0.8, answer="sub answer", cost=0.01,
        meta={"step_scores": {"m1": 0.8, "m2": 0.5}, "sent_prompt": "HISTORY\nsub prompt",
             "escalation_reason": "cheapest model scoring above m0 (0.400): m1 (0.800)"},
    )
    res = AgenticResult(prompt="top prompt", mode="orchestrated", answer="final answer",
                        triage_reason="reason", steps=[sub],
                        triage_scores={"m0": 0.4, "m1": 0.8, "m2": 0.5})
    text = res.debug_trace()
    assert "top prompt" in text and "final answer" in text and "sub answer" in text
    assert "triage_scores: m1=0.800, m2=0.500, m0=0.400" in text   # ranked, highest first
    assert "considered: m1=0.800, m2=0.500" in text
    assert "sent: HISTORY\nsub prompt" in text
    assert "escalation_reason: cheapest model scoring above m0 (0.400): m1 (0.800)" in text


def test_debug_trace_includes_helper_calls_plans_and_quality_check():
    res = AgenticResult(
        prompt="p", mode="orchestrated", answer="a", triage_reason="r",
        helper_calls=[{"role": "decompose", "model_id": "h1", "request": "split this",
                       "output": "parts", "cost": 0.0, "latency": 0.1}],
        plans=[{"plan_id": "p0", "revision": 0, "n_steps": 1, "outcome": "ok",
               "estimated_cost": 0.1, "actual_cost": 0.09}],
        quality_check={"best_round": 0, "passed": True, "score": 0.9, "status": "ok",
                       "rounds": [{"plan_id": "p0", "passed": True, "score": 0.9}]},
    )
    text = res.debug_trace()
    assert "split this" in text and "parts" in text
    assert "p0" in text and "outcome=ok" in text
    assert "passed=True" in text


def test_run_debug_writes_a_trace_file(tmp_path):
    path = tmp_path / "trace.txt"
    res = _agent().run("What is the capital of France?", debug=str(path))
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "capital of France" in text
    assert res.selected_model_id in text


def test_run_debug_true_writes_under_debug_traces_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = _agent().run("What is the capital of France?", debug=True)
    files = list((tmp_path / "debug_traces").glob("*.txt"))
    assert len(files) == 1
    assert res.selected_model_id in files[0].read_text(encoding="utf-8")


def test_run_debug_write_failure_is_logged_not_raised(tmp_path, caplog):
    (tmp_path / "is_a_dir").mkdir()          # writing a file here must fail
    with caplog.at_level(logging.ERROR, logger="router.agentic.router"):
        res = _agent().run("hi", debug=str(tmp_path / "is_a_dir"))
    assert res.mode == "single"              # run() still succeeds
    assert "debug trace write failed" in caplog.text
