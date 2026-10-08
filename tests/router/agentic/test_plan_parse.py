"""Planner output with dependencies: parse_plan_lines, LLMDecomposer.plan and
AgenticRouter.decompose_plan (router.agentic.decompose / router)."""

from __future__ import annotations

import pandas as pd
import pytest

from router.agentic import AgenticRouter, ClientRegistry, LLMDecomposer
from router.agentic.decompose import SubTask, parse_plan_lines
from router.routing import RouterModel


def test_after_marker_becomes_a_dependency():
    text = ("1. Implement quicksort in Rust.\n"
            "2. [after 1] Analyse the time complexity of the Rust quicksort from step 1.")
    assert parse_plan_lines(text) == [
        SubTask("Implement quicksort in Rust."),
        SubTask("Analyse the time complexity of the Rust quicksort from step 1.", (1,)),
    ]


def test_unmarked_lines_are_independent():
    assert [t.depends_on for t in parse_plan_lines("Explain A in detail.\nExplain B in detail.")] == [(), ()]


def test_forward_self_and_unknown_references_are_dropped():
    text = ("1. [after 2] Do the first thing.\n"
            "2. [after 2, 9] Do the second thing.\n"
            "3. [after 1 and 2] Do the third thing.")
    assert [t.depends_on for t in parse_plan_lines(text)] == [(), (), (1, 2)]


def test_reference_to_a_dropped_repeat_points_at_the_original():
    text = ("1. Implement quicksort in Rust.\n"
            "2. implement quicksort in rust\n"
            "3. [after 2] Test the Rust quicksort.")
    tasks = parse_plan_lines(text)
    assert [t.text for t in tasks] == ["Implement quicksort in Rust.", "Test the Rust quicksort."]
    assert tasks[1].depends_on == (1,)


def test_none_and_preamble_lines_are_not_tasks():
    assert parse_plan_lines("Here is the plan:\nnone\n1. Do the thing.") == [SubTask("Do the thing.")]


def test_overflow_folds_into_the_last_kept_task_with_its_dependencies():
    text = "1. Task one here.\n2. Task two here.\n3. [after 1] Task three here."
    assert parse_plan_lines(text, max_parts=2) == [
        SubTask("Task one here."),
        SubTask("Task two here. Task three here.", (1,)),
    ]


def test_external_references_are_checked_against_ref_limit_only():
    text = "[after 2] Fix the Rust quicksort from step 2.\n[after 7] Something else to do."
    assert [t.depends_on for t in parse_plan_lines(text, ref_limit=3)] == [(2,), ()]


class _Chat:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def invoke(self, text):
        self.prompts.append(text)
        return self.reply


class _WeakRouter(RouterModel):
    """Every prompt scores 0.3, so triage always asks to orchestrate."""

    kind = "weak_text"
    can_route_text = True

    def __init__(self):
        super().__init__(["a", "b"], name="weak")

    def predict_scores(self, query_ids):
        return pd.DataFrame(0.3, index=[str(q) for q in query_ids], columns=self._model_ids)

    def predict_scores_text(self, texts, *, encoder=None):
        return pd.DataFrame(0.3, index=range(len(texts)), columns=self._model_ids)


def _agent(decomposer):
    return AgenticRouter(_WeakRouter(), ClientRegistry.echo(["a", "b"]), decomposer=decomposer)


def test_llm_decomposer_plan_parses_dependencies_with_the_plan_prompt():
    chat = _Chat("1. Implement quicksort in Rust.\n2. [after 1] Test the Rust quicksort from step 1.")
    dec = LLMDecomposer(chat, plan_prompt="PLAN {prompt}")
    assert dec.plan("do it") == [SubTask("Implement quicksort in Rust."),
                                 SubTask("Test the Rust quicksort from step 1.", (1,))]
    assert chat.prompts == ["PLAN do it"]


def test_llm_decomposer_plan_falls_back_on_a_failed_call():
    class Broken:
        def invoke(self, text):
            raise RuntimeError("down")

    dec = LLMDecomposer(Broken(), fallback=lambda p: ["part one here", "part two here"])
    with pytest.warns(UserWarning, match="call failed"):
        assert dec.plan("x") == [SubTask("part one here"), SubTask("part two here")]


def test_decompose_plan_gates_atomic_prompts_like_decompose():
    chat = _Chat("1. A thing to do.\n2. Another thing to do.")
    agent = _agent(LLMDecomposer(chat))
    assert agent.decompose_plan("What is the capital of France?", 0) == [
        SubTask("What is the capital of France?")]
    assert agent.decompose_plan("Explain the cache miss.", 1) == [SubTask("Explain the cache miss.")]
    assert chat.prompts == []


def test_decompose_plan_calls_the_planner_for_a_multipart_prompt():
    chat = _Chat("1. Explain quicksort.\n2. [after 1] Implement the quicksort from step 1 in Rust.")
    tasks = _agent(LLMDecomposer(chat)).decompose_plan(
        "Explain quicksort and then implement it in Rust", 0)
    assert tasks[1].depends_on == (1,) and len(chat.prompts) == 1


def test_decompose_plan_with_a_plain_decomposer_chains_only_when_sequential():
    agent = _agent(lambda p: ["one part", "two part", "three part"])
    assert [t.depends_on for t in agent.decompose_plan("x", 0)] == [(), (), ()]
    assert [t.depends_on for t in agent.decompose_plan("x", 0, sequential=True)] == [(), (1,), (2,)]


def test_shipped_plan_prompt_formats():
    from router.agentic.schemas import default_prompts
    assert "[after N]" in default_prompts()["plan"].format(prompt="p")


def test_planner_prompt_setting_reaches_the_decomposer():
    from router.agentic.schemas import AgenticRouterSettings, load_settings

    base = load_settings()
    settings = AgenticRouterSettings.model_validate({
        **base.model_dump(), "warmup": False, "orchestrator": "plan",
        "prompts": {**base.prompts, "my_plan": "MY PLAN {prompt}"},
        "decomposer": {"kind": "routed"}, "plan": {"planner_prompt": "my_plan"},
    })
    agent = AgenticRouter.from_settings(settings, router=_WeakRouter())
    assert agent.decomposer.plan_prompt == "MY PLAN {prompt}"


def test_unknown_planner_prompt_is_rejected_for_the_plan_orchestrator():
    from router.agentic.schemas import AgenticRouterSettings, load_settings

    with pytest.raises(ValueError, match="planner_prompt"):
        AgenticRouterSettings.model_validate({
            **load_settings().model_dump(), "orchestrator": "plan",
            "plan": {"planner_prompt": "nope"}})


def test_llm_decomposer_plan_with_an_empty_fallback_keeps_the_prompt():
    class Broken:
        def invoke(self, text):
            raise RuntimeError("down")

    dec = LLMDecomposer(Broken(), fallback=lambda p: [])
    with pytest.warns(UserWarning, match="call failed"):
        assert dec.plan("x") == [SubTask("x")]
