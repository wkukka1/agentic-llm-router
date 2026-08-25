import json

import pytest

from router.agent import Action, AgentConfig, AgenticRouter, PolicyThresholds, Session
from router.agent.backends import HeuristicBackend, _extract_json
from router.agent.contracts import SubQuery
from router.agent.difficulty import HeuristicDifficulty
from router.agent.orchestrator import Orchestrator


def domain_fn(_prompt):
    return "cs_general", 0.9, {"cs_general": 0.9, "technology": 0.05}


@pytest.fixture
def router():
    return AgenticRouter(domain_fn=domain_fn, task_type_fn=lambda _: "code")


def test_underspecified_prompt_is_returned_to_the_user(router):
    result = router.route("write me something")
    assert result.decision.action is Action.CLARIFY
    assert result.needs_user_input
    assert result.clarifying_questions
    assert result.answer is None


def test_empty_prompt_is_gated(router):
    assert router.route("   ").decision.action is Action.CLARIFY


def test_clear_prompt_is_routed_not_gated(router):
    result = router.route("What is the time complexity of merge sort on a linked list?")
    assert result.decision.action is not Action.CLARIFY
    assert result.decision.model is not None


def test_gate_does_not_re_interrogate_on_a_follow_up(router):
    session = Session()
    router.route("Explain how B-trees keep themselves balanced on insertion.", session)
    result = router.route("now do it for red-black trees", session)
    assert result.decision.action is not Action.CLARIFY, "history supplies the missing referent"


def test_session_context_is_carried_into_the_next_turn():
    session = Session()
    session.history = ["first thing", "second thing"]
    contextualized = session.contextualize("third thing", max_history=1)
    assert "[previous] second thing" in contextualized
    assert "[current] third thing" in contextualized
    assert "first thing" not in contextualized


def test_multi_part_prompt_produces_a_routed_sub_query_plan(router):
    prompt = ("1. Refactor the parser for testability\n"
              "2. Then also profile the tokenizer and explain the bottleneck\n"
              "3. Write property tests for the result")
    result = router.route(prompt)
    assert result.decision.action is Action.ORCHESTRATE
    assert len(result.sub_queries) == 3
    assert all(s.assigned_model is not None for s in result.sub_queries)
    assert result.decision.estimated_cost > 0


def test_plan_mode_does_not_fabricate_an_answer(router):
    """Plan mode must return no answer rather than placeholder text."""
    result = router.route("Design a consistent-hashing scheme and then also justify the replica count.")
    assert result.answer is None


def test_budget_cap_downgrades_instead_of_failing():
    router = AgenticRouter(
        domain_fn=domain_fn,
        task_type_fn=lambda _: "code",
        config=AgentConfig(max_cost_per_turn=0.0001),
    )
    result = router.route("1. Do the hard thing\n2. Then also do the other hard thing")
    assert result.decision.tier.value == "weak"
    assert any("over budget" in r for r in result.decision.rationale)
    assert result.sub_queries == []


def test_session_accumulates_cost_across_turns(router):
    session = Session()
    router.route("What is a monad in category theory?", session)
    first = session.total_cost
    router.route("Explain the free monad construction in detail.", session)
    assert session.total_cost > first
    assert len(session.turns) == 2


def test_decision_explain_mentions_every_reason(router):
    result = router.route("Explain how a B-tree splits a full node during insertion.")
    text = result.decision.explain()
    assert result.decision.model in text
    assert len(result.decision.rationale) >= 2


# -- component-level ---------------------------------------------------


def test_difficulty_ordering_is_sensible():
    estimator = HeuristicDifficulty()
    easy = estimator.estimate("What is 2+2?", domain="language", task_type="factual_qa", n_tokens=5)
    hard = estimator.estimate(
        "Derive the bound, then also prove it is tight and analyze the trade-off.",
        domain="cs_general", task_type="math", n_tokens=400,
    )
    assert 0.0 <= easy < hard <= 1.0


def test_difficulty_length_term_saturates():
    estimator = HeuristicDifficulty()
    assert estimator.length_term(0) == 0.0
    assert estimator.length_term(10_000) == pytest.approx(1.0, abs=1e-3)


def test_extract_json_handles_fences_and_surrounding_prose():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('Sure! [1, 2, 3] hope that helps') == [1, 2, 3]
    assert _extract_json("no json at all") is None


def test_decomposition_drops_forward_dependencies():
    """A dependency on a later sub-query would deadlock the executor."""
    class FixedBackend:
        requires_llm = False

        def complete(self, system, user, *, max_tokens=1024):
            return json.dumps([
                {"text": "first", "depends_on": [1]},   # forward -> must be dropped
                {"text": "second", "depends_on": [0]},  # backward -> kept
            ])

    router = AgenticRouter(domain_fn=domain_fn, backend=FixedBackend())
    subs = router.orchestrator.decompose("anything")
    assert subs[0].depends_on == []
    assert subs[1].depends_on == [0]


def test_execute_runs_sub_queries_in_dependency_order():
    orchestrator = Orchestrator(HeuristicBackend(), None, None)
    subs = [
        SubQuery(index=0, text="a", depends_on=[1]),
        SubQuery(index=1, text="b"),
    ]
    order = [s.index for s in orchestrator._topological_order(subs)]
    assert order == [1, 0]


def test_topological_order_survives_a_cycle():
    orchestrator = Orchestrator(HeuristicBackend(), None, None)
    subs = [SubQuery(index=0, text="a", depends_on=[1]), SubQuery(index=1, text="b", depends_on=[0])]
    assert len(orchestrator._topological_order(subs)) == 2


def test_dependent_sub_query_receives_the_earlier_answer():
    from router.agent.orchestrator import OrchestrationPlan

    orchestrator = Orchestrator(HeuristicBackend(), None, None)
    plan = OrchestrationPlan(
        sub_queries=[SubQuery(index=0, text="a"), SubQuery(index=1, text="b", depends_on=[0])],
        total_estimated_cost=0.0,
    )
    seen = {}

    def answer_fn(sub, context):
        seen[sub.index] = context
        return f"answer-{sub.index}"

    orchestrator.execute(plan, answer_fn)
    assert seen[0] == ""
    assert "answer-0" in seen[1]


def test_gate_failure_never_blocks_a_turn():
    class BrokenBackend:
        requires_llm = True

        def complete(self, system, user, *, max_tokens=1024):
            raise RuntimeError("provider down")

    router = AgenticRouter(domain_fn=domain_fn, backend=BrokenBackend())
    result = router.route("Explain the CAP theorem and its practical consequences.")
    assert result.decision.action is not Action.CLARIFY


def test_self_contained_prompt_does_not_inherit_history():
    """History must not pollute the classifier input for a standalone prompt."""
    session = Session()
    session.history = ["Explain how B-trees rebalance on insertion."]
    assert not session.needs_context("What year did the Berlin Wall fall?")
    assert session.needs_context("now explain why it happened")
    assert session.needs_context("do it in Rust")


def test_task_type_survives_a_follow_up_turn():
    """Regression: prepending '[previous] ...' broke the anchored task-type rules."""
    from router.agent.tasktype import infer_task_type

    real = AgenticRouter(domain_fn=domain_fn, task_type_fn=infer_task_type)
    session = Session()
    real.route("Explain how B-trees rebalance on insertion.", session)
    result = real.route("What year did the Berlin Wall fall?", session)
    assert result.decision.signals.task_type == "factual_qa"


class FakeGeneratingBackend:
    """A backend that can answer, so the execute path can be tested offline."""

    requires_llm = False
    can_generate = True

    def __init__(self):
        self.calls = []

    def complete(self, system, user, *, max_tokens=1024):
        self.calls.append((system, user))
        if "decompose" in system.lower():
            return json.dumps([
                {"text": "part one", "depends_on": []},
                {"text": "part two", "depends_on": [0]},
            ])
        if "sufficient" in system.lower() or "specified well enough" in system.lower():
            return json.dumps({"sufficient": True, "missing": [], "reason": "ok"})
        if "synthesise" in system.lower() or "synthesize" in system.lower():
            return "SYNTHESIZED"
        return f"ANSWER({user.strip()[:24]})"


def test_execute_mode_refuses_a_backend_that_cannot_generate():
    """Regression: this used to return the gate's JSON as if it were an answer."""
    with pytest.raises(ValueError, match="cannot generate answers"):
        AgenticRouter(domain_fn=domain_fn, config=AgentConfig(mode="execute"))


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        AgenticRouter(domain_fn=domain_fn, config=AgentConfig(mode="dry-run"))


def test_execute_mode_returns_a_real_answer_on_the_direct_path():
    backend = FakeGeneratingBackend()
    router = AgenticRouter(
        domain_fn=domain_fn, backend=backend, config=AgentConfig(mode="execute"),
    )
    result = router.route("Explain how a B-tree splits a full node during insertion.")
    assert result.answer is not None
    assert result.answer.startswith("ANSWER(")


def _executing_router(backend):
    """Force the orchestrated path so the test exercises execution mechanics,
    not whichever band the heuristic difficulty estimator happens to land in."""
    return AgenticRouter(
        domain_fn=domain_fn,
        backend=backend,
        thresholds=PolicyThresholds(decompose_difficulty=0.0),
        config=AgentConfig(mode="execute"),
    )


def test_execute_mode_runs_and_synthesizes_the_orchestrated_path():
    backend = FakeGeneratingBackend()
    router = _executing_router(backend)
    result = router.route(
        "1. Do the first hard thing\n2. Then also do the second hard thing"
    )
    assert result.decision.action is Action.ORCHESTRATE
    assert all(sub.answer is not None for sub in result.sub_queries)
    assert result.answer == "SYNTHESIZED"


def test_execute_mode_feeds_earlier_answers_into_dependent_sub_queries():
    backend = FakeGeneratingBackend()
    router = _executing_router(backend)
    router.route("1. First step\n2. Then also the dependent step")
    # The dependent sub-query's user message must carry the earlier answer.
    answering = [user for system, user in backend.calls if "You are" in system]
    assert any("[answer to sub-query 0]" in user for user in answering)
