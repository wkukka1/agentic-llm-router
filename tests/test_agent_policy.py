import pytest

from router.agent import (
    Action,
    ModelPool,
    PolicyThresholds,
    RoutingPolicy,
    Signals,
    SkillRegistry,
    Tier,
)
from router.agent.skills import Skill

POOL_ROWS = [
    {"name": "weak-1", "tier": "weak", "cost_in": 1.0, "cost_out": 5.0, "quality": 0.6,
     "skills": ["calculator"], "strengths": ["language"]},
    {"name": "strong-1", "tier": "strong", "cost_in": 3.0, "cost_out": 15.0, "quality": 0.85,
     "skills": ["calculator", "code_interpreter"], "strengths": ["cs_general"]},
    {"name": "orch-1", "tier": "orchestrator", "cost_in": 5.0, "cost_out": 25.0, "quality": 0.93,
     "skills": ["calculator", "code_interpreter"]},
]


@pytest.fixture
def policy():
    skills = SkillRegistry([
        Skill(name="calculator", description="math", task_types=["math"]),
        Skill(name="code_interpreter", description="code", task_types=["code"]),
    ])
    return RoutingPolicy(ModelPool.from_dicts(POOL_ROWS), skills)


def signals(**kw):
    base = {"domain": "cs_general", "domain_confidence": 0.9, "difficulty": 0.2,
            "task_type": "other", "n_tokens": 100}
    base.update(kw)
    return Signals(**base)


def test_easy_confident_prompt_takes_the_weak_path(policy):
    decision = policy.decide("What is the capital of France?", signals(difficulty=0.1))
    assert decision.action is Action.DIRECT
    assert decision.tier is Tier.WEAK


def test_hard_prompt_takes_the_strong_path(policy):
    decision = policy.decide("Derive it.", signals(difficulty=0.6))
    assert decision.action is Action.DIRECT
    assert decision.tier is Tier.STRONG


def test_very_hard_prompt_orchestrates(policy):
    decision = policy.decide("Derive it.", signals(difficulty=0.9))
    assert decision.action is Action.ORCHESTRATE
    assert decision.orchestrator == "orch-1"


def test_multi_part_prompt_orchestrates_at_moderate_difficulty(policy):
    """Several asks need interleaving even when none is individually hard."""
    prompt = "1. Summarise the paper\n2. Then also critique the method"
    decision = policy.decide(prompt, signals(difficulty=0.5))
    assert decision.action is Action.ORCHESTRATE

    single = policy.decide("Summarise the paper", signals(difficulty=0.5))
    assert single.action is Action.DIRECT


def test_ambiguous_domain_escalates_only_above_the_difficulty_floor(policy):
    """Domain ambiguity is irrelevant when nothing about the prompt is hard."""
    trivial = policy.decide("x", signals(domain_confidence=0.3, difficulty=0.1))
    assert trivial.tier is Tier.WEAK

    nontrivial = policy.decide("x", signals(domain_confidence=0.3, difficulty=0.4))
    assert nontrivial.tier is Tier.STRONG


def test_skills_are_gated_by_what_the_selected_model_may_call(policy):
    decision = policy.decide("x", signals(task_type="code", difficulty=0.1))
    assert decision.model == "weak-1"
    assert decision.skills == [], "the weak model has no code_interpreter permission"

    strong = policy.decide("x", signals(task_type="code", difficulty=0.6))
    assert "code_interpreter" in strong.skills


def test_cost_weight_moves_selection_within_a_tier():
    skills = SkillRegistry([])
    rows = [
        {"name": "cheap", "tier": "strong", "cost_in": 1.0, "cost_out": 2.0, "quality": 0.80},
        {"name": "pricey", "tier": "strong", "cost_in": 20.0, "cost_out": 60.0, "quality": 0.90},
    ]
    pool = ModelPool.from_dicts(rows)
    quality_first = RoutingPolicy(pool, skills, PolicyThresholds(cost_weight=0.0))
    cost_first = RoutingPolicy(pool, skills, PolicyThresholds(cost_weight=3.0))
    assert quality_first.decide("x", signals(difficulty=0.6)).model == "pricey"
    assert cost_first.decide("x", signals(difficulty=0.6)).model == "cheap"


def test_orchestrator_is_eligible_for_strong_work_but_not_the_reverse():
    pool = ModelPool.from_dicts(POOL_ROWS)
    strong_names = {m.name for m in pool.candidates(Tier.STRONG, signals())}
    assert strong_names == {"strong-1", "orch-1"}
    assert {m.name for m in pool.candidates(Tier.WEAK, signals())} == {"weak-1"}


def test_pool_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate model names"):
        ModelPool.from_dicts(POOL_ROWS + [POOL_ROWS[0]])


def test_excluded_task_types_are_never_selected():
    pool = ModelPool.from_dicts([
        {"name": "no-code", "tier": "strong", "cost_in": 1.0, "cost_out": 1.0, "excludes": ["code"]},
    ])
    with pytest.raises(ValueError, match="no candidate"):
        pool.select(Tier.STRONG, signals(task_type="code"))


def test_shipped_pool_and_skills_configs_load():
    pool = ModelPool.from_yaml()
    skills = SkillRegistry.from_yaml()
    assert pool.candidates(Tier.ORCHESTRATOR, signals()), "pool needs an orchestrator"
    known = {s.name for s in skills.skills}
    for model in pool.models:
        assert set(model.skills) <= known, f"{model.name} references an unknown skill"
