import numpy as np
import pytest

from router.agent import AgenticRouter, PolicyThresholds
from router.agent.evaluate import evaluate_routing
from router.agent.policy import fit_thresholds_to_quantiles
from router.agent.tasktype import infer_task_type

PROMPTS = [
    "What year did the Berlin Wall fall?",
    "Who wrote Moby Dick?",
    "Prove that the halting problem is undecidable and then also explain the diagonalisation.",
    "Refactor the parser for testability\nThen also profile the tokenizer",
    "Translate this sentence into French.",
]


def domain_fn(_prompt):
    return "cs_general", 0.9, {"cs_general": 0.9, "technology": 0.05}


def test_routing_report_covers_every_prompt():
    router = AgenticRouter(domain_fn=domain_fn, task_type_fn=infer_task_type)
    report = evaluate_routing(router, PROMPTS)
    assert report.summary["n"] == len(PROMPTS)
    assert len(report.per_prompt) == len(PROMPTS)
    assert set(report.summary["action_share"]) <= {"direct", "orchestrate", "clarify"}


def test_routing_costs_land_between_the_two_baselines():
    """A router that beats neither baseline is not doing anything useful."""
    router = AgenticRouter(domain_fn=domain_fn, task_type_fn=infer_task_type)
    s = evaluate_routing(router, PROMPTS).summary
    assert s["baseline_cheapest"] < s["baseline_best"]
    assert s["mean_cost"] < s["baseline_best"], "routing must cost less than always-best"


def test_report_renders_without_error():
    router = AgenticRouter(domain_fn=domain_fn, task_type_fn=infer_task_type)
    text = evaluate_routing(router, PROMPTS).render()
    assert "Routing policy evaluation" in text
    assert "Model mix" in text


def test_quantile_thresholds_are_ordered():
    rng = np.random.default_rng(0)
    difficulties = list(np.clip(rng.normal(0.23, 0.08, 2000), 0, 1))
    t = fit_thresholds_to_quantiles(difficulties)
    assert t.decompose_difficulty < t.strong_difficulty < t.orchestrate_difficulty


def test_quantile_thresholds_hit_the_requested_traffic_share():
    """strong_quantile=0.7 must mean 'the hardest ~30% escalates'."""
    rng = np.random.default_rng(1)
    difficulties = list(np.clip(rng.normal(0.23, 0.08, 5000), 0, 1))
    t = fit_thresholds_to_quantiles(difficulties, strong_quantile=0.7)
    share = float(np.mean(np.asarray(difficulties) >= t.strong_difficulty))
    assert share == pytest.approx(0.30, abs=0.02)


def test_quantile_thresholds_fall_back_when_given_no_samples():
    base = PolicyThresholds(strong_difficulty=0.55)
    assert fit_thresholds_to_quantiles([], base=base).strong_difficulty == 0.55


def test_calibration_revives_the_middle_tier():
    """Regression: uncalibrated thresholds left the strong tier unused."""
    rng = np.random.default_rng(2)
    # A compressed difficulty distribution, as the heuristic estimator produces.
    difficulties = list(np.clip(rng.normal(0.23, 0.08, 3000), 0, 1))
    default = PolicyThresholds()
    assert np.mean(np.asarray(difficulties) >= default.strong_difficulty) < 0.01

    calibrated = fit_thresholds_to_quantiles(difficulties, strong_quantile=0.70)
    share = float(np.mean(np.asarray(difficulties) >= calibrated.strong_difficulty))
    assert share > 0.25
