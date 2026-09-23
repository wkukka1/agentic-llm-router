"""``OptimizationObjective.utility`` (router.constraints) -- the single definition
of ``u(m, i, lambda)`` that the router and offline evaluation both score against."""

from __future__ import annotations

import math

import pytest

from router.constraints import OptimizationObjective
from router.decision import ModelScore
from router.llm.profile import LLMProfile

_PROFILE = LLMProfile(name="m", provider="openai", model_id="m")


def _score(**kwargs) -> ModelScore:
    return ModelScore(model=_PROFILE, score=0.0, **kwargs)


def test_utility_combines_all_four_terms():
    objective = OptimizationObjective(
        quality_weight=1.0, cost_weight=2.0, latency_weight=3.0, risk_weight=4.0
    )
    score = _score(expected_quality=0.9, expected_cost=0.1, expected_latency=0.2, confidence=0.75)

    assert objective.utility(score) == pytest.approx(0.9 - 2.0 * 0.1 - 3.0 * 0.2 - 4.0 * 0.25)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_zero_cost_weight_ignores_a_non_finite_cost(bad):
    score = _score(expected_quality=0.8, expected_cost=bad)

    assert OptimizationObjective(quality_weight=1.0, cost_weight=0.0).utility(score) == pytest.approx(0.8)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_zero_latency_weight_ignores_a_non_finite_latency(bad):
    score = _score(expected_quality=0.8, expected_latency=bad)

    assert OptimizationObjective(quality_weight=1.0, latency_weight=0.0).utility(score) == pytest.approx(0.8)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_zero_risk_weight_ignores_a_non_finite_confidence(bad):
    score = _score(expected_quality=0.8, confidence=bad)

    assert OptimizationObjective(quality_weight=1.0, risk_weight=0.0).utility(score) == pytest.approx(0.8)


def test_a_non_finite_cost_still_poisons_the_utility_when_its_weight_is_nonzero():
    score = _score(expected_quality=0.8, expected_cost=math.inf)

    assert OptimizationObjective(quality_weight=1.0, cost_weight=1.0).utility(score) == -math.inf
