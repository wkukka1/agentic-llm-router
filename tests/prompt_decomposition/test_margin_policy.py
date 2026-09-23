"""`MarginRoutingPolicy`: keep the ranking, change the selection.

The behaviour under test is the one the measurements called for -- take the
cheapest candidate the ranker rates within a margin of its best, rather than
the top of the ranking, which on RouterBench meant gpt-4 for 98.6% of queries
and scored no better than having no router at all.
"""

from __future__ import annotations

import pytest

from decompose.signals import PromptSignals
from router.constraints import OptimizationObjective, RoutingConstraints
from router.context import RoutingContext, RoutingRequest
from router.decision import DecisionDestination, ModelScore
from router.llm.profile import LLMProfile
from router.policy import DefaultRoutingPolicy, MarginRoutingPolicy


def _profile(name: str) -> LLMProfile:
    return LLMProfile(name=name, provider="test", model_id=name, version="1")


def _context(quality_only: bool = True) -> RoutingContext:
    objective = OptimizationObjective(quality_weight=1.0, cost_weight=0.0 if quality_only else 1.0)
    request = RoutingRequest(request_id="r1", prompt="p",
                             constraints=RoutingConstraints(objective=objective))
    return RoutingContext(request=request, signals=PromptSignals())


def _scores() -> list[ModelScore]:
    # the strong model is a hair better and ten times the price
    return [
        ModelScore(model=_profile("strong"), score=0.80, expected_quality=0.80,
                   expected_cost=1.00, confidence=0.9),
        ModelScore(model=_profile("mid"), score=0.79, expected_quality=0.79,
                   expected_cost=0.10, confidence=0.8),
        ModelScore(model=_profile("cheap"), score=0.60, expected_quality=0.60,
                   expected_cost=0.01, confidence=0.7),
    ]


class TestSelection:
    def test_it_takes_the_cheapest_model_within_the_margin(self):
        d = MarginRoutingPolicy(margin=0.02).decide(_context(), _scores())
        assert d.selected_model_id == "mid"          # 0.79 is within 0.02 of 0.80, and 10x cheaper
        assert d.ranked_models[0].model.model_id == "strong"   # ranking is untouched

    def test_a_zero_margin_reproduces_argmax(self):
        """The knob has to contain the old behaviour, or it cannot be rolled back."""
        scores = _scores()
        margin = MarginRoutingPolicy(margin=0.0).decide(_context(), scores)
        default = DefaultRoutingPolicy().decide(_context(), scores)
        assert margin.selected_model_id == default.ranked_models[0].model.model_id == "strong"

    def test_a_wide_margin_reaches_the_cheapest_candidate(self):
        d = MarginRoutingPolicy(margin=0.5).decide(_context(), _scores())
        assert d.selected_model_id == "cheap"

    def test_a_model_outside_the_margin_is_never_taken(self):
        """`cheap` is 0.20 below the best and must not be picked at margin 0.05,
        however little it costs."""
        d = MarginRoutingPolicy(margin=0.05).decide(_context(), _scores())
        assert d.selected_model_id == "mid"

    def test_ties_on_cost_keep_the_better_ranked_model(self):
        scores = [
            ModelScore(model=_profile("a"), score=0.80, expected_quality=0.80, expected_cost=0.5),
            ModelScore(model=_profile("b"), score=0.79, expected_quality=0.79, expected_cost=0.5),
        ]
        assert MarginRoutingPolicy(margin=0.05).decide(_context(), scores).selected_model_id == "a"

    def test_the_decision_says_what_it_did(self):
        d = MarginRoutingPolicy(margin=0.02).decide(_context(), _scores())
        assert "cheapest within" in d.reason
        assert d.policy_version == "1"

    def test_no_candidates_is_not_a_crash(self):
        d = MarginRoutingPolicy().decide(_context(), [])
        assert d.selected_model_id is None and d.confidence == 0.0

    def test_a_negative_margin_is_refused(self):
        with pytest.raises(ValueError, match="margin must be >= 0"):
            MarginRoutingPolicy(margin=-0.1)


class TestSelectedAccessor:
    def test_selected_returns_the_chosen_score_not_the_top_one(self):
        """The whole point of the field: the choice cannot be read off the
        ranking, because it is deliberately not rank 1."""
        d = MarginRoutingPolicy(margin=0.02).decide(_context(), _scores())
        assert d.selected().model.model_id == "mid"

    def test_selected_falls_back_to_the_ranking_when_a_policy_sets_nothing(self):
        d = DefaultRoutingPolicy().decide(_context(), _scores())
        assert d.selected_model_id is None
        assert d.selected().model.model_id == "strong"


class TestEscalation:
    def test_a_hopeless_best_candidate_goes_to_the_orchestrator(self):
        """`escalation_threshold` exists on the base class and the default
        policy ignores it; this one does not."""
        weak = [ModelScore(model=_profile("cheap"), score=0.2, expected_quality=0.2,
                           expected_cost=0.01)]
        policy = MarginRoutingPolicy(margin=0.02, escalation_threshold=0.5)
        assert policy.select_destination(_context(), weak) is DecisionDestination.PARENT_ORCHESTRATOR

    def test_a_good_candidate_answers_the_user(self):
        policy = MarginRoutingPolicy(margin=0.02, escalation_threshold=0.5)
        assert policy.select_destination(_context(), _scores()) is DecisionDestination.USER

    def test_no_candidates_still_returns_to_the_user(self):
        policy = MarginRoutingPolicy(margin=0.02, escalation_threshold=0.5)
        assert policy.select_destination(_context(), []) is DecisionDestination.USER


def test_cost_weighted_objectives_still_work():
    """With `cost_weight > 0` the utility already penalises price; the margin
    then applies to that utility, not to raw quality."""
    d = MarginRoutingPolicy(margin=0.0).decide(_context(quality_only=False), _scores())
    assert d.selected_model_id in {"mid", "cheap"}      # never the dear one at cost_weight=1
