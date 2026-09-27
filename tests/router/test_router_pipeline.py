"""``Router`` (router.router) -- the one piece of the production
router-design scaffolding that composes entirely out of already-working code
(PromptDecomposer, RouterModel.route_text, DefaultRoutingPolicy)."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from router.constraints import OptimizationObjective, RoutingConstraints
from router.context import RoutingRequest
from router.decision import DecisionDestination
from router.llm.profile import LLMProfile
from router.llm.registry import LLMRegistry
from router.routing import RouterModel
from router.routing.base import UnsupportedCandidatePolicy
from router.router import Router

POOL = ["fast-small", "big-strong", "coder"]


class FakeTextRouter(RouterModel):
    """Constant per-model scores, 'big-strong' always best -- enough to prove
    Router actually calls through to route_text and interprets the
    result, not a stand-in for real routing quality."""

    kind = "fake_text"
    can_route_text = True

    def __init__(self, model_ids=POOL):
        super().__init__(model_ids, name="fake")

    def predict_scores(self, query_ids):
        ids = [str(q) for q in query_ids]
        return pd.DataFrame([[0.5] * len(self._model_ids) for _ in ids],
                            index=ids, columns=self._model_ids)

    def predict_scores_text(self, texts, *, encoder=None):
        row = {"fast-small": 0.4, "big-strong": 0.9, "coder": 0.6}
        return pd.DataFrame([row for _ in texts],
                            index=list(range(len(texts))), columns=self._model_ids)


def _registry() -> LLMRegistry:
    reg = LLMRegistry()
    for mid in POOL:
        reg.register(LLMProfile(name=mid, provider="openai", model_id=mid))
    return reg


def test_route_picks_the_highest_utility_model():
    pipeline = Router(FakeTextRouter(), llm_registry=_registry())
    decision = pipeline.route(RoutingRequest(request_id="r1", prompt="explain quicksort"))

    assert decision.destination is DecisionDestination.USER
    assert decision.ranked_models[0].model.model_id == "big-strong"
    assert decision.ranked_models[0].score == pytest.approx(0.9)
    # ranked descending by utility
    assert [m.score for m in decision.ranked_models] == sorted(
        (m.score for m in decision.ranked_models), reverse=True
    )


def test_route_without_a_registry_falls_back_to_bare_model_ids():
    """No LLMRegistry -- Router still runs against a plain RouterModel,
    using its string model_ids as candidates."""
    pipeline = Router(FakeTextRouter())
    decision = pipeline.route(RoutingRequest(request_id="r2", prompt="anything"))
    assert decision.ranked_models[0].model.model_id == "big-strong"


def test_required_capability_filters_out_unsupported_candidates():
    reg = LLMRegistry()
    reg.register(LLMProfile(name="fast-small", provider="openai", model_id="fast-small"))
    reg.register(LLMProfile(name="big-strong", provider="openai", model_id="big-strong",
                            capabilities=["vision"]))
    reg.register(LLMProfile(name="coder", provider="openai", model_id="coder"))

    constraints = RoutingConstraints(required_capabilities=["vision"],
                                     objective=OptimizationObjective())
    pipeline = Router(FakeTextRouter(), llm_registry=reg)
    decision = pipeline.route(
        RoutingRequest(request_id="r3", prompt="describe this image", constraints=constraints)
    )
    assert len(decision.ranked_models) == 1
    assert decision.ranked_models[0].model.model_id == "big-strong"


def test_required_capability_with_no_registry_warns_and_keeps_candidates():
    """ROUTER-003: no-registry mode can't know any candidate's capabilities.
    Silently emptying every candidate (the old behaviour) must become a
    warning plus keeping the pool, not a silent, unexplained empty result."""
    constraints = RoutingConstraints(required_capabilities=["vision"])
    pipeline = Router(FakeTextRouter(), llm_registry=None)
    with pytest.warns(UserWarning, match="required_capabilities"):
        decision = pipeline.route(
            RoutingRequest(request_id="r4", prompt="anything", constraints=constraints)
        )
    assert len(decision.ranked_models) > 0


def test_route_warns_when_max_latency_is_set_since_no_latency_signal_exists():
    """RC-01: ModelScore.expected_latency is never populated on the live path
    (no latency source exists anywhere in the package), so a non-default
    max_latency silently filters nothing. Callers must be warned rather than
    getting a no-op."""
    pipeline = Router(FakeTextRouter(), llm_registry=_registry())
    with pytest.warns(UserWarning, match="max_latency"):
        pipeline.route(RoutingRequest(
            request_id="r5", prompt="anything",
            constraints=RoutingConstraints(max_latency=0.05),
        ))


def test_route_warns_when_latency_weight_is_set_since_no_latency_signal_exists():
    """RC-01, the soft-objective half: latency_weight always multiplies a
    permanent 0.0, so a non-default weight has zero effect on ranking."""
    pipeline = Router(FakeTextRouter(), llm_registry=_registry())
    objective = OptimizationObjective(quality_weight=1.0, latency_weight=1.0)
    with pytest.warns(UserWarning, match="latency_weight"):
        pipeline.route(RoutingRequest(
            request_id="r6", prompt="anything",
            constraints=RoutingConstraints(objective=objective),
        ))


def test_route_does_not_warn_when_latency_knobs_are_left_at_default():
    pipeline = Router(FakeTextRouter(), llm_registry=_registry())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipeline.route(RoutingRequest(request_id="r7", prompt="anything"))


class _CostedTextRouter(FakeTextRouter):
    """Pool of two with a per-query USD cost vector (the scale ``Router.route``
    reports ``expected_cost`` on)."""

    def __init__(self, **kwargs):
        super().__init__(["big", "small"], **kwargs)

    @property
    def default_model_costs(self):
        return np.array([0.010, 0.001])

    def predict_scores_text(self, texts, *, encoder=None):
        return pd.DataFrame([[0.80, 0.70]] * len(texts), columns=self._model_ids)


def test_unsupported_candidate_is_costed_on_the_pool_scale():
    """ROUTER-001: an off-pool candidate kept by SCORE_WITH_PRIOR used to get its
    per-*token* price (6e-5) next to per-*query* costs, so a model the router has no
    evidence for looked ~1000x cheaper than everything else and won."""
    reg = LLMRegistry()
    for mid, out_cost in [("big", 6e-5), ("small", 6e-6), ("unseen", 6e-5)]:
        reg.register(LLMProfile(name=mid, provider="x", model_id=mid,
                                output_cost_per_token=out_cost))
    rm = _CostedTextRouter()
    rm.unsupported_policy = UnsupportedCandidatePolicy.SCORE_WITH_PRIOR

    decision = Router(rm, llm_registry=reg).route(RoutingRequest(
        request_id="r8", prompt="hi",
        constraints=RoutingConstraints(objective=OptimizationObjective(cost_weight=10.0)),
    ))

    by_id = {s.model.model_id: s for s in decision.ranked_models}
    assert by_id["unseen"].supported is False
    assert by_id["unseen"].expected_cost == pytest.approx(0.010)   # pool max, not 6e-5
    assert decision.ranked_models[0].model.model_id != "unseen"


def test_profile_price_is_used_only_when_the_router_has_no_cost_vector():
    """The per-token profile fallback stays for routers with no cost signal at all."""
    reg = LLMRegistry()
    for mid, out_cost in [("fast-small", 1.0), ("big-strong", 2.0), ("coder", 3.0)]:
        reg.register(LLMProfile(name=mid, provider="x", model_id=mid,
                                output_cost_per_token=out_cost))
    decision = Router(FakeTextRouter(), llm_registry=reg).route(
        RoutingRequest(request_id="r9", prompt="hi"))
    assert {s.model.model_id: s.expected_cost for s in decision.ranked_models} == {
        "fast-small": 1.0, "big-strong": 2.0, "coder": 3.0}


def test_decision_records_the_artifact_that_produced_it():
    """ROUTER-002: RoutingDecision.artifact_id was never set."""
    artifact = SimpleNamespace(artifact_id="run-42")
    router_model = FakeTextRouter()
    router_model.artifact = artifact
    decision = Router(router_model, llm_registry=_registry()).route(
        RoutingRequest(request_id="r10", prompt="anything"))
    assert decision.artifact_id == "run-42"


def test_decision_artifact_id_is_none_without_an_artifact():
    decision = Router(FakeTextRouter(), llm_registry=_registry()).route(
        RoutingRequest(request_id="r11", prompt="anything"))
    assert decision.artifact_id is None


def test_cost_weight_can_change_the_winner():
    reg = LLMRegistry()
    reg.register(LLMProfile(name="fast-small", provider="openai", model_id="fast-small",
                            output_cost_per_token=0.0))
    reg.register(LLMProfile(name="big-strong", provider="openai", model_id="big-strong",
                            output_cost_per_token=10.0))
    reg.register(LLMProfile(name="coder", provider="openai", model_id="coder",
                            output_cost_per_token=0.0))

    expensive_objective = OptimizationObjective(quality_weight=1.0, cost_weight=1.0)
    pipeline = Router(FakeTextRouter(), llm_registry=reg)
    decision = pipeline.route(RoutingRequest(
        request_id="r4", prompt="anything",
        constraints=RoutingConstraints(objective=expensive_objective),
    ))
    # big-strong's quality edge (0.9 vs 0.6) is wiped out by its cost (10.0 * 1.0 weight)
    assert decision.ranked_models[0].model.model_id == "coder"
