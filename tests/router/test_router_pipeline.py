"""``RoutingPipeline`` (router.router) -- the one piece of the production
router-design scaffolding that composes entirely out of already-working code
(PromptDecomposer, Router.route_text, DefaultRoutingPolicy). Everything else
under router.execution / router.llm / router.tools is exercised only by
import (see test_scaffolding_imports.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from router.constraints import OptimizationObjective, RoutingConstraints
from router.context import RoutingRequest
from router.decision import DecisionDestination
from router.llm.profile import LLMProfile
from router.llm.registry import LLMRegistry
from router.routing import Router
from router.router import RoutingPipeline

POOL = ["fast-small", "big-strong", "coder"]


class FakeTextRouter(Router):
    """Constant per-model scores, 'big-strong' always best -- enough to prove
    RoutingPipeline actually calls through to route_text and interprets the
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
    pipeline = RoutingPipeline(FakeTextRouter(), llm_registry=_registry())
    decision = pipeline.route(RoutingRequest(request_id="r1", prompt="explain quicksort"))

    assert decision.destination is DecisionDestination.USER
    assert decision.ranked_models[0].model.model_id == "big-strong"
    assert decision.ranked_models[0].score == pytest.approx(0.9)
    # ranked descending by utility
    assert [m.score for m in decision.ranked_models] == sorted(
        (m.score for m in decision.ranked_models), reverse=True
    )


def test_route_without_a_registry_falls_back_to_bare_model_ids():
    """No LLMRegistry -- RoutingPipeline still runs against a plain Router,
    using its string model_ids as candidates."""
    pipeline = RoutingPipeline(FakeTextRouter())
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
    pipeline = RoutingPipeline(FakeTextRouter(), llm_registry=reg)
    decision = pipeline.route(
        RoutingRequest(request_id="r3", prompt="describe this image", constraints=constraints)
    )
    assert len(decision.ranked_models) == 1
    assert decision.ranked_models[0].model.model_id == "big-strong"


def test_cost_weight_can_change_the_winner():
    reg = LLMRegistry()
    reg.register(LLMProfile(name="fast-small", provider="openai", model_id="fast-small",
                            output_cost_per_token=0.0))
    reg.register(LLMProfile(name="big-strong", provider="openai", model_id="big-strong",
                            output_cost_per_token=10.0))
    reg.register(LLMProfile(name="coder", provider="openai", model_id="coder",
                            output_cost_per_token=0.0))

    expensive_objective = OptimizationObjective(quality_weight=1.0, cost_weight=1.0)
    pipeline = RoutingPipeline(FakeTextRouter(), llm_registry=reg)
    decision = pipeline.route(RoutingRequest(
        request_id="r4", prompt="anything",
        constraints=RoutingConstraints(objective=expensive_objective),
    ))
    # big-strong's quality edge (0.9 vs 0.6) is wiped out by its cost (10.0 * 1.0 weight)
    assert decision.ranked_models[0].model.model_id == "coder"
