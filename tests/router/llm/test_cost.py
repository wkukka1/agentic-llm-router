"""``CostModel`` -- predicted cost only (docs/architecture.md, router/llm/cost.py)."""

from __future__ import annotations

import pytest

from router.constraints import RoutingConstraints
from router.context import RoutingContext, RoutingRequest
from router.llm.cost import CostModel
from router.llm.profile import LLMProfile


def _profile(model_id="m", input_cost=0.001, output_cost=0.002) -> LLMProfile:
    return LLMProfile(name=model_id, provider="openai", model_id=model_id,
                      input_cost_per_token=input_cost, output_cost_per_token=output_cost)


def test_predict_cost_multiplies_token_counts_by_profile_prices():
    model = CostModel()
    cost = model.predict_cost(_profile(input_cost=0.001, output_cost=0.002), 100, 50)
    assert cost == pytest.approx(100 * 0.001 + 50 * 0.002)


def test_predict_output_tokens_uses_the_prior_when_present():
    model = CostModel(output_token_priors={"m": 500})
    assert model.predict_output_tokens(_profile(), input_tokens=10) == 500


def test_predict_output_tokens_falls_back_for_an_unknown_model():
    model = CostModel(output_token_priors={"other": 500})
    assert model.predict_output_tokens(_profile(), input_tokens=10) > 0


def test_predict_cost_for_context_composes_output_tokens_and_cost():
    model = CostModel(output_token_priors={"m": 100})
    profile = _profile(input_cost=0.001, output_cost=0.002)
    context = RoutingContext(
        request=RoutingRequest(request_id="r", prompt="x" * 40,
                               constraints=RoutingConstraints()),
        signals=None,
    )
    cost = model.predict_cost_for_context(profile, context)
    # 40 chars / 4 chars-per-token = 10 input tokens; prior gives 100 output tokens
    assert cost == pytest.approx(10 * 0.001 + 100 * 0.002)


def test_predict_cost_for_context_never_zeroes_input_tokens_on_a_short_prompt():
    model = CostModel()
    profile = _profile()
    context = RoutingContext(
        request=RoutingRequest(request_id="r", prompt="hi", constraints=RoutingConstraints()),
        signals=None,
    )
    # even a tiny prompt must still cost something, not divide-to-zero
    assert model.predict_cost_for_context(profile, context) > 0
