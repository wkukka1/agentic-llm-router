"""``CostModel`` -- predicted cost only (docs/architecture.md, router/llm/cost.py)."""

from __future__ import annotations

import pytest

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


def test_price_tokens_unknown_is_never_free():
    from router.llm.cost import price_tokens

    assert price_tokens(10, 2, 1.0, 5.0) == 20.0
    assert price_tokens(None, None, 1.0, 5.0) is None          # no usage at all
    assert price_tokens(10, 2, None, 5.0) is None              # used side unpriced
    assert price_tokens(0, 2, None, 5.0) == 10.0               # unused side may be unpriced


def test_estimate_call_cost_with_explicit_tokens():
    from router.llm.cost import estimate_call_cost

    model = LLMProfile(name="m", provider="x", model_id="m",
                       input_cost_per_token=1.0, output_cost_per_token=2.0)
    assert estimate_call_cost(model, input_tokens=3, output_tokens=4) == (11.0, "profile")
    assert estimate_call_cost(model, 40, output_tokens=0) == (10.0, "profile")   # 40 chars -> 10 tokens
