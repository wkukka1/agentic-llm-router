"""ROUTERMODEL-004 / ROUTERMODEL-002: every RouterModel subclass must accept
cost_model=/artifact= at direct construction (not just through
RouterModelFactory), and a set cost_model must actually change
default_model_costs instead of being silently ignored."""

from __future__ import annotations

import pandas as pd
import pytest

from router.llm.cost import CostModel
from router.routing import MatrixRouter, RandomRouter
from router.routing.routers import KNNRouter, MLPRouter, NIRTRouter

_IDS = ["a", "b"]

_CONSTRUCTORS = {
    "nirt": lambda **kw: NIRTRouter(object(), {"a": 0, "b": 1}, data=object(), **kw),
    "knn": lambda **kw: KNNRouter(data=object(), model_ids=_IDS, **kw),
    "mlp": lambda **kw: MLPRouter(object(), _IDS, data=object(), **kw),
    "matrix": lambda **kw: MatrixRouter(pd.DataFrame([[0.1, 0.2]], index=["q"], columns=_IDS), **kw),
    "random": lambda **kw: RandomRouter(_IDS, **kw),
}


@pytest.mark.parametrize("kind", sorted(_CONSTRUCTORS))
def test_router_accepts_cost_model_and_artifact(kind):
    sentinel_cost_model, sentinel_artifact = object(), object()
    r = _CONSTRUCTORS[kind](cost_model=sentinel_cost_model, artifact=sentinel_artifact)
    assert r.cost_model is sentinel_cost_model
    assert r.artifact is sentinel_artifact


def test_default_model_costs_uses_the_injected_cost_model_when_set():
    """ROUTERMODEL-002: a CostModel passed at construction must change the
    router's cost vector, not be silently ignored in favour of the
    train-mean cost."""
    cost_model = CostModel(output_token_priors={"a": 0.02, "b": 0.05})
    r = KNNRouter(data=object(), model_ids=_IDS, cost_model=cost_model)
    costs = r.default_model_costs
    assert list(costs) == [0.02, 0.05]


def test_default_model_costs_is_none_cost_model_by_default():
    r = KNNRouter(data=object(), model_ids=_IDS)
    assert r.cost_model is None
