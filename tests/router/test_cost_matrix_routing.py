"""Per-cell ``[Q, M]`` decision costs: strict shapes, alignment, and cost roles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.routing.oracle import evaluate_router, routing_evaluation
from router.nirt.routing_decision import cost_aware_utility, routing_decision
from router.routing.routers import MatrixRouter

PRED = np.array([[0.9, 0.8], [0.9, 0.8], [0.9, 0.8]])


@pytest.mark.parametrize("shape", [(3,), (2, 3), (3, 3), (1, 2, 1), ()])
def test_cost_aware_utility_rejects_bad_shapes(shape):
    with pytest.raises(ValueError, match="model_costs has shape"):
        cost_aware_utility(PRED, lam=1.0, model_costs=np.ones(shape))


def test_cost_aware_utility_rejects_wrong_width():
    with pytest.raises(ValueError):
        cost_aware_utility(PRED, lam=1.0, model_costs=np.ones((3, 3)))


def test_square_one_d_costs_are_per_model():
    pred = np.zeros((2, 2))
    util = cost_aware_utility(pred, lam=1.0, model_costs=np.array([1.0, 2.0]))
    assert np.array_equal(util, -np.array([[1.0, 2.0], [1.0, 2.0]]))


def test_per_model_behaviour_unchanged_and_per_cell_changes_rows():
    per_model = routing_decision(PRED, lam=1.0, model_costs=np.array([0.05, 0.0]))
    assert per_model.tolist() == [0, 0, 0]
    per_cell = np.array([[0.05, 0.0], [0.5, 0.0], [0.0, 0.5]])
    assert routing_decision(PRED, lam=1.0, model_costs=per_cell).tolist() == [0, 1, 0]


def _router(cost_matrix=None):
    scores = pd.DataFrame(PRED, index=["q0", "q1", "q2"], columns=["a", "b"])
    return MatrixRouter(scores, model_costs=[0.05, 0.0], cost_matrix=cost_matrix)


def test_router_uses_predicted_costs_over_default():
    cm = pd.DataFrame([[0.05, 0.0], [0.5, 0.0], [0.0, 0.5]], index=["q0", "q1", "q2"],
                      columns=["a", "b"])
    res = _router(cm).route(["q0", "q1", "q2"], lam=1.0)
    assert res.selected_model_ids == ["a", "b", "a"]
    assert res.model_costs.shape == (3, 2)
    # explicit per-model costs still win
    res2 = _router(cm).route(["q0", "q1", "q2"], lam=1.0, model_costs=[0.05, 0.0])
    assert res2.selected_model_ids == ["a", "a", "a"]


def test_cost_frame_is_aligned_by_label():
    cm = pd.DataFrame([[0.0, 0.5], [0.5, 0.0]], index=["q2", "q1"], columns=["b", "a"])
    cm = cm.reindex(index=["q2", "q1", "q0"]).fillna(0.0)
    # q2: a=0.5, b=0 -> a; q1: a=0, b=0.5 -> a (0.9-0 vs 0.8-0.5); q0: both 0 -> a
    res = _router(cm).route(["q0", "q1", "q2"], lam=1.0)
    assert res.model_costs[2].tolist() == [0.5, 0.0]
    assert res.model_costs[1].tolist() == [0.0, 0.5]


@pytest.mark.parametrize("drop", ["row", "col", "nan"])
def test_incomplete_predicted_costs_raise(drop):
    cm = pd.DataFrame(np.full((3, 2), 0.1), index=["q0", "q1", "q2"], columns=["a", "b"])
    if drop == "row":
        cm = cm.drop(index="q1")
    elif drop == "col":
        cm = cm.drop(columns="b")
    else:
        cm.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="cost matrix"):
        _router(cm).route(["q0", "q1", "q2"], lam=1.0)


def test_quality_only_routing_ignores_predicted_costs():
    cm = pd.DataFrame(np.full((1, 2), 0.1), index=["q0"], columns=["a", "b"])  # incomplete
    assert _router(cm).route(["q0", "q1", "q2"]).selected_model_ids == ["a", "a", "a"]


def test_scoring_uses_true_cost_not_the_decision_cost():
    true = np.array([[1.0, 0.9], [1.0, 0.9]])
    true_cost = np.array([[1.0, 0.0], [1.0, 0.0]])
    wrong_guess = np.array([[0.0, 1.0], [0.0, 1.0]])     # router thinks `a` is free
    out = routing_evaluation(PRED[:2], true, true_cost, ["a", "b"], lam=1.0,
                             model_costs=wrong_guess)
    ca = out["cost_aware"]
    assert out["selected_model_mix"] == {"a": 2}                      # decided on its guess
    assert ca["mean_selected_utility"] == pytest.approx(1.0 - 1.0)    # scored on true cost
    assert ca["mean_cost_aware_oracle_utility"] == pytest.approx(0.9)  # oracle on true cost
    assert ca["cost_aware_oracle_model_mix"] == {"b": 2}


def test_evaluate_router_selects_with_predicted_costs():
    cm = pd.DataFrame([[0.05, 0.0], [0.5, 0.0], [0.0, 0.5]], index=["q0", "q1", "q2"],
                      columns=["a", "b"])
    true_df = pd.DataFrame([[1.0, 0.0]] * 3, index=["q0", "q1", "q2"], columns=["a", "b"])
    cost_df = pd.DataFrame([[0.1, 0.1]] * 3, index=["q0", "q1", "q2"], columns=["a", "b"])
    out = evaluate_router(_router(cm), true_df, cost_df, lam=1.0)
    assert out["selected_model_mix"] == {"a": 2, "b": 1}
