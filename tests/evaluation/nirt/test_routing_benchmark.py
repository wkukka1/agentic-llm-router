"""Routing benchmark policies + summaries on tiny hand-checked pools."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.nirt.routing_benchmark import combine, per_model_mean_cost, source_picks, summarize

MODELS = ["cheap", "mid", "gpt4"]
# 3 queries x 3 models
Y = np.array([[1.0, 1.0, 1.0],
              [0.0, 1.0, 1.0],
              [0.0, 0.0, 1.0]])
C = np.array([[0.1, 0.5, 2.0],
              [0.1, 0.5, 2.0],
              [0.1, 0.5, 2.0]])


def _picks(routers=None, lams=(0.0,)):
    return source_picks(Y, C, MODELS, "gpt4", routers or {}, lams)


def test_reference_always_picks_its_column():
    y, c, is_ref = _picks()["reference"]
    assert y.tolist() == [1.0, 1.0, 1.0] and c.tolist() == [2.0, 2.0, 2.0] and is_ref.all()


def test_oracle_best_quality_breaks_ties_to_cheapest():
    y, c, _ = _picks()["oracle_best_quality"]
    assert y.tolist() == [1.0, 1.0, 1.0]
    assert c.tolist() == [0.1, 0.5, 2.0]


def test_oracle_match_reference_is_cheapest_at_least_as_good():
    y, c, is_ref = _picks()["oracle_match_reference"]
    assert c.tolist() == [0.1, 0.5, 2.0]
    assert is_ref.tolist() == [False, False, True]


def test_router_uses_predictions_not_truth():
    pred = np.array([[0.2, 0.3, 0.9]] * 3)          # always prefers gpt4 at lam 0
    p = _picks({"r": (pred, C)}, lams=(0.0, 1.0))
    assert p["r@0"][2].all()
    # lam 1: utilities 0.1, -0.2, -1.1 -> cheap everywhere
    assert p["r@1"][1].tolist() == [0.1, 0.1, 0.1]


def test_per_model_cost_vector_is_accepted():
    pred = np.array([[0.2, 0.8, 0.9]] * 3)
    p = _picks({"r": (pred, np.array([0.1, 0.5, 2.0]))}, lams=(1.0,))
    assert p["r@1"][1].tolist() == [0.5, 0.5, 0.5]   # 0.8 - 0.5 = 0.3 beats 0.1 and -1.1


def test_summarize_relative_to_reference_and_combine():
    s = summarize(_picks())
    assert s["reference"]["q_vs_reference"] == 0.0 and s["reference"]["reference_share"] == 1.0
    m = s["oracle_match_reference"]
    assert m["cost_per_1k"] == pytest.approx(1e3 * (0.1 + 0.5 + 2.0) / 3)
    assert m["cost_vs_reference"] == pytest.approx((2.6 / 3) / 2.0 - 1)
    both = combine({"a": _picks(), "b": _picks()})
    assert len(both["reference"][0]) == 6
    assert summarize(both)["oracle_match_reference"]["quality"] == pytest.approx(m["quality"])


def test_per_model_mean_cost_uses_finite_train_cells_only():
    obs = pd.DataFrame({"model_id": ["a", "a", "a", "b", "b"],
                        "split": ["train", "train", "test", "train", "train"],
                        "cost": [1.0, 3.0, 100.0, np.nan, 4.0]})
    s = per_model_mean_cost(obs)
    assert s["a"] == 2.0 and s["b"] == 4.0
