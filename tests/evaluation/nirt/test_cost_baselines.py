"""Expected-cost baselines B0 / B1 / B2 and their metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.nirt.cost_baselines import cost_baselines, cost_prediction_metrics, lambda_grid

PRICES = {"cheap": {"input_per_1m": 1.0, "output_per_1m": 2.0},
          "dear": {"input_per_1m": 10.0, "output_per_1m": 20.0}}


def _train():
    return pd.DataFrame({
        "query_id": ["a", "a", "b", "b"], "model_id": ["cheap", "dear", "cheap", "dear"],
        "cost": [1e-4, 1e-3, 3e-4, np.nan],
        "input_tokens": [10, 10, 30, 30], "output_tokens": [100, 200, 300, 400],
    })


def test_baselines_use_train_only_statistics():
    cells = pd.DataFrame({"query_id": ["c", "c", "c"], "model_id": ["cheap", "dear", "unseen"],
                          "input_tokens": [1000, 1000, 1000]})
    b = cost_baselines(_train(), cells, PRICES)
    finite = np.array([1e-4, 1e-3, 3e-4])
    assert np.allclose(b["B0"], finite.mean())
    assert np.allclose(b["B1"], [2e-4, 1e-3, finite.mean()])     # NaN train cost dropped
    # cheap: (1*1000 + 2*mean_out(200)) / 1e6 ; dear: (10*1000 + 20*300) / 1e6
    assert b["B2"][0] == pytest.approx((1000 + 400) / 1e6)
    assert b["B2"][1] == pytest.approx((10000 + 6000) / 1e6)
    assert np.isnan(b["B2"][2])                                    # unpriced -> NaN


def test_b2_is_nan_without_prices_or_tokens():
    cells = pd.DataFrame({"query_id": ["c"], "model_id": ["cheap"]})
    assert np.isnan(cost_baselines(_train(), cells, PRICES)["B2"]).all()
    assert np.isnan(cost_baselines(_train(), cells.assign(input_tokens=5), None)["B2"]).all()


def test_metrics_perfect_and_coverage():
    true = np.array([1e-4, 2e-4, 4e-4, 1e-3, 2e-3, 4e-3])
    q = np.array(["a", "a", "a", "b", "b", "b"])
    m = cost_prediction_metrics(true, true, q)
    assert m["mae_log"] == 0 and m["spearman"] == pytest.approx(1)
    assert m["within_query_spearman"] == pytest.approx(1) and m["weighted_abs_error"] == 0
    pred = true.copy()
    pred[0] = np.nan
    assert cost_prediction_metrics(pred, true, q)["coverage"] == pytest.approx(5 / 6)


def test_within_query_spearman_ignores_between_query_scale():
    true = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    pred = np.array([30.0, 20.0, 10.0, 3.0, 2.0, 1.0])   # within-query order reversed
    m = cost_prediction_metrics(pred, true, np.array(list("aaabbb")))
    assert m["within_query_spearman"] == pytest.approx(-1)


def test_lambda_grid():
    pq = np.array([[0.9, 0.7], [0.8, 0.6]])               # dq = 0.2
    pc = np.array([[1e-3, 1e-4], [2e-3, 1e-4]])
    g = lambda_grid(pq, pc)
    assert g[0] == 0 and np.all(np.diff(g) > 0)
    assert g[1] == pytest.approx(0.2 / np.quantile([9e-4, 1.9e-3], 0.9))
    assert lambda_grid(pq, np.ones_like(pc)).tolist() == [0.0]


def test_free_model_mean_cost_is_zero_not_missing():
    train = pd.DataFrame({"query_id": ["a", "b"], "model_id": ["free", "free"], "cost": [0.0, 0.0]})
    cells = pd.DataFrame({"query_id": ["c"], "model_id": ["free"]})
    assert cost_baselines(train, cells)["B1"].tolist() == [0.0]
