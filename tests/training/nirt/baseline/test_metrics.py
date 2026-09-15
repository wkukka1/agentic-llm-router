"""Phase 1 metrics: accuracy / Brier / log-loss / ECE / reliability on toy inputs."""

from __future__ import annotations

import numpy as np
import pytest

from training.nirt.baseline.calibration import calibration_report
from training.nirt.metrics import (
    bias_argmax_agreement,
    brier_score,
    effective_rank,
    log_loss,
    marginal_baselines,
    per_group_metrics,
    prediction_metrics,
    reliability_curve,
)


def test_brier_known_values():
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert brier_score([1, 1], [0.0, 0.0]) == 1.0
    assert brier_score([1, 0], [0.5, 0.5]) == 0.25


def test_log_loss_known_values():
    assert log_loss([1, 0], [0.5, 0.5]) == np.log(2)
    assert log_loss([1], [1.0]) < 1e-6
    assert log_loss([1], [0.0]) > 10          # clipped, large


def test_prediction_metrics_perfect_classifier():
    y = [1, 1, 0, 0]
    p = [0.9, 0.8, 0.1, 0.2]
    m = prediction_metrics(y, p)
    assert m["accuracy"] == 1.0
    assert m["auc"] == 1.0
    assert m["brier"] < 0.05
    assert m["pos_rate"] == 0.5


def test_prediction_metrics_known():
    perfect = prediction_metrics([1, 1, 0, 0], [1.0, 1.0, 0.0, 0.0])
    assert perfect["mse"] == pytest.approx(0.0)
    assert perfect["bce"] == pytest.approx(0.0, abs=1e-4)
    assert perfect["pearson_r"] == pytest.approx(1.0)
    assert perfect["acc@0.5"] == pytest.approx(1.0)

    flat = prediction_metrics([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5])
    assert flat["mse"] == pytest.approx(0.25)
    assert flat["bce"] == pytest.approx(np.log(2), abs=1e-4)
    assert np.isnan(flat["pearson_r"])  # constant prediction


def test_marginal_baselines():
    base = marginal_baselines(
        train_targets=[1.0, 0.0, 1.0, 1.0],
        train_model_ids=["a", "a", "b", "b"],
        eval_targets=[1.0, 0.0],
        eval_model_ids=["a", "b"],
    )
    assert base["global_mean"]["value"] == pytest.approx(0.75)
    assert base["model_mean"]["per_model"] == {"a": pytest.approx(0.5), "b": pytest.approx(1.0)}


def test_reliability_curve_bins_and_ece():
    # perfectly calibrated: in every occupied bin, accuracy == confidence
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(float)
    rc = reliability_curve(y, p, n_bins=10)
    assert rc["ece"] < 0.02
    assert sum(b["count"] for b in rc["bins"]) == 20000


def test_reliability_curve_detects_miscalibration():
    y = np.zeros(1000)
    p = np.full(1000, 0.9)          # confident and wrong
    rc = reliability_curve(y, p, n_bins=10)
    assert rc["ece"] > 0.8


def test_calibration_report_fields():
    rng = np.random.default_rng(1)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(float)
    rep = calibration_report(y, p, n_bins=10)
    assert {"ece", "mce", "brier", "log_loss", "reliability_curve"} <= set(rep)
    assert len(rep["reliability_curve"]) == 10


# --------------------------------------------------------------------------- #
# P7: monotone-collapse diagnostics (effective_rank, bias_argmax_agreement)   #
# --------------------------------------------------------------------------- #
def test_effective_rank_isotropic_vs_collapsed():
    rng = np.random.default_rng(0)
    iso = rng.standard_normal((500, 3))                                    # eff rank ~ 3
    collapsed = np.outer(rng.standard_normal(500), [1.0, 0.0, 0.0])        # eff rank ~ 1
    assert effective_rank(collapsed) == pytest.approx(1.0, abs=0.05)
    assert effective_rank(iso) > 2.5


def test_effective_rank_nan_on_degenerate_input():
    assert np.isnan(effective_rank(np.ones((1, 3))))       # < 2 rows
    assert np.isnan(effective_rank(np.zeros((10, 3))))     # zero variance everywhere


def test_bias_argmax_agreement_full_and_zero():
    b_m = np.array([1.0, -1.0, 0.0])   # argmax(-b_m) = model 1 (-b_m = [-1, 1, 0])
    agree = np.tile([0.0, 5.0, 0.0], (10, 1))
    disagree = np.tile([5.0, 0.0, 0.0], (10, 1))
    assert bias_argmax_agreement(agree, b_m) == pytest.approx(1.0)
    assert bias_argmax_agreement(disagree, b_m) == pytest.approx(0.0)


def test_per_group_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300).astype(float)
    p = rng.random(300)
    groups = np.where(np.arange(300) < 150, "math", "code")
    groups[:5] = "tiny"                      # below min_n -> pooled into "(small)"
    out = per_group_metrics(y, p, groups, min_n=20)
    assert "overall" in out and out["overall"]["n"] == 300
    assert out["math"]["n"] + out["code"]["n"] + out["(small)"]["n"] == 300
    assert "tiny" not in out
