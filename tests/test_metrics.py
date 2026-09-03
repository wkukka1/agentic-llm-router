"""Phase 1 metrics: accuracy / Brier / log-loss / ECE / reliability on toy inputs."""

from __future__ import annotations

import numpy as np

from router.nirt.calibration import calibration_report
from router.nirt.metrics import (
    brier_score,
    log_loss,
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
