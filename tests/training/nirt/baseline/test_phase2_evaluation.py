"""Phase 2 evaluation utilities: boundary stats, continuous metrics, comparison shape."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.nirt.baseline.continuous_eval import _binned, _frac, continuous_metrics


def test_frac_partition_sums_to_one():
    y = np.array([0.0, 0.0, 1.0, 0.25, 0.5, 0.75, 1.0])
    f = _frac(y)
    assert abs(f["y==0"] + f["y==1"] + f["0<y<1"] - 1.0) < 1e-9
    assert f["y==0"] == pytest.approx(2 / 7)
    assert f["y==1"] == pytest.approx(2 / 7)


def test_continuous_metrics_perfect_mean():
    rng = np.random.default_rng(0)
    y = rng.random(5000)
    nll = np.full_like(y, 0.3)
    m = continuous_metrics(y, mean=y.copy(), proba=np.clip(y, 0, 1), nll=nll,
                           lower=y - 0.1, upper=y + 0.1, lower50=y - 0.05, upper50=y + 0.05)
    assert m["mae"] < 1e-9 and m["rmse"] < 1e-9
    assert m["nll_mean"] == pytest.approx(0.3)
    assert m["mean_calibration_ece"] < 1e-6
    assert m["coverage_90"] == 1.0


def test_coverage_reflects_interval_width():
    rng = np.random.default_rng(1)
    y = rng.random(4000)
    mean = np.full_like(y, 0.5)
    nll = np.zeros_like(y)
    wide = continuous_metrics(y, mean, mean, nll, lower=np.zeros_like(y), upper=np.ones_like(y))
    narrow = continuous_metrics(y, mean, mean, nll, lower=np.full_like(y, 0.49),
                                upper=np.full_like(y, 0.51))
    assert wide["coverage_90"] == 1.0
    assert narrow["coverage_90"] < 0.1


def test_binned_calibration():
    pred = np.array([0.05, 0.05, 0.95, 0.95])
    actual = np.array([0.0, 0.1, 1.0, 0.9])
    rows = _binned(pred, actual, n_bins=10)
    assert rows[0]["pred"] == pytest.approx(0.05)
    assert rows[0]["actual"] == pytest.approx(0.05)


def test_boundary_statistics_on_real_data(tmp_path):
    from router.config import load_config
    from training.nirt.baseline.continuous_eval import boundary_statistics

    cfg = load_config()
    try:
        stats = boundary_statistics(cfg, write=False)
    except FileNotFoundError:
        pytest.skip("phase 0 artifacts not present")
    for sp, f in stats["by_split"].items():
        assert abs(f["y==0"] + f["y==1"] + f["0<y<1"] - 1.0) < 1e-6
