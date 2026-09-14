"""Heteroskedastic Normal head: positivity, likelihood, recovery."""

from __future__ import annotations

import math

import numpy as np
import pytest
from helpers import baseline_cfg

torch = pytest.importorskip("torch")

from training.nirt.baseline.train import fit
from training.nirt.baseline.continuous_normal import NormalResponseHead
from training.nirt.baseline.continuous_synthetic import make_synthetic_continuous, recovery_report, to_arrays


def _head():
    return NormalResponseHead(feature_dim=12, hidden=8, cfg={"epsilon": 1e-6})


def test_sigma_positive_and_heteroskedastic():
    head = _head()
    feats = torch.randn(64, 12)
    out = head(torch.randn(64), feats)
    assert (out["sigma"] > 0).all()
    assert out["sigma"].std() > 0            # varies with input


def test_nll_matches_analytic_normal():
    head = _head()
    z = torch.tensor([0.0, 1.0, -2.0])
    out = head(z, torch.zeros(3, 12))
    y = torch.tensor([0.2, 0.9, 0.1])
    sigma = out["sigma"]
    manual = 0.5 * ((y - z) / sigma) ** 2 + torch.log(sigma) + 0.5 * math.log(2 * math.pi)
    torch.testing.assert_close(head.nll(y, out), manual, rtol=1e-4, atol=1e-5)


def test_mean_and_variance():
    head = _head()
    out = head(torch.tensor([0.5, -1.0]), torch.zeros(2, 12))
    torch.testing.assert_close(head.mean(out), torch.tensor([0.5, -1.0]))
    torch.testing.assert_close(head.variance(out), out["sigma"] ** 2)


def test_extreme_inputs_finite():
    head = _head()
    z = torch.tensor([50.0, -50.0, 0.0])
    out = head(z, torch.zeros(3, 12))
    for y in (torch.zeros(3), torch.ones(3), torch.full((3,), 0.5)):
        assert torch.isfinite(head.nll(y, out)).all()


def test_analytic_interval_covers():
    head = _head()
    out = head(torch.zeros(1), torch.zeros(1, 12))
    lo, hi = head.interval(out, 0.9)
    # central 90% of N(0, sigma) ~ +/- 1.645 sigma
    assert abs((hi - lo).item() / out["sigma"].item() - 2 * 1.6449) < 0.05


def test_synthetic_recovery():
    syn = make_synthetic_continuous("normal", n_queries=1000, n_models=10, K=3, seed=1)
    tr, va = to_arrays(syn, seed=1)
    cfg = baseline_cfg(response="normal", k=3, epochs=45, patience=15, seed=1)
    rep = recovery_report(fit(cfg, arrays=(tr, va), save=False, verbose=False).model, syn, va)
    assert rep["nll_finite"]
    assert rep["mean_corr"] > 0.8
    assert rep["ability_spearman_1d"] > 0.5
    assert abs(rep["dispersion_spearman"]) > 0.2
