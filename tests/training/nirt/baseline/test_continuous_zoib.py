"""ZOIB head: simplex, boundary-mass vs interior density, recovery."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import baseline_cfg

torch = pytest.importorskip("torch")

from training.nirt.baseline.train import fit
from training.nirt.baseline.continuous_synthetic import make_synthetic_continuous, recovery_report, to_arrays
from training.nirt.baseline.continuous_zoib import ZOIBResponseHead


def _head(**kw):
    return ZOIBResponseHead(feature_dim=12, hidden=8, cfg={"min_concentration": 1e-4,
                                                           "max_concentration": 1e4, **kw})


def test_probabilities_form_a_simplex():
    head = _head()
    out = head(torch.randn(64), torch.randn(64, 12))
    s = out["pi0"] + out["pi1"] + out["pic"]
    torch.testing.assert_close(s, torch.ones_like(s), rtol=1e-5, atol=1e-5)
    for k in ("pi0", "pi1", "pic"):
        assert (out[k] >= 0).all() and (out[k] <= 1).all()


def test_exact_zero_uses_boundary_mass():
    head = _head()
    out = head(torch.tensor([0.3, 0.3]), torch.zeros(2, 12))
    nll0 = head.nll(torch.tensor([0.0, 0.0]), out)
    torch.testing.assert_close(nll0, -out["log_pi0"], rtol=1e-5, atol=1e-5)


def test_exact_one_uses_boundary_mass():
    head = _head()
    out = head(torch.tensor([0.3, -1.0]), torch.zeros(2, 12))
    nll1 = head.nll(torch.tensor([1.0, 1.0]), out)
    torch.testing.assert_close(nll1, -out["log_pi1"], rtol=1e-5, atol=1e-5)


def test_interior_uses_beta_density():
    head = _head()
    out = head(torch.tensor([0.2]), torch.zeros(1, 12))
    y = torch.tensor([0.4])
    beta_lp = torch.distributions.Beta(out["alpha"], out["beta"]).log_prob(y)
    torch.testing.assert_close(head.nll(y, out), -(out["log_pic"] + beta_lp), rtol=1e-4, atol=1e-5)


def test_exact_vs_near_boundary_differ():
    head = _head()
    out = head(torch.tensor([0.5, 0.5]), torch.zeros(2, 12))
    exact0 = head.nll(torch.tensor([0.0, 0.0]), out)[0]
    near0 = head.nll(torch.tensor([1e-4, 1e-4]), out)[0]
    assert not torch.isclose(exact0, near0, atol=1e-2)   # boundary mass vs Beta density


def test_mean_is_convex_combo():
    head = _head()
    out = head(torch.tensor([0.7]), torch.zeros(1, 12))
    expected = out["pi1"] + out["pic"] * out["mu"]
    torch.testing.assert_close(head.mean(out), expected)
    assert 0 <= head.mean(out).item() <= 1


def test_boundary_probs_and_finite_extremes():
    head = _head()
    out = head(torch.tensor([100.0, -100.0]), torch.zeros(2, 12))
    pi0, pi1 = head.boundary_probs(out)
    assert pi0.shape == (2,) and pi1.shape == (2,)
    for y in (torch.zeros(2), torch.ones(2), torch.full((2,), 0.5)):
        assert torch.isfinite(head.nll(y, out)).all()


def test_synthetic_recovery():
    syn = make_synthetic_continuous("zoib", n_queries=1200, n_models=10, K=3, seed=3)
    tr, va = to_arrays(syn, seed=3)
    cfg = baseline_cfg(response="zoib", k=3, epochs=45, patience=12, seed=3)
    rep = recovery_report(fit(cfg, arrays=(tr, va), save=False, verbose=False).model, syn, va)
    assert rep["nll_finite"]
    assert rep["mean_corr"] > 0.85
    assert rep["dispersion_spearman"] > 0.4
    # boundary masses roughly track the empirical fractions
    assert abs(rep["pred_pi0_mean"] - rep["true_frac0"]) < 0.08
    assert abs(rep["pred_pi1_mean"] - rep["true_frac1"]) < 0.08
