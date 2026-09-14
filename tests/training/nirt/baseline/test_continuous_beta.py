"""Beta head: valid concentrations, mean parameterisation, boundary handling, recovery."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from router.nirt.baseline.train import fit
from router.nirt.baseline.continuous_beta import BetaResponseHead
from router.nirt.baseline.continuous_synthetic import make_synthetic_continuous, recovery_report, to_arrays


def _head(**kw):
    return BetaResponseHead(feature_dim=12, hidden=8, cfg={"min_concentration": 1e-4,
                                                           "max_concentration": 1e4, **kw})


def test_alpha_beta_positive():
    head = _head()
    out = head(torch.linspace(-30, 30, 64), torch.randn(64, 12))
    assert (out["alpha"] > 0).all() and (out["beta"] > 0).all()
    assert (out["kappa"] >= 1e-4).all() and (out["kappa"] <= 1e4).all()


def test_mean_equals_mu():
    head = _head()
    out = head(torch.tensor([0.0, 2.0, -2.0]), torch.zeros(3, 12))
    torch.testing.assert_close(head.mean(out), torch.sigmoid(torch.tensor([0.0, 2.0, -2.0])),
                               rtol=1e-3, atol=1e-3)


def test_variance_shrinks_with_concentration():
    lo = _head()
    out_lo = lo(torch.zeros(1), torch.zeros(1, 12))
    # force large kappa via the global offset
    lo.log_kappa0.data.fill_(6.0)
    out_hi = lo(torch.zeros(1), torch.zeros(1, 12))
    assert lo.variance(out_hi) < lo.variance(out_lo)


def test_boundary_and_near_boundary_finite():
    head = _head()
    out = head(torch.tensor([0.0, 10.0, -10.0]), torch.zeros(3, 12))
    for y in (torch.zeros(3), torch.ones(3), torch.full((3,), 1e-6), torch.full((3,), 1 - 1e-6)):
        nll = head.nll(y, out)
        assert torch.isfinite(nll).all()


def test_mu_extremes_no_nan():
    head = _head()
    out = head(torch.tensor([200.0, -200.0]), torch.zeros(2, 12))
    assert torch.isfinite(head.nll(torch.tensor([0.5, 0.5]), out)).all()
    assert (out["mu"] > 0).all() and (out["mu"] < 1).all()


def test_interior_only_flag():
    assert _head().interior_only is True


def test_synthetic_recovery():
    syn = make_synthetic_continuous("beta", n_queries=1000, n_models=10, K=3, seed=2)
    tr, va = to_arrays(syn, seed=2)
    cfg = {"seed": 2, "model": {"theta_dim": 3, "model_params": "free", "query_hidden": 64,
                                "use_length_head": False},
           "ablation": {"use_relevance": False, "use_interaction": False, "use_warmup": False},
           "response": {"model": "beta", "cfg": {}},
           "regularization": {"theta_l2": 1e-4, "theta_center_l2": 1e-3, "difficulty_l2": 1e-4},
           "train": {"target": "soft", "lr": 3e-3, "batch_size": 2048, "epochs": 40,
                     "patience": 12, "device": "cpu"}}
    rep = recovery_report(fit(cfg, arrays=(tr, va), save=False, verbose=False).model, syn, va)
    assert rep["nll_finite"]
    assert rep["mean_corr"] > 0.9
    assert rep["difficulty_spearman"] > 0.6
    assert rep["dispersion_spearman"] > 0.4
