"""Phase 1 training: BCE decreases, synthetic IRT recovery, no cold-start leakage."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from router.nirt.baseline.train import fit
from router.nirt.baseline.synthetic import make_synthetic, recovery_report, to_arrays


def _tiny_cfg(k=3, epochs=20):
    return {
        "seed": 0,
        "model": {"theta_dim": k, "model_params": "free", "query_hidden": 32,
                  "use_length_head": False},
        "ablation": {"use_relevance": False, "use_warmup": False, "use_interaction": False},
        "regularization": {"theta_l2": 1e-4, "theta_center_l2": 1e-3, "difficulty_l2": 1e-4},
        "train": {"target": "binary", "lr": 3e-3, "batch_size": 1024, "epochs": epochs,
                  "patience": 20, "device": "cpu"},
    }


def test_bce_decreases_on_synthetic():
    syn = make_synthetic(n_queries=300, n_models=6, K=2, seed=1)
    tr, va = to_arrays(syn, seed=1)
    res = fit(_tiny_cfg(k=2, epochs=12), arrays=(tr, va), save=False, verbose=False)
    first = res.history[0]["train_loss"]
    last = res.history[-1]["train_loss"]
    assert last < first - 0.02


def test_synthetic_recovery_orderings():
    syn = make_synthetic(n_queries=1200, n_models=10, K=4, seed=0)
    tr, va = to_arrays(syn, seed=0)
    res = fit(_tiny_cfg(k=4, epochs=45), arrays=(tr, va), save=False, verbose=False)
    rep = recovery_report(res.model, syn, va)
    assert rep["bce_gap_vs_bayes"] < 0.09
    assert rep["difficulty_spearman"] > 0.55
    assert rep["ability_spearman_1d"] > 0.55


def test_reproducible_with_same_seed():
    syn = make_synthetic(n_queries=250, n_models=5, K=2, seed=2)
    tr, va = to_arrays(syn, seed=2)
    a = fit(_tiny_cfg(k=2, epochs=8), arrays=(tr, va), save=False, verbose=False)
    b = fit(_tiny_cfg(k=2, epochs=8), arrays=(tr, va), save=False, verbose=False)
    assert a.val_metrics["bce"] == pytest.approx(b.val_metrics["bce"], abs=1e-6)


def test_cold_start_models_not_optimized():
    """Models absent from the training arrays get no gradient: their theta row
    stays at initialisation."""
    syn = make_synthetic(n_queries=400, n_models=8, K=3, seed=3)
    tr, va = to_arrays(syn, seed=3)
    # drop model index 7 from every training row (simulate a held-out model)
    keep = tr.midx != 7
    tr.e_q, tr.midx, tr.y, tr.y_soft = tr.e_q[keep], tr.midx[keep], tr.y[keep], tr.y_soft[keep]
    tr.query_ids, tr.model_ids = tr.query_ids[keep], tr.model_ids[keep]
    tr.e_m = tr.e_m[keep]

    res = fit(_tiny_cfg(k=3, epochs=10), arrays=(tr, va), save=False, verbose=False)
    theta = res.model.theta.weight.detach().numpy()
    # row 7 was never in a batch -> unchanged from the seeded init
    from router.determinism import seed_everything
    from router.nirt.baseline.model import BaselineNIRT

    seed_everything(0)
    ref = BaselineNIRT(query_dim=tr.query_dim, dim=3, n_models=8, model_params="free",
                       query_hidden=32, use_length_head=False).theta.weight.detach().numpy()
    assert np.allclose(theta[7], ref[7], atol=1e-6)
    assert not np.allclose(theta[0], ref[0], atol=1e-6)   # trained rows did move


def test_imbalance_report_shape():
    syn = make_synthetic(n_queries=200, n_models=4, K=2, seed=4)
    tr, _ = to_arrays(syn, seed=4)
    rep = tr.imbalance_report()
    assert 0.0 <= rep["positive_rate"] <= 1.0
    assert rep["n_models"] == 4
    assert set(rep["obs_per_model"]) == set(tr.meta["model_index"])
