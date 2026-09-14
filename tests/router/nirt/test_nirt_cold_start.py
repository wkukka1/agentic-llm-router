"""Cold-start scoring (``evaluation.nirt.evaluate``): a projected model must score a
model that never appeared in training, from its profile embedding, better than
the global mean -- plus the nearest-profile fallback and the free-model guard.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import make_synthetic_irt, nirt_cfg, nirt_datasets
from helpers.stores import make_store

torch = pytest.importorskip("torch")

from evaluation.nirt.evaluate import cold_start_eval, nearest_profile_model
from router.nirt.model import NIRTModel
from router.nirt.predict import predict_dataset
from training.nirt.metrics import prediction_metrics
from training.nirt.train import fit


def test_cold_start_rejects_free_model():
    m = NIRTModel(query_dim=4, dim=1, n_models=2, model_params="free", profile_dim=4)
    with pytest.raises(ValueError, match="projected"):
        cold_start_eval(m, {"a": 0}, data=object())


def test_nearest_profile_model():
    mat = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    st = make_store(["a", "b", "c"], mat, "model_id")
    assert nearest_profile_model(st, "a", ["b", "c"]) == "b"


def test_projected_model_scores_held_out_model():
    """The core cold-start capability: a projected model predicts a model that was
    never in training, from its profile embedding, better than the global mean."""
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(
        n_queries=1400, n_models=8, seed=5, informative_profiles=True
    )
    held = {"m6", "m7"}
    tr = train_obs[~train_obs.model_id.isin(held)].reset_index(drop=True)
    train_ds, val_ds = nirt_datasets(tr, val_obs, q_store, m_store)   # val_ds still has all 8
    cfg = nirt_cfg(model_params="projected", dim=2)
    cfg["train"].update(epochs=120, patience=25, lr=1e-2)
    res = fit(cfg, datasets=(train_ds, val_ds), save=False, verbose=False)

    y, p = predict_dataset(res.model, val_ds, res.model_index)
    mids = val_ds.model_ids
    cold = np.isin(mids, list(held))
    cold_bce = prediction_metrics(y[cold], p[cold])["bce"]
    gm = float(tr.target.mean())
    gm_bce = prediction_metrics(y[cold], np.full(cold.sum(), gm))["bce"]
    assert cold_bce < gm_bce, (cold_bce, gm_bce)
