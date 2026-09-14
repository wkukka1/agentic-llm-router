"""``training.nirt.train``: synthetic IRT recovery, checkpoint round-trips,
determinism, the query-grouped sampler + routing-aware losses, LR schedules /
grad clipping, and the auxiliary pairwise (Arena/Judge) objective.

The synthetic world, datasets and the compact ``fit`` config come from
``helpers`` -- see ``helpers/nirt.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import make_synthetic_irt, nirt_cfg, nirt_datasets, toy_pairwise_arrays
from helpers.stores import make_store

torch = pytest.importorskip("torch")

from router.nirt.checkpoint import load_run
from router.nirt.model import IRTRouterModel
from router.nirt.predict import predict_dataset
from training.nirt.metrics import marginal_baselines, prediction_metrics
from training.nirt.train import fit


# --------------------------------------------------------------------------- #
# well-specified recovery                                                      #
# --------------------------------------------------------------------------- #
def test_synthetic_recovery():
    """A linear query head should recover theta_q (and thus the true P(correct))."""
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(seed=1)
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)
    cfg = nirt_cfg(model_params="free", dim=2, query_hidden=None)
    cfg["train"].update(epochs=150, patience=30, lr=1e-2)

    res = fit(cfg, datasets=(train_ds, val_ds), save=False, verbose=False)
    _, y_prob = predict_dataset(res.model, val_ds, res.model_index)

    p_true = val_ds.obs["p_true"].to_numpy()
    rho = prediction_metrics(p_true, y_prob)["pearson_r"]
    assert rho > 0.85, f"recovered corr with true probabilities too low: {rho}"

    base = marginal_baselines(train_ds.targets, train_ds.model_ids,
                              val_ds.targets, val_ds.model_ids)
    assert res.val_metrics["bce"] < base["model_mean"]["bce"]


@pytest.mark.parametrize("mode", ["projected", "free"])
def test_fit_runs_and_saves(mode, tmp_path):
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(n_queries=200, seed=2)
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)
    res = fit(nirt_cfg(model_params=mode), datasets=(train_ds, val_ds),
              name=f"t-{mode}", runs_dir=tmp_path, verbose=False)
    assert res.path == tmp_path / f"t-{mode}"
    assert (res.path / "model.pt").exists()
    assert (res.path / "run.json").exists()
    assert np.isfinite(res.val_metrics["bce"])
    assert len(res.history) >= 1

    model, cfg, model_index = load_run(f"t-{mode}", runs_dir=tmp_path)
    assert model.model_params == mode
    y, p = predict_dataset(model, val_ds, model_index)
    assert p.shape == y.shape and np.all((p >= 0) & (p <= 1))


def test_determinism():
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(n_queries=200, seed=3)
    ds = nirt_datasets(train_obs, val_obs, q_store, m_store)
    a = fit(nirt_cfg(), datasets=ds, save=False, verbose=False)
    b = fit(nirt_cfg(), datasets=ds, save=False, verbose=False)
    assert a.history[0]["train_loss"] == pytest.approx(b.history[0]["train_loss"])
    assert a.val_metrics["bce"] == pytest.approx(b.val_metrics["bce"])


def test_free_mode_rejects_unseen_model():
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(n_queries=120, seed=4)
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)
    res = fit(nirt_cfg(model_params="free"), datasets=(train_ds, val_ds), save=False, verbose=False)
    with pytest.raises(ValueError, match="cannot score models absent"):
        predict_dataset(res.model, val_ds, {"only-this": 0})


# --------------------------------------------------------------------------- #
# multidimensional difficulty: recovery + checkpoint                           #
# --------------------------------------------------------------------------- #
def test_vector_difficulty_recovers_and_checkpoints(tmp_path):
    """A world with per-dimension thresholds: vector difficulty should fit it and
    round-trip through a checkpoint."""
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(seed=1, K=3)
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)
    cfg = nirt_cfg(model_params="free", dim=3, query_hidden=None, difficulty="vector")
    cfg["train"].update(epochs=120, patience=25, lr=1e-2)
    fit(cfg, datasets=(train_ds, val_ds), name="cx-vec", runs_dir=tmp_path, verbose=False)

    model, loaded_cfg, midx = load_run("cx-vec", runs_dir=tmp_path)
    assert model.difficulty == "vector"
    _, y_prob = predict_dataset(model, val_ds, midx)
    rho = prediction_metrics(val_ds.obs["p_true"].to_numpy(), y_prob)["pearson_r"]
    assert rho > 0.8, rho


# --------------------------------------------------------------------------- #
# P2a: deeper query head still recovers + checkpoints                          #
# --------------------------------------------------------------------------- #
def test_deeper_head_fits_synthetic(tmp_path):
    """A deeper/normed query head still recovers a well-specified IRT world and
    round-trips through a checkpoint."""
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(seed=1, K=2)
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)
    cfg = nirt_cfg(model_params="free", dim=2, query_hidden=32, query_head_layers=2,
                   query_head_norm="layernorm", query_head_activation="gelu",
                   query_head_dropout=0.1)
    cfg["train"].update(epochs=120, patience=25, lr=1e-2)
    fit(cfg, datasets=(train_ds, val_ds), name="cx-head", runs_dir=tmp_path, verbose=False)

    from router.nirt.model import _MLPHead

    model, _, midx = load_run("cx-head", runs_dir=tmp_path)
    assert isinstance(model.query_head, _MLPHead)
    _, y_prob = predict_dataset(model, val_ds, midx)
    rho = prediction_metrics(val_ds.obs["p_true"].to_numpy(), y_prob)["pearson_r"]
    assert rho > 0.8, rho


# --------------------------------------------------------------------------- #
# P2c: interaction residual -- from config + checkpoint                        #
# --------------------------------------------------------------------------- #
def test_interaction_from_config_and_checkpoint(tmp_path):
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(seed=2, K=2)
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)
    cfg = nirt_cfg(model_params="free", dim=2, query_hidden=16, interaction=True,
                   interaction_hidden=16)
    cfg["train"].update(epochs=60, patience=20, lr=1e-2)
    fit(cfg, datasets=(train_ds, val_ds), name="cx-int", runs_dir=tmp_path, verbose=False)

    model, _, midx = load_run("cx-int", runs_dir=tmp_path)
    assert model.interaction and "interaction_gamma" in model.state_dict()
    _, y_prob = predict_dataset(model, val_ds, midx)
    rho = prediction_metrics(val_ds.obs["p_true"].to_numpy(), y_prob)["pearson_r"]
    assert rho > 0.8, rho


# --------------------------------------------------------------------------- #
# LR schedule / grad clipping                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("schedule", ["cosine", "plateau"])
def test_trainer_grad_clip_and_lr_schedule(schedule, tmp_path):
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(n_queries=200, seed=6)
    ds = nirt_datasets(train_obs, val_obs, q_store, m_store)
    cfg = nirt_cfg(model_params="free", dim=2)
    cfg["train"].update(epochs=6, patience=10, grad_clip=1.0, lr_schedule=schedule)
    res = fit(cfg, datasets=ds, save=False, verbose=False)
    assert np.isfinite(res.val_metrics["bce"])


def test_trainer_defaults_unchanged():
    """grad_clip=0 + lr_schedule=none reproduces the legacy training path."""
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(n_queries=200, seed=7)
    ds = nirt_datasets(train_obs, val_obs, q_store, m_store)
    a = fit(nirt_cfg(), datasets=ds, save=False, verbose=False)
    cfg = nirt_cfg()
    cfg["train"].update(grad_clip=0.0, lr_schedule="none")
    b = fit(cfg, datasets=ds, save=False, verbose=False)
    assert a.val_metrics["bce"] == pytest.approx(b.val_metrics["bce"])


# --------------------------------------------------------------------------- #
# P5: query-grouped sampler + routing-aware losses                             #
# --------------------------------------------------------------------------- #
def test_grouped_batches_partition():
    from training.nirt.train import _grouped_batches

    qgroup = np.array([0, 0, 1, 1, 1, 2, 3, 3])
    gen = torch.Generator().manual_seed(0)
    batches = list(_grouped_batches(qgroup, batch_queries=2, gen=gen))
    allrows = np.concatenate(batches)
    assert sorted(allrows.tolist()) == list(range(len(qgroup)))     # every row once
    for b in batches:                                               # whole groups only
        gs = set(qgroup[b].tolist())
        assert all((qgroup == g).sum() == (qgroup[b] == g).sum() for g in gs)


def test_sampler_default_unchanged():
    """sampler=cell + loss=soft_bce reproduces the legacy training path."""
    ds = nirt_datasets(*make_synthetic_irt(n_queries=200, seed=7))
    a = fit(nirt_cfg(), datasets=ds, save=False, verbose=False)
    cfg = nirt_cfg()
    cfg["train"].update(sampler="cell", loss="soft_bce")
    b = fit(cfg, datasets=ds, save=False, verbose=False)
    assert a.val_metrics["bce"] == pytest.approx(b.val_metrics["bce"])


@pytest.mark.parametrize("loss", ["pairwise", "listwise", "bce_pairwise"])
def test_routing_aware_loss_learns_ranking(loss):
    """A within-query loss should recover a low routing regret on a synthetic
    world (and auto-switch the sampler to 'query')."""
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(seed=1, n_models=6, K=2)
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)
    cfg = nirt_cfg(model_params="free", dim=4, query_hidden=32)
    cfg["train"].update(loss=loss, epochs=60, patience=20, lr=1e-2,
                        batch_queries=128, val_metric="regret")
    res = fit(cfg, datasets=(train_ds, val_ds), save=False, verbose=False)
    assert "regret" in res.val_metrics
    # oracle regret on this world is ~0.05-0.15; a working ranker gets well under 0.25
    assert res.val_metrics["regret"] < 0.25, res.val_metrics["regret"]


def test_val_metric_regret_early_stops(tmp_path):
    ds = nirt_datasets(*make_synthetic_irt(n_queries=200, seed=3))
    cfg = nirt_cfg()
    cfg["train"].update(val_metric="regret", epochs=8, patience=3)
    res = fit(cfg, datasets=ds, name="rg", runs_dir=tmp_path, verbose=False)
    assert "regret" in res.val_metrics and np.isfinite(res.val_metrics["regret"])
    assert any("val_regret" in h for h in res.history)


def test_fit_with_concatenated_query_features(tmp_path):
    """A NIRTDataset built with a structured feature_store widens query_dim, and
    the model sized from it trains + checkpoints normally (P3-style concat)."""
    from router.data.nirt import NIRTDataset

    train_obs, val_obs, q_store, m_store = make_synthetic_irt(seed=1, K=2)
    feat_ids = sorted(set(train_obs.query_id) | set(val_obs.query_id))
    feat_store = make_store(feat_ids, 3, "query_id", seed=0)
    train_ds = NIRTDataset(train_obs, q_store, m_store, feature_store=feat_store)
    val_ds = NIRTDataset(val_obs, q_store, m_store, feature_store=feat_store)
    assert train_ds.query_dim == q_store.dim + 3

    cfg = nirt_cfg(model_params="free", dim=2, query_hidden=16)
    cfg["train"].update(epochs=15, patience=10)
    fit(cfg, datasets=(train_ds, val_ds), name="cx-qf", runs_dir=tmp_path, verbose=False)

    model, _, midx = load_run("cx-qf", runs_dir=tmp_path)
    assert model.query_dim == q_store.dim + 3
    _, y_prob = predict_dataset(model, val_ds, midx)
    assert np.all(np.isfinite(y_prob))


# --------------------------------------------------------------------------- #
# auxiliary pairwise (Arena/Judge) training                                    #
# --------------------------------------------------------------------------- #
def test_preference_disabled_by_default_unchanged():
    ds = nirt_datasets(*make_synthetic_irt(n_queries=150, seed=5))
    cfg = nirt_cfg(model_params="projected", dim=2)
    a = fit(cfg, datasets=ds, save=False, verbose=False)
    cfg2 = nirt_cfg(model_params="projected", dim=2)
    cfg2["train"]["preference"] = {"enabled": False}
    b = fit(cfg2, datasets=ds, save=False, verbose=False)
    assert a.val_metrics["bce"] == pytest.approx(b.val_metrics["bce"])
    assert "pairwise_loss" not in a.history[0] and "pairwise_loss" not in b.history[0]


def test_preference_enabled_trains_with_explicit_arrays(tmp_path):
    ds = nirt_datasets(*make_synthetic_irt(n_queries=150, q_dim=10, seed=6))
    pw = toy_pairwise_arrays(q_dim=10, m_dim=5)
    cfg = nirt_cfg(model_params="projected", dim=2)
    cfg["train"]["preference"] = {"enabled": True, "weight": 0.5, "batch_size": 16}
    cfg["train"].update(epochs=3)
    res = fit(cfg, datasets=ds, pairwise_arrays=pw, name="cx-pref", runs_dir=tmp_path, verbose=False)
    assert "pairwise_loss" in res.history[0]
    assert np.isfinite(res.history[0]["pairwise_loss"])

    model, _, _ = load_run("cx-pref", runs_dir=tmp_path)
    assert model.model_params == "projected"


def test_preference_rejects_free_model_params():
    ds = nirt_datasets(*make_synthetic_irt(n_queries=80, seed=7))
    pw = toy_pairwise_arrays()
    cfg = nirt_cfg(model_params="free", dim=2)
    cfg["train"]["preference"] = {"enabled": True}
    with pytest.raises(ValueError, match="projected"):
        fit(cfg, datasets=ds, pairwise_arrays=pw, save=False, verbose=False)


def test_preference_without_data_or_arrays_raises():
    ds = nirt_datasets(*make_synthetic_irt(n_queries=80, seed=8))
    cfg = nirt_cfg(model_params="projected", dim=2)
    cfg["train"]["preference"] = {"enabled": True}
    with pytest.raises(ValueError, match="pairwise_arrays"):
        fit(cfg, datasets=ds, save=False, verbose=False)


# --------------------------------------------------------------------------- #
# P7b/c/d: monotone-collapse fixes (theta_cov_weight, pairwise_weighting,      #
# zero_variance_weight, per-epoch collapse alarm)                             #
# --------------------------------------------------------------------------- #
def test_query_variance_mask_flags_constant_queries():
    from training.nirt.train import _query_variance_mask

    y = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    qgroup = np.array([0, 0, 0, 1, 1, 1])   # group 0: all 1.0 (zero variance); group 1: mixed
    mask = _query_variance_mask(y, qgroup)
    assert mask.tolist() == [True, True, True, False, False, False]


def test_theta_cov_weight_default_off_unchanged():
    ds = nirt_datasets(*make_synthetic_irt(n_queries=200, seed=10))
    a = fit(nirt_cfg(model_params="free", dim=2), datasets=ds, save=False, verbose=False)
    cfg = nirt_cfg(model_params="free", dim=2)
    cfg["train"].update(theta_cov_weight=0.0)
    b = fit(cfg, datasets=ds, save=False, verbose=False)
    assert a.val_metrics["bce"] == pytest.approx(b.val_metrics["bce"])


def test_theta_cov_weight_runs_and_decorrelates():
    ds = nirt_datasets(*make_synthetic_irt(n_queries=300, seed=11, K=3))
    cfg = nirt_cfg(model_params="free", dim=3)
    cfg["train"].update(epochs=15, theta_cov_weight=1.0)
    res = fit(cfg, datasets=ds, save=False, verbose=False)
    assert np.isfinite(res.val_metrics["bce"])
    assert all(np.isfinite(h["theta_effective_rank"]) for h in res.history)


def test_pairwise_weighting_default_none_unchanged():
    ds = nirt_datasets(*make_synthetic_irt(n_queries=200, n_models=6, seed=12))
    cfg = nirt_cfg(model_params="free", dim=2, query_hidden=16)
    cfg["train"].update(loss="pairwise", batch_queries=64, epochs=5)
    a = fit(cfg, datasets=ds, save=False, verbose=False)
    cfg2 = nirt_cfg(model_params="free", dim=2, query_hidden=16)
    cfg2["train"].update(loss="pairwise", batch_queries=64, epochs=5, pairwise_weighting="none")
    b = fit(cfg2, datasets=ds, save=False, verbose=False)
    assert a.val_metrics["bce"] == pytest.approx(b.val_metrics["bce"])


def test_pairwise_weighting_abs_diff_runs():
    ds = nirt_datasets(*make_synthetic_irt(n_queries=200, n_models=6, seed=13))
    cfg = nirt_cfg(model_params="free", dim=2, query_hidden=16)
    cfg["train"].update(loss="bce_pairwise", batch_queries=64, epochs=5,
                        pairwise_weighting="abs_diff")
    res = fit(cfg, datasets=ds, save=False, verbose=False)
    assert np.isfinite(res.val_metrics["bce"])


def test_pairwise_weighting_rejects_bad_choice():
    ds = nirt_datasets(*make_synthetic_irt(n_queries=80, seed=14))
    cfg = nirt_cfg(model_params="free", dim=2)
    cfg["train"].update(pairwise_weighting="bogus")
    with pytest.raises(ValueError, match="pairwise_weighting"):
        fit(cfg, datasets=ds, save=False, verbose=False)


def test_zero_variance_weight_default_unchanged():
    ds = nirt_datasets(*make_synthetic_irt(n_queries=200, seed=15))
    a = fit(nirt_cfg(model_params="free"), datasets=ds, save=False, verbose=False)
    cfg = nirt_cfg(model_params="free")
    cfg["train"].update(zero_variance_weight=1.0)
    b = fit(cfg, datasets=ds, save=False, verbose=False)
    assert a.val_metrics["bce"] == pytest.approx(b.val_metrics["bce"])


def test_zero_variance_weight_runs():
    ds = nirt_datasets(*make_synthetic_irt(n_queries=200, seed=16))
    cfg = nirt_cfg(model_params="free")
    cfg["train"].update(zero_variance_weight=0.1, epochs=5)
    res = fit(cfg, datasets=ds, save=False, verbose=False)
    assert np.isfinite(res.val_metrics["bce"])


def test_collapse_diagnostics_logged_for_query_latent():
    ds = nirt_datasets(*make_synthetic_irt(n_queries=150, seed=17))
    res = fit(nirt_cfg(model_params="free", dim=2), datasets=ds, save=False, verbose=False)
    for h in res.history:
        assert "theta_effective_rank" in h
        assert np.isfinite(h["theta_effective_rank"])
        assert "bias_argmax_agreement" in h    # free + scalar difficulty -> cheaply available
        assert 0.0 <= h["bias_argmax_agreement"] <= 1.0


def test_collapse_diagnostics_absent_for_model_latent():
    ds = nirt_datasets(*make_synthetic_irt(n_queries=150, seed=18, orientation="model_latent"))
    res = fit(nirt_cfg(orientation="model_latent", model_params="free", dim=2),
              datasets=ds, save=False, verbose=False)
    assert "theta_effective_rank" not in res.history[0]


def test_abort_on_collapse_stops_training():
    """collapse_rank_threshold set above K (2) so every epoch counts as
    'collapsed' regardless of actual behaviour -- a deterministic, non-flaky
    way to exercise the abort path."""
    ds = nirt_datasets(*make_synthetic_irt(n_queries=200, seed=19))
    cfg = nirt_cfg(model_params="free", dim=2)
    cfg["train"].update(epochs=10, patience=10, abort_on_collapse=True,
                        collapse_rank_threshold=1000.0, collapse_patience=1)
    res = fit(cfg, datasets=ds, save=False, verbose=False)
    assert len(res.history) == 1


# --------------------------------------------------------------------------- #
# orientation: model_latent recovery + checkpoint                              #
# --------------------------------------------------------------------------- #
def test_synthetic_recovery_model_latent():
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(seed=2, orientation="model_latent")
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)
    cfg = nirt_cfg(orientation="model_latent", model_params="free", dim=2, query_hidden=None)
    cfg["train"].update(epochs=150, patience=30, lr=1e-2)
    res = fit(cfg, datasets=(train_ds, val_ds), save=False, verbose=False)
    _, y_prob = predict_dataset(res.model, val_ds, res.model_index)
    rho = prediction_metrics(val_ds.obs["p_true"].to_numpy(), y_prob)["pearson_r"]
    assert rho > 0.85, rho


def test_fit_load_run_model_latent(tmp_path):
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(n_queries=200, seed=3,
                                                              orientation="model_latent")
    ds = nirt_datasets(train_obs, val_obs, q_store, m_store)
    fit(nirt_cfg(orientation="model_latent", model_params="projected"),
        datasets=ds, name="irt-t", runs_dir=tmp_path, verbose=False)

    model, cfg, midx = load_run("irt-t", runs_dir=tmp_path)
    assert isinstance(model, IRTRouterModel)
    y, p = predict_dataset(model, ds[1], midx)
    assert np.all((p >= 0) & (p <= 1))
