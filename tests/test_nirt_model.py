from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from router.nirt.evaluate import cold_start_eval, nearest_profile_model, predict_dataset
from router.nirt.metrics import marginal_baselines, prediction_metrics
from router.nirt.model import IRTRouterModel, NIRTModel, build_model
from router.nirt.train import fit


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _store(ids, matrix, id_field):
    from router.embeddings.encoder import EmbeddingStore

    manifest = {"dim": matrix.shape[1], "id_field": id_field,
                "count": len(ids), "complete": True}
    return EmbeddingStore(list(ids), matrix.astype(np.float32), manifest, id_field=id_field)


def _synthetic(n_queries=1200, q_dim=16, n_models=6, K=2, seed=0,
               orientation="query_latent", informative_profiles=False):
    """A well-specified IRT world. ``y ~ Bernoulli(sigma(a . theta - b))`` where the
    latent vector sits on the query (``query_latent``) or the model (``model_latent``)."""
    rng = np.random.default_rng(seed)
    Eq = rng.standard_normal((n_queries, q_dim)).astype(np.float32)
    W = rng.standard_normal((q_dim, K)) / np.sqrt(q_dim)

    if orientation == "query_latent":
        theta_q = Eq @ W                                  # (n_queries, K)
        a_m = np.abs(rng.standard_normal((n_models, K))) + 0.5
        b_m = rng.standard_normal(n_models) * 0.5
        def logit(i, j): return float(a_m[j] @ theta_q[i] - b_m[j])
        prof_signal = np.hstack([a_m, b_m[:, None]])
    else:  # model_latent
        A_q = np.abs(Eq @ W) + 0.3                        # query discrimination (n_queries, K)
        b_q = (Eq @ rng.standard_normal((q_dim, 1))).ravel() * 0.5
        theta_m = rng.random((n_models, K))               # model ability in [0, 1]
        def logit(i, j): return float(A_q[i] @ theta_m[j] - b_q[i])
        prof_signal = theta_m

    Em = rng.standard_normal((n_models, 5)).astype(np.float32)
    if informative_profiles:
        Em = np.hstack([prof_signal, 0.05 * rng.standard_normal((n_models, 3))]).astype(np.float32)

    qids = [f"q{i}" for i in range(n_queries)]
    mids = [f"m{j}" for j in range(n_models)]
    rows = []
    for i in range(n_queries):
        for j in range(n_models):
            p = 1.0 / (1.0 + np.exp(-logit(i, j)))
            rows.append((qids[i], mids[j], float(rng.random() < p), p))
    obs = pd.DataFrame(rows, columns=["query_id", "model_id", "target", "p_true"])
    obs["metric_type"] = "accuracy"
    obs["source"] = "synth"
    obs["cost"] = np.nan

    cut = int(n_queries * 0.8)
    train_obs = obs[obs["query_id"].isin(qids[:cut])].reset_index(drop=True)
    val_obs = obs[obs["query_id"].isin(qids[cut:])].reset_index(drop=True)
    q_store = _store(qids, Eq, "query_id")
    m_store = _store(mids, Em, "model_id")
    return train_obs, val_obs, q_store, m_store


def _datasets(train_obs, val_obs, q_store, m_store):
    from router.data.nirt import NIRTDataset

    return (
        NIRTDataset(train_obs, q_store, m_store),
        NIRTDataset(val_obs, q_store, m_store),
    )


def _splitted_obs(n_queries=900, n_models=9, K=2, seed=0):
    """A synthetic observation table with train/validation/test splits + cost."""
    rng = np.random.default_rng(seed)
    Eq = rng.standard_normal((n_queries, 12))
    theta = Eq @ rng.standard_normal((12, K))          # strong latent signal
    a = np.abs(rng.standard_normal((n_models, K))) + 1.0
    b = rng.standard_normal(n_models)
    model_cost = np.linspace(0.0001, 0.004, n_models)
    qids = [f"q{i}" for i in range(n_queries)]
    mids = [f"m{j}" for j in range(n_models)]
    parts = np.array(["train"] * int(n_queries * 0.6)
                     + ["validation"] * int(n_queries * 0.2)
                     + ["test"] * (n_queries - int(n_queries * 0.6) - int(n_queries * 0.2)))
    rng.shuffle(parts)
    rows = []
    for i in range(n_queries):
        for j in range(n_models):
            p = 1.0 / (1.0 + np.exp(-(a[j] @ theta[i] - b[j])))
            rows.append((qids[i], mids[j], float(rng.random() < p), parts[i],
                         "accuracy", "synth", float(model_cost[j])))
    return pd.DataFrame(rows, columns=["query_id", "model_id", "target", "split",
                                       "metric_type", "source", "cost"])


class _FakeData:
    def __init__(self, obs, q_dim=None):
        self._obs = obs
        self._q = None
        if q_dim:
            qids = sorted(obs["query_id"].unique())
            mat = np.random.default_rng(0).standard_normal((len(qids), q_dim))
            self._q = _store(qids, mat, "query_id")

    def nirt_observations(self):
        return self._obs

    def query_embeddings(self, pathway=None):
        return self._q


def _cfg(**model_over):
    model = {"dim": 1, "model_params": "projected", "query_hidden": 64,
             "constrain_discrimination": "auto"}
    model.update(model_over)
    return {
        "seed": 0,
        "data": {"pathway": "irt"},
        "model": model,
        "train": {"loss": "soft_bce", "lr": 1e-2, "weight_decay": 0.0,
                  "batch_size": 512, "epochs": 3, "patience": 10, "val_metric": "bce"},
    }


# --------------------------------------------------------------------------- #
# model                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["projected", "free"])
@pytest.mark.parametrize("K", [1, 4])
def test_forward_shapes_and_range(mode, K):
    B = 32
    model = NIRTModel(query_dim=10, dim=K, n_models=5, model_params=mode,
                      query_hidden=8, profile_dim=7)
    e_q = torch.randn(B, 10)
    ref = torch.randn(B, 7) if mode == "projected" else torch.randint(0, 5, (B,))
    logit = model(e_q, ref)
    assert logit.shape == (B,)
    proba = model.predict_proba(e_q, ref)
    assert proba.shape == (B,)
    assert torch.all((proba >= 0) & (proba <= 1))


def test_constrain_discrimination_positive():
    model = NIRTModel(query_dim=4, dim=1, n_models=3, model_params="projected",
                      query_hidden=None, constrain_discrimination=True, profile_dim=4)
    a, _ = model.model_parameters(torch.randn(16, 4))
    assert torch.all(a > 0)


def test_from_config_auto_constraint():
    m1 = NIRTModel.from_config({"dim": 1}, n_models=2)
    m4 = NIRTModel.from_config({"dim": 4}, n_models=2)
    assert m1.constrain_discrimination is True
    assert m4.constrain_discrimination is False


# --------------------------------------------------------------------------- #
# multidimensional difficulty (b_m in R^K)                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["projected", "free"])
def test_vector_difficulty_shapes(mode):
    B, K = 20, 8
    m = NIRTModel(query_dim=10, dim=K, n_models=5, model_params=mode,
                  query_hidden=16, profile_dim=7, difficulty="vector")
    e_q = torch.randn(B, 10)
    ref = torch.randn(B, 7) if mode == "projected" else torch.randint(0, 5, (B,))
    a, b = m.model_parameters(ref)
    assert a.shape == (B, K) and b.shape == (B, K)          # b is now a K-vector
    assert m(e_q, ref).shape == (B,)
    p = m.predict_proba(e_q, ref)
    assert torch.all((p >= 0) & (p <= 1))


def test_scalar_difficulty_unchanged():
    """difficulty='scalar' is the default and bit-identical to before."""
    torch.manual_seed(0)
    m_default = NIRTModel(query_dim=8, dim=4, n_models=3, model_params="projected",
                          query_hidden=None, profile_dim=6)
    torch.manual_seed(0)
    m_explicit = NIRTModel(query_dim=8, dim=4, n_models=3, model_params="projected",
                           query_hidden=None, profile_dim=6, difficulty="scalar")
    e_q, ref = torch.randn(12, 8), torch.randn(12, 6)
    assert torch.allclose(m_default(e_q, ref), m_explicit(e_q, ref))
    _, b = m_default.model_parameters(ref)
    assert b.shape == (12,)


def test_difficulty_from_config_and_validation():
    assert NIRTModel.from_config({"dim": 4}, n_models=2).difficulty == "scalar"
    assert NIRTModel.from_config({"dim": 4, "difficulty": "vector"}, n_models=2).difficulty == "vector"
    assert build_model({"orientation": "query_latent", "dim": 4, "difficulty": "vector"},
                       n_models=3).difficulty == "vector"
    with pytest.raises(ValueError, match="difficulty"):
        NIRTModel(query_dim=4, dim=2, n_models=2, difficulty="bogus")


def test_vector_difficulty_recovers_and_checkpoints(tmp_path):
    """A world with per-dimension thresholds: vector difficulty should fit it and
    round-trip through a checkpoint."""
    train_obs, val_obs, q_store, m_store = _synthetic(seed=1, K=3)
    train_ds, val_ds = _datasets(train_obs, val_obs, q_store, m_store)
    cfg = _cfg(model_params="free", dim=3, query_hidden=None, difficulty="vector")
    cfg["train"].update(epochs=120, patience=25, lr=1e-2)
    res = fit(cfg, datasets=(train_ds, val_ds), name="cx-vec", runs_dir=tmp_path, verbose=False)

    from router.nirt.train import load_run

    model, loaded_cfg, midx = load_run("cx-vec", runs_dir=tmp_path)
    assert model.difficulty == "vector"
    _, y_prob = predict_dataset(model, val_ds, midx)
    rho = prediction_metrics(val_ds.obs["p_true"].to_numpy(), y_prob)["pearson_r"]
    assert rho > 0.8, rho


# --------------------------------------------------------------------------- #
# metrics                                                                     #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# training                                                                    #
# --------------------------------------------------------------------------- #
def test_synthetic_recovery():
    """A linear query head should recover theta_q (and thus the true P(correct))."""
    train_obs, val_obs, q_store, m_store = _synthetic(seed=1)
    train_ds, val_ds = _datasets(train_obs, val_obs, q_store, m_store)
    cfg = _cfg(model_params="free", dim=2, query_hidden=None)
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
    train_obs, val_obs, q_store, m_store = _synthetic(n_queries=200, seed=2)
    train_ds, val_ds = _datasets(train_obs, val_obs, q_store, m_store)
    res = fit(_cfg(model_params=mode), datasets=(train_ds, val_ds),
              name=f"t-{mode}", runs_dir=tmp_path, verbose=False)
    assert res.path == tmp_path / f"t-{mode}"
    assert (res.path / "model.pt").exists()
    assert (res.path / "run.json").exists()
    assert np.isfinite(res.val_metrics["bce"])
    assert len(res.history) >= 1

    from router.nirt.train import load_run

    model, cfg, model_index = load_run(f"t-{mode}", runs_dir=tmp_path)
    assert model.model_params == mode
    y, p = predict_dataset(model, val_ds, model_index)
    assert p.shape == y.shape and np.all((p >= 0) & (p <= 1))


def test_determinism():
    train_obs, val_obs, q_store, m_store = _synthetic(n_queries=200, seed=3)
    ds = _datasets(train_obs, val_obs, q_store, m_store)
    a = fit(_cfg(), datasets=ds, save=False, verbose=False)
    b = fit(_cfg(), datasets=ds, save=False, verbose=False)
    assert a.history[0]["train_loss"] == pytest.approx(b.history[0]["train_loss"])
    assert a.val_metrics["bce"] == pytest.approx(b.val_metrics["bce"])


def test_free_mode_rejects_unseen_model():
    train_obs, val_obs, q_store, m_store = _synthetic(n_queries=120, seed=4)
    train_ds, val_ds = _datasets(train_obs, val_obs, q_store, m_store)
    res = fit(_cfg(model_params="free"), datasets=(train_ds, val_ds), save=False, verbose=False)
    with pytest.raises(ValueError, match="cannot score models absent"):
        predict_dataset(res.model, val_ds, {"only-this": 0})


# --------------------------------------------------------------------------- #
# classical IRT baseline + routing                                            #
# --------------------------------------------------------------------------- #
def test_classical_irt_cell_beats_column_mean():
    from router.nirt.baselines import fit_classical_irt
    from router.nirt.metrics import prediction_metrics

    d = _FakeData(_splitted_obs(seed=7))
    r = fit_classical_irt(d, split="test", protocol="cell", dim=2,
                          holdout_frac=0.2, epochs=400, seed=7)
    # column-mean (per-model rate over the fitted cells) on the same held-out cells
    train_cells = (~np.isnan(r.true_matrix)) & ~r.heldout_mask
    col_mean = np.array([r.true_matrix[train_cells[:, j], j].mean() for j in range(r.true_matrix.shape[1])])
    hq, hm = np.nonzero(r.heldout_mask)
    base = prediction_metrics(r.true_matrix[hq, hm], col_mean[hm])

    assert r.metrics["bce"] < base["bce"]          # latent theta_q adds signal
    assert r.metrics["auc"] > 0.7
    assert r.pred_matrix.shape[1] == 9


def test_classical_irt_main_effects_matches_model_rate():
    from router.nirt.baselines import fit_classical_irt

    d = _FakeData(_splitted_obs(seed=8))
    r = fit_classical_irt(d, split="test", protocol="main_effects", dim=1, epochs=300, seed=8)
    obs = d.nirt_observations()
    fitrate = obs[obs.split.isin(["train", "validation"])].groupby("model_id").target.mean()
    # theta = 0 -> prediction is sigmoid(-b_m), a single value per model column
    col = r.pred_matrix[:, r.model_ids.index(fitrate.index[0])]
    assert np.allclose(col, col[0])
    assert abs(col[0] - fitrate.iloc[0]) < 0.1


def test_routing_report_and_lambda_tradeoff():
    from router.nirt.routing import eval_matrices, pareto, routing_report, train_quality

    d = _FakeData(_splitted_obs(seed=9))
    true_df, cost_df = eval_matrices(d, split="test")
    true, cost = true_df.to_numpy(float), cost_df.to_numpy(float)
    model_ids = list(true_df.columns)
    tq = train_quality(d, model_ids)

    rep = routing_report({"perfect": true}, true, cost, model_ids=model_ids,
                         train_quality=tq, reference=model_ids[-1])
    oracle = rep.iloc[0]
    assert oracle["policy"].startswith("oracle")
    # a policy routing on the true matrix should equal the oracle quality
    perfect = rep[rep.policy.str.startswith("route: perfect")].iloc[0]
    assert perfect["quality"] == pytest.approx(oracle["quality"])
    assert (rep["policy"] == f"fixed: {model_ids[-1]}").any()

    sweep = pareto(true, true, cost, lams=(0.0, 1.0, 5.0))
    assert sweep["cost"].iloc[-1] <= sweep["cost"].iloc[0] + 1e-9   # more lambda -> cheaper


def test_oracle_choice_breaks_ties_by_cost():
    from router.nirt.routing import oracle_choice

    true = np.array([[1.0, 1.0, 0.0], [0.0, 0.3, 0.3], [1.0, 1.0, 1.0]])
    cost = np.array([[9.0, 1.0, 2.0], [5.0, 8.0, 3.0], [4.0, 2.0, 7.0]])
    # cheapest model attaining the row max: col1 (1.0), col2 (0.3), col1 (2.0)
    assert oracle_choice(true, cost).tolist() == [1, 2, 1]
    # never worse quality than the plain argmax
    q_oracle = true[np.arange(3), oracle_choice(true, cost)]
    assert np.allclose(q_oracle, true.max(axis=1))


# --------------------------------------------------------------------------- #
# part 2a: Reward / AIQ / router baselines / OOD                              #
# --------------------------------------------------------------------------- #
def test_linear_cost_and_reward():
    from router.nirt.routing import linear_cost, reward

    pool = [0.0001, 0.001, 0.004]
    assert linear_cost(0.0001, pool) == pytest.approx(0.0)
    assert linear_cost(0.004, pool) == pytest.approx(1.0)
    # reward: up in quality, down in cost
    assert reward(0.8, 0.0001, 0.7, pool_mean_costs=pool) > reward(0.6, 0.0001, 0.7, pool_mean_costs=pool)
    assert reward(0.8, 0.0001, 0.7, pool_mean_costs=pool) > reward(0.8, 0.004, 0.7, pool_mean_costs=pool)


def test_aiq_improvement_sign():
    from router.nirt.routing import aiq

    pool_costs = np.array([0.001, 0.002, 0.004])
    pool_qual = np.array([0.4, 0.6, 0.8])
    # a frontier well above the baseline line -> positive improvement
    good = pd.DataFrame({"cost": [0.001, 0.002, 0.004], "quality": [0.7, 0.8, 0.85]})
    assert aiq(good, pool_mean_costs=pool_costs, pool_quality=pool_qual)["aiq_improvement"] > 0
    # a frontier ON the baseline line -> ~zero
    onln = pd.DataFrame({"cost": [0.001, 0.004], "quality": [0.4, 0.8]})
    assert abs(aiq(onln, pool_mean_costs=pool_costs, pool_quality=pool_qual)["aiq_improvement"]) < 0.02


def test_knn_router_matrix_shape_and_variation():
    from router.nirt.baselines import knn_router_matrix

    d = _FakeData(_splitted_obs(seed=11), q_dim=16)
    eval_ids = sorted(d._obs.query_id.unique())[:30]
    M = knn_router_matrix(d, eval_ids, k=5, pathway="irt")
    assert M.shape == (30, 9)
    assert np.all((M.to_numpy() >= 0) & (M.to_numpy() <= 1))
    assert M.to_numpy().std(0).mean() > 0        # not constant across queries


def test_mlp_router_runs():
    from router.nirt.baselines import fit_mlp_router, mlp_router_matrix

    d = _FakeData(_splitted_obs(seed=12), q_dim=16)
    model, mids = fit_mlp_router(d, hidden=16, epochs=3, pathway="irt", seed=1)
    M = mlp_router_matrix(model, mids, d, sorted(d._obs.query_id.unique())[:20], pathway="irt")
    assert M.shape == (20, 9)
    assert np.all((M.to_numpy() >= 0) & (M.to_numpy() <= 1))


def test_ood_split_real():
    from router.config import load_config
    from router.data.phase1 import load_phase1
    from router.nirt.ood import split_observations

    try:
        d = load_phase1(load_config())
    except FileNotFoundError:
        pytest.skip("processed tables not built")
    train_obs, ood_obs = split_observations(d, ("math", "code"))
    assert set(ood_obs["family"]) <= {"math", "code"}
    assert "math" not in set(train_obs["family"]) and "code" not in set(train_obs["family"])
    assert set(train_obs["split"]) <= {"train", "validation"}
    assert len(ood_obs) > 0 and len(train_obs) > 0


# --------------------------------------------------------------------------- #
# orientation: IRTRouterModel (model_latent) + build_model                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["projected", "free"])
@pytest.mark.parametrize("K", [1, 4])
def test_irtrouter_forward_shapes_and_range(mode, K):
    B = 24
    m = IRTRouterModel(query_dim=10, dim=K, n_models=5, model_params=mode,
                       query_hidden=8, profile_dim=7)
    e_q = torch.randn(B, 10)
    ref = torch.randn(B, 7) if mode == "projected" else torch.randint(0, 5, (B,))
    assert m(e_q, ref).shape == (B,)
    p = m.predict_proba(e_q, ref)
    assert torch.all((p >= 0) & (p <= 1))
    assert m.orientation == "model_latent"


def test_irtrouter_bounded_ability_and_softplus():
    m = IRTRouterModel(query_dim=6, dim=1, n_models=3, model_params="projected",
                       query_hidden=None, constrain_discrimination=True,
                       bound_ability=True, profile_dim=6)
    theta = m.latent_ability(torch.randn(12, 6))
    a, _ = m.query_parameters(torch.randn(12, 6))
    assert torch.all((theta >= 0) & (theta <= 1))
    assert torch.all(a > 0)


def test_build_model_orientation():
    assert build_model({"orientation": "query_latent", "dim": 2}, n_models=3).orientation == "query_latent"
    assert build_model({"orientation": "model_latent", "dim": 2}, n_models=3).orientation == "model_latent"
    assert build_model({"dim": 1}, n_models=3).orientation == "query_latent"   # default
    with pytest.raises(ValueError, match="orientation"):
        build_model({"orientation": "nonsense"}, n_models=3)


def test_synthetic_recovery_model_latent():
    train_obs, val_obs, q_store, m_store = _synthetic(seed=2, orientation="model_latent")
    train_ds, val_ds = _datasets(train_obs, val_obs, q_store, m_store)
    cfg = _cfg(orientation="model_latent", model_params="free", dim=2, query_hidden=None)
    cfg["train"].update(epochs=150, patience=30, lr=1e-2)
    res = fit(cfg, datasets=(train_ds, val_ds), save=False, verbose=False)
    _, y_prob = predict_dataset(res.model, val_ds, res.model_index)
    rho = prediction_metrics(val_ds.obs["p_true"].to_numpy(), y_prob)["pearson_r"]
    assert rho > 0.85, rho


def test_fit_load_run_model_latent(tmp_path):
    train_obs, val_obs, q_store, m_store = _synthetic(n_queries=200, seed=3, orientation="model_latent")
    ds = _datasets(train_obs, val_obs, q_store, m_store)
    res = fit(_cfg(orientation="model_latent", model_params="projected"),
              datasets=ds, name="irt-t", runs_dir=tmp_path, verbose=False)
    from router.nirt.train import load_run

    model, cfg, midx = load_run("irt-t", runs_dir=tmp_path)
    assert isinstance(model, IRTRouterModel)
    y, p = predict_dataset(model, ds[1], midx)
    assert np.all((p >= 0) & (p <= 1))


# --------------------------------------------------------------------------- #
# cold-start                                                                  #
# --------------------------------------------------------------------------- #
def test_cold_start_rejects_free_model():
    m = NIRTModel(query_dim=4, dim=1, n_models=2, model_params="free", profile_dim=4)
    with pytest.raises(ValueError, match="projected"):
        cold_start_eval(m, {"a": 0}, data=object())


def test_nearest_profile_model():
    mat = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    st = _store(["a", "b", "c"], mat, "model_id")
    assert nearest_profile_model(st, "a", ["b", "c"]) == "b"


def test_projected_model_scores_held_out_model():
    """The core cold-start capability: a projected model predicts a model that was
    never in training, from its profile embedding, better than the global mean."""
    train_obs, val_obs, q_store, m_store = _synthetic(
        n_queries=1400, n_models=8, seed=5, informative_profiles=True
    )
    held = {"m6", "m7"}
    tr = train_obs[~train_obs.model_id.isin(held)].reset_index(drop=True)
    train_ds, val_ds = _datasets(tr, val_obs, q_store, m_store)   # val_ds still has all 8
    cfg = _cfg(model_params="projected", dim=2)
    cfg["train"].update(epochs=120, patience=25, lr=1e-2)
    res = fit(cfg, datasets=(train_ds, val_ds), save=False, verbose=False)

    y, p = predict_dataset(res.model, val_ds, res.model_index)
    mids = val_ds.model_ids
    cold = np.isin(mids, list(held))
    cold_bce = prediction_metrics(y[cold], p[cold])["bce"]
    gm = float(tr.target.mean())
    gm_bce = prediction_metrics(y[cold], np.full(cold.sum(), gm))["bce"]
    assert cold_bce < gm_bce, (cold_bce, gm_bce)
