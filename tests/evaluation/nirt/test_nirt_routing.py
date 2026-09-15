"""Classical IRT baseline (``evaluation.baselines.classical_irt``), the routing report /
Pareto sweep / cost-aware oracle (``evaluation.nirt.routing``), the kNN + MLP router
baselines, and the OOD family split (``evaluation.nirt.ood``).

Fixtures come from ``helpers``: ``FakeTrainingData`` wraps a ``make_split_obs`` table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import FakeTrainingData, make_split_obs


# --------------------------------------------------------------------------- #
# classical IRT baseline                                                       #
# --------------------------------------------------------------------------- #
def test_classical_irt_cell_beats_column_mean():
    from evaluation.baselines.classical_irt import fit_classical_irt
    from training.nirt.metrics import prediction_metrics

    d = FakeTrainingData(make_split_obs(seed=7))
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
    from evaluation.baselines.classical_irt import fit_classical_irt

    d = FakeTrainingData(make_split_obs(seed=8))
    r = fit_classical_irt(d, split="test", protocol="main_effects", dim=1, epochs=300, seed=8)
    obs = d.nirt_observations()
    fitrate = obs[obs.split.isin(["train", "validation"])].groupby("model_id").target.mean()
    # theta = 0 -> prediction is sigmoid(-b_m), a single value per model column
    col = r.pred_matrix[:, r.model_ids.index(fitrate.index[0])]
    assert np.allclose(col, col[0])
    assert abs(col[0] - fitrate.iloc[0]) < 0.1


def test_classical_irt_main_effects_does_not_fit_scored_split():
    from evaluation.baselines.classical_irt import fit_classical_irt

    d = FakeTrainingData(make_split_obs(n_queries=200, seed=8))
    obs = d.nirt_observations()
    r = fit_classical_irt(d, split="validation", protocol="main_effects", epochs=20, seed=8)
    val_ids = set(obs.loc[obs.split == "validation", "query_id"])
    assert len(r.query_ids) == len(set(r.query_ids))          # no query appears twice
    fitted = {q for q, row in zip(r.query_ids, r.heldout_mask) if not row.any()}
    assert not fitted & val_ids                                 # scored split never trained on
    assert {q for q, row in zip(r.query_ids, r.heldout_mask) if row.any()} == val_ids


def test_classical_irt_main_effects_rejects_bad_args():
    from evaluation.baselines.classical_irt import fit_classical_irt

    d = FakeTrainingData(make_split_obs(n_queries=50, seed=8))
    with pytest.raises(ValueError, match="split"):
        fit_classical_irt(d, split="bogus", protocol="main_effects", epochs=1)
    with pytest.raises(ValueError, match="query_ids"):
        fit_classical_irt(d, protocol="main_effects", query_ids=["q0"], epochs=1)


# --------------------------------------------------------------------------- #
# routing report + lambda trade-off                                            #
# --------------------------------------------------------------------------- #
def test_routing_report_and_lambda_tradeoff():
    from evaluation.nirt.routing import eval_matrices, pareto, routing_report, train_quality

    d = FakeTrainingData(make_split_obs(seed=9))
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
    from evaluation.nirt.routing import oracle_choice

    true = np.array([[1.0, 1.0, 0.0], [0.0, 0.3, 0.3], [1.0, 1.0, 1.0]])
    cost = np.array([[9.0, 1.0, 2.0], [5.0, 8.0, 3.0], [4.0, 2.0, 7.0]])
    # cheapest model attaining the row max: col1 (1.0), col2 (0.3), col1 (2.0)
    assert oracle_choice(true, cost).tolist() == [1, 2, 1]
    # never worse quality than the plain argmax
    q_oracle = true[np.arange(3), oracle_choice(true, cost)]
    assert np.allclose(q_oracle, true.max(axis=1))


# --------------------------------------------------------------------------- #
# Reward / AIQ / router baselines / OOD                                        #
# --------------------------------------------------------------------------- #
def test_linear_cost_and_reward():
    from evaluation.nirt.routing import linear_cost, reward

    pool = [0.0001, 0.001, 0.004]
    assert linear_cost(0.0001, pool) == pytest.approx(0.0)
    assert linear_cost(0.004, pool) == pytest.approx(1.0)
    # reward: up in quality, down in cost
    assert reward(0.8, 0.0001, 0.7, pool_mean_costs=pool) > reward(0.6, 0.0001, 0.7, pool_mean_costs=pool)
    assert reward(0.8, 0.0001, 0.7, pool_mean_costs=pool) > reward(0.8, 0.004, 0.7, pool_mean_costs=pool)


def test_aiq_improvement_sign():
    from evaluation.nirt.routing import aiq

    pool_costs = np.array([0.001, 0.002, 0.004])
    pool_qual = np.array([0.4, 0.6, 0.8])
    # a frontier well above the baseline line -> positive improvement
    good = pd.DataFrame({"cost": [0.001, 0.002, 0.004], "quality": [0.7, 0.8, 0.85]})
    assert aiq(good, pool_mean_costs=pool_costs, pool_quality=pool_qual)["aiq_improvement"] > 0
    # a frontier ON the baseline line -> ~zero
    onln = pd.DataFrame({"cost": [0.001, 0.004], "quality": [0.4, 0.8]})
    assert abs(aiq(onln, pool_mean_costs=pool_costs, pool_quality=pool_qual)["aiq_improvement"]) < 0.02


def test_knn_router_matrix_shape_and_variation():
    from router.nirt.baselines_infer import knn_router_matrix

    d = FakeTrainingData(make_split_obs(seed=11), query_dim=16)
    eval_ids = sorted(d.observations.query_id.unique())[:30]
    M = knn_router_matrix(d, eval_ids, k=5, pathway="irt")
    assert M.shape == (30, 9)
    assert np.all((M.to_numpy() >= 0) & (M.to_numpy() <= 1))
    assert M.to_numpy().std(0).mean() > 0        # not constant across queries


def test_mlp_router_runs():
    from router.nirt.baselines_infer import mlp_router_matrix
    from training.trainers.mlp_router import fit_mlp_router

    d = FakeTrainingData(make_split_obs(seed=12), query_dim=16)
    model, mids = fit_mlp_router(d, hidden=16, epochs=3, pathway="irt", seed=1)
    M = mlp_router_matrix(model, mids, d, sorted(d.observations.query_id.unique())[:20], pathway="irt")
    assert M.shape == (20, 9)
    assert np.all((M.to_numpy() >= 0) & (M.to_numpy() <= 1))


def test_ood_split_real():
    from router.config import load_config
    from training.data.facade import load_training_data
    from evaluation.nirt.ood import split_observations

    try:
        d = load_training_data(load_config())
    except FileNotFoundError:
        pytest.skip("processed tables not built")
    train_obs, ood_obs = split_observations(d, ("math", "code"))
    assert set(ood_obs["family"]) <= {"math", "code"}
    assert "math" not in set(train_obs["family"]) and "code" not in set(train_obs["family"])
    assert set(train_obs["split"]) <= {"train", "validation"}
    assert len(ood_obs) > 0 and len(train_obs) > 0
