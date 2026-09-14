"""Oracle-routing evaluation (router.nirt.routing_eval)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from router.nirt.routing_eval import (
    compare_routing_strategies,
    oracle_classifier_matrix,
    oracle_labels,
    per_query_table,
    routing_decision,
    routing_evaluation,
    soft_oracle_targets,
)

M = ["GPT-A", "GPT-B", "GPT-C"]


def _mats():
    #                A     B     C
    true = np.array([[0.72, 0.94, 0.81],   # oracle = B
                     [0.50, 0.50, 0.20],   # oracle tie A & B
                     [0.10, 0.20, 0.20]])  # oracle tie B & C
    cost = np.array([[0.01, 0.03, 0.02],
                     [0.05, 0.02, 0.01],
                     [0.01, 0.09, 0.04]])
    q = ["q001", "q002", "q003"]
    return pd.DataFrame(true, index=q, columns=M), pd.DataFrame(cost, index=q, columns=M)


# --------------------------------------------------------------------------- #
# oracle generation                                                          #
# --------------------------------------------------------------------------- #
def test_oracle_labels_basic():
    true_df, cost_df = _mats()
    lab = oracle_labels(true_df, cost_df)
    q1 = lab[lab.query_id == "q001"].set_index("model_id")
    assert q1.loc["GPT-B", "oracle_score"] == pytest.approx(0.94)
    assert q1.loc["GPT-B", "oracle_flag"] == 1
    assert q1.loc["GPT-A", "oracle_flag"] == 0
    assert q1.loc["GPT-B", "oracle_rank"] == 1
    assert q1.loc["GPT-C", "oracle_rank"] == 2
    assert q1.loc["GPT-A", "oracle_rank"] == 3
    assert (q1["oracle_model_id"] == "GPT-B").all()
    assert (q1["n_oracle_ties"] == 1).all()


def test_oracle_labels_ties_are_all_flagged():
    true_df, cost_df = _mats()
    lab = oracle_labels(true_df, cost_df).set_index(["query_id", "model_id"])
    # q002: A and B both 0.50 -> both flagged, deterministic winner is the cheaper (B: .02 < A: .05)
    assert lab.loc[("q002", "GPT-A"), "oracle_flag"] == 1
    assert lab.loc[("q002", "GPT-B"), "oracle_flag"] == 1
    assert lab.loc[("q002", "GPT-C"), "oracle_flag"] == 0
    assert lab.loc[("q002", "GPT-A"), "n_oracle_ties"] == 2
    assert lab.loc[("q002", "GPT-A"), "oracle_model_id"] == "GPT-B"
    # tied models share rank 1
    assert lab.loc[("q002", "GPT-A"), "oracle_rank"] == 1
    assert lab.loc[("q002", "GPT-B"), "oracle_rank"] == 1


def test_oracle_labels_single_model_pool():
    true_df = pd.DataFrame([[0.3], [0.9]], index=["q1", "q2"], columns=["only"])
    lab = oracle_labels(true_df)
    assert (lab["oracle_flag"] == 1).all()
    assert (lab["oracle_model_id"] == "only").all()
    assert lab.loc[lab.query_id == "q2", "oracle_score"].iloc[0] == pytest.approx(0.9)


def test_oracle_is_pool_relative():
    """E0/E1/E2 -- the oracle is the best WITHIN the candidate pool passed in."""
    true_df, _ = _mats()
    full = oracle_labels(true_df).groupby("query_id")["oracle_score"].first()
    # drop GPT-B from the pool -> q001 oracle score must fall from .94 to .81
    reduced = oracle_labels(true_df[["GPT-A", "GPT-C"]]).groupby("query_id")["oracle_score"].first()
    assert full["q001"] == pytest.approx(0.94)
    assert reduced["q001"] == pytest.approx(0.81)


def test_soft_oracle_targets_sum_to_one():
    true_df, _ = _mats()
    s = soft_oracle_targets(true_df, tau=0.1)
    assert np.allclose(s.to_numpy().sum(axis=1), 1.0)
    # lower tau -> peakier: the best model gets more mass
    assert soft_oracle_targets(true_df, 0.02).iloc[0].max() > soft_oracle_targets(true_df, 1.0).iloc[0].max()


# --------------------------------------------------------------------------- #
# routing evaluation                                                         #
# --------------------------------------------------------------------------- #
def test_perfect_router_has_zero_regret():
    true_df, cost_df = _mats()
    true, cost = true_df.to_numpy(), cost_df.to_numpy()
    r = routing_evaluation(true.copy(), true, cost, M)  # predict the truth
    assert r["mean_regret"] == pytest.approx(0.0)
    assert r["zero_regret_rate"] == pytest.approx(1.0)
    assert r["oracle_hit_rate_any_best"] == pytest.approx(1.0)
    assert r["mean_selected_quality"] == pytest.approx(r["mean_oracle_quality"])


def test_bad_router_has_positive_regret():
    true_df, cost_df = _mats()
    true, cost = true_df.to_numpy(), cost_df.to_numpy()
    bad = -true  # deliberately pick the worst model
    r = routing_evaluation(bad, true, cost, M)
    assert r["mean_regret"] > 0.0
    assert r["oracle_hit_rate"] < 1.0


def test_hit_rate_and_regret_are_different_under_ties():
    true_df, cost_df = _mats()
    true, cost = true_df.to_numpy(), cost_df.to_numpy()
    # q002: A & B tie at .50 (oracle picks B, the cheaper). A router that always
    # picks A gets q002 exact-hit=0 but regret=0.
    pred = np.array([[0.0, 1.0, 0.0],    # q001 -> B (correct)
                     [1.0, 0.0, 0.0],    # q002 -> A (tied best, not the oracle model)
                     [0.0, 1.0, 0.0]])   # q003 -> B (tied best)
    r = routing_evaluation(pred, true, cost, M, query_ids=["q001", "q002", "q003"])
    assert r["oracle_hit_rate"] < 1.0            # missed the exact oracle model on q002
    assert r["oracle_hit_rate_any_best"] == pytest.approx(1.0)
    assert r["mean_regret"] == pytest.approx(0.0)
    pq = r["_per_query"]
    assert pq.loc[pq.query_id == "q002", "oracle_hit"].iloc[0] == 0
    assert pq.loc[pq.query_id == "q002", "oracle_hit_any_best"].iloc[0] == 1
    assert pq.loc[pq.query_id == "q002", "oracle_regret"].iloc[0] == pytest.approx(0.0)


def test_within_tolerance_rate():
    true = np.array([[1.0, 0.99, 0.5], [1.0, 0.2, 0.2]])
    pred = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # q0 -> B (0.01 short), q1 -> A (exact)
    r = routing_evaluation(pred, true, None, ["A", "B", "C"], tolerance=0.05)
    assert r["within_tolerance_rate"] == pytest.approx(1.0)
    assert r["zero_regret_rate"] == pytest.approx(0.5)
    r2 = routing_evaluation(pred, true, None, ["A", "B", "C"], tolerance=0.005)
    assert r2["within_tolerance_rate"] == pytest.approx(0.5)


def test_missing_prediction_is_never_selected():
    true = np.array([[0.9, 0.1, 0.5]])
    pred = np.array([[np.nan, 0.8, 0.2]])   # no prediction for the actually-best model
    sel = routing_decision(pred)
    assert sel[0] == 1                      # never the NaN cell
    r = routing_evaluation(pred, true, None, ["A", "B", "C"])
    # forced onto B (0.1); the un-predicted best model A (0.9) is unreachable
    assert r["mean_regret"] == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# cost-aware routing                                                         #
# --------------------------------------------------------------------------- #
def test_lambda_zero_reproduces_quality_routing():
    true_df, cost_df = _mats()
    true, cost = true_df.to_numpy(), cost_df.to_numpy()
    pred = np.array([[0.6, 0.9, 0.7], [0.4, 0.5, 0.3], [0.1, 0.15, 0.2]])
    a = routing_decision(pred, lam=0.0)
    b = routing_decision(pred, lam=0.0, model_costs=cost.mean(0))
    assert a.tolist() == b.tolist()


def test_increasing_lambda_can_change_selection():
    pred = np.array([[0.90, 0.88]])          # A slightly better
    model_costs = np.array([1.0, 0.0])       # but A is far more expensive
    assert routing_decision(pred, lam=0.0, model_costs=model_costs)[0] == 0
    assert routing_decision(pred, lam=0.1, model_costs=model_costs)[0] == 1


def test_cost_aware_utility_block():
    true = np.array([[1.0, 0.9]])
    cost = np.array([[0.10, 0.01]])
    pred = true.copy()
    r = routing_evaluation(pred, true, cost, ["A", "B"], lam=1.0)
    ca = r["cost_aware"]
    # U(A) = 1.0 - 1.0*0.10 = 0.9 ; U(B) = 0.9 - 1.0*0.01 = 0.89 -> oracle still A
    assert ca["mean_cost_aware_oracle_utility"] == pytest.approx(0.9)
    assert ca["mean_utility_regret"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# strategy comparison + classifier baseline                                  #
# --------------------------------------------------------------------------- #
def test_compare_strategies_includes_oracle_bound():
    true_df, cost_df = _mats()
    true, cost = true_df.to_numpy(), cost_df.to_numpy()
    preds = {"good": true.copy(), "bad": -true}
    summary, detail = compare_routing_strategies(preds, true, cost, M, lam=0.5,
                                                 query_ids=list(true_df.index))
    assert (summary["strategy"] == "hard oracle (upper bound)").any()
    hard = summary[summary.strategy == "hard oracle (upper bound)"].iloc[0]
    assert hard["mean_regret"] == pytest.approx(0.0)
    good = summary[summary.strategy == "good (quality)"].iloc[0]
    bad = summary[summary.strategy == "bad (quality)"].iloc[0]
    assert good["mean_regret"] <= bad["mean_regret"]
    # cost-aware rows present when lam and cost are given
    assert summary["strategy"].str.contains("cost-aware").any()
    assert "_per_query" not in detail["good (quality)"]


def test_oracle_classifier_baseline_trains_on_train_only():
    rng = np.random.default_rng(0)
    # 2 clusters of queries, each cluster's best model differs -> learnable
    n = 120
    emb = rng.standard_normal((n, 4))
    cluster = (emb[:, 0] > 0).astype(int)
    true = np.zeros((n, 3))
    for i in range(n):
        true[i, cluster[i]] = 1.0
        true[i, (cluster[i] + 1) % 3] = 0.3
    tr = slice(0, 90)
    ev = slice(90, n)
    tr_df = pd.DataFrame(true[tr], columns=["m0", "m1", "m2"],
                         index=[f"q{i}" for i in range(90)])
    mat = oracle_classifier_matrix(emb[tr], tr_df, emb[ev], ["m0", "m1", "m2"])
    assert mat.shape == (30, 3)
    r = routing_evaluation(mat, true[ev], None, ["m0", "m1", "m2"])
    assert r["oracle_hit_rate_any_best"] > 0.8   # clearly learnable signal


# --------------------------------------------------------------------------- #
# leakage: oracle info is evaluation-only                                     #
# --------------------------------------------------------------------------- #
def test_oracle_columns_absent_from_model_input_tensors():
    """The NIRT dataset / materialised arrays must expose only e_q, e_m, target,
    cost, metric, source -- never oracle_* / oracle_model_id / oracle_flag."""
    from router.data.nirt import NIRTDataset

    forbidden = {"oracle_flag", "oracle_model_id", "oracle_score", "oracle_rank"}
    obs = pd.DataFrame({
        "query_id": ["q0", "q0"], "model_id": ["m0", "m1"],
        "target": [1.0, 0.0], "metric_type": ["accuracy", "accuracy"],
        "source": ["s", "s"], "cost": [0.1, 0.2],
    })
    from router.embeddings.encoder import EmbeddingStore

    def _store(ids, dim, field):
        man = {"dim": dim, "id_field": field, "count": len(ids), "complete": True}
        return EmbeddingStore(list(ids), np.zeros((len(ids), dim), np.float32), man, id_field=field)

    ds = NIRTDataset(obs, _store(["q0"], 8, "query_id"), _store(["m0", "m1"], 8, "model_id"))
    assert not (set(ds[0]) & forbidden)
    assert not (set(ds.gather()) & forbidden)


def test_per_query_table_columns():
    true_df, cost_df = _mats()
    true, cost = true_df.to_numpy(), cost_df.to_numpy()
    from router.nirt.routing import oracle_choice

    sel = np.array([1, 0, 2])
    o = oracle_choice(true, cost)
    t = per_query_table(true.copy(), true, cost, M, list(true_df.index), sel, o)
    for c in ("query_id", "selected_model_id", "selected_predicted_score",
              "selected_actual_score", "oracle_model_id", "oracle_score",
              "oracle_hit", "oracle_regret"):
        assert c in t.columns
    assert len(t) == 3
