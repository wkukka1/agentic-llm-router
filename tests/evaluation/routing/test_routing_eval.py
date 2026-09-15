"""Oracle-routing evaluation (evaluation.routing.oracle, router.nirt.routing_decision)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import make_store

from evaluation.nirt.routing import oracle_choice
from evaluation.routing.oracle import (
    compare_routing_strategies,
    oracle_classifier_matrix,
    oracle_labels,
    per_query_table,
    per_task_argmax_baseline,
    routing_evaluation,
    soft_oracle_targets,
)
from router.nirt.routing_decision import routing_decision

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
# per-family fixed lookup (zero-parameter routing floor)                     #
# --------------------------------------------------------------------------- #
def test_per_task_argmax_baseline_picks_family_best_and_falls_back():
    # fam_a's best train model is GPT-A; fam_b's best is GPT-C; overall-best is
    # GPT-A (0.55 mean) -- distinct from fam_b's own pick, so an unseen family
    # must fall back to the GLOBAL best, not fam_b's.
    train_true = pd.DataFrame(
        {"GPT-A": [0.9, 0.9, 0.2, 0.2],
         "GPT-B": [0.5, 0.5, 0.5, 0.5],
         "GPT-C": [0.1, 0.1, 0.95, 0.95]},
        index=["t1", "t2", "t3", "t4"],
    )
    family_of = {"t1": "fam_a", "t2": "fam_a", "t3": "fam_b", "t4": "fam_b",
                 "e_a": "fam_a", "e_b": "fam_b", "e_unseen": "fam_c"}
    mat = per_task_argmax_baseline(train_true, ["e_a", "e_b", "e_unseen"], M, family_of)
    assert mat.shape == (3, 3)
    assert np.allclose(mat.sum(axis=1), 1.0)  # one-hot
    picked = [M[i] for i in mat.argmax(axis=1)]
    assert picked == ["GPT-A", "GPT-C", "GPT-A"]


def test_per_task_argmax_baseline_plugs_into_routing_evaluation():
    train_true = pd.DataFrame(
        {"GPT-A": [0.9, 0.9], "GPT-B": [0.5, 0.5], "GPT-C": [0.1, 0.1]},
        index=["t1", "t2"],
    )
    family_of = {"t1": "fam", "t2": "fam", "e1": "fam"}
    eval_true = np.array([[0.9, 0.5, 0.1]])
    mat = per_task_argmax_baseline(train_true, ["e1"], M, family_of)
    r = routing_evaluation(mat, eval_true, None, M)
    assert r["mean_regret"] == pytest.approx(0.0)


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
    ds = NIRTDataset(obs, make_store(["q0"], 8, "query_id"),
                     make_store(["m0", "m1"], 8, "model_id"))
    assert not (set(ds[0]) & forbidden)
    assert not (set(ds.gather()) & forbidden)


def test_per_query_table_columns():
    true_df, cost_df = _mats()
    true, cost = true_df.to_numpy(), cost_df.to_numpy()
    from evaluation.nirt.routing import oracle_choice

    sel = np.array([1, 0, 2])
    o = oracle_choice(true, cost)
    t = per_query_table(true.copy(), true, cost, M, list(true_df.index), sel, o)
    for c in ("query_id", "selected_model_id", "selected_predicted_score",
              "selected_actual_score", "oracle_model_id", "oracle_score",
              "oracle_hit", "oracle_regret"):
        assert c in t.columns
    assert len(t) == 3


# --------------------------------------------------------------------------- #
# one oracle tie-break rule (XD-02)                                          #
# --------------------------------------------------------------------------- #
def test_oracle_choice_tie_break_levels():
    ids = ["Z", "A"]
    # (a) the max score wins despite a higher cost
    assert oracle_choice(np.array([[0.9, 0.8]]), np.array([[5.0, 0.1]]), model_ids=ids)[0] == 0
    # (b) equal score -> the cheaper model, even when model_id order disagrees
    assert oracle_choice(np.array([[0.9, 0.9]]), np.array([[0.1, 5.0]]), model_ids=ids)[0] == 0
    # (c) equal score and equal cost -> smallest model_id ("A", column 1)
    assert oracle_choice(np.array([[0.9, 0.9]]), np.array([[0.1, 0.1]]), model_ids=ids)[0] == 1


def test_oracle_choice_legacy_mode_keeps_column_order():
    assert oracle_choice(np.array([[0.9, 0.9]]), np.array([[0.1, 0.1]]))[0] == 0


def test_oracle_choice_is_column_permutation_invariant():
    rng = np.random.default_rng(3)
    ids = ["m3", "m0", "m2", "m1", "m4"]
    true = rng.integers(0, 3, size=(200, 5)) / 2.0     # heavy score ties
    cost = rng.integers(0, 2, size=(200, 5)) * 0.01    # heavy cost ties
    base = [ids[k] for k in oracle_choice(true, cost, model_ids=ids)]
    for _ in range(5):
        perm = rng.permutation(5)
        p_ids = [ids[k] for k in perm]
        got = [p_ids[k] for k in oracle_choice(true[:, perm], cost[:, perm], model_ids=p_ids)]
        assert got == base


def test_oracle_labels_and_routing_evaluation_agree_on_oracle_model():
    rng = np.random.default_rng(4)
    ids = ["Z", "B", "A"]
    q = [f"q{i}" for i in range(50)]
    true = rng.integers(0, 2, size=(50, 3)).astype(float)
    cost = np.full((50, 3), 0.01)  # every cost tied -> model_id decides
    r = routing_evaluation(rng.random((50, 3)), true, cost, ids, query_ids=q)
    lab = oracle_labels(pd.DataFrame(true, index=q, columns=ids),
                        pd.DataFrame(cost, index=q, columns=ids))
    per_q = lab.groupby("query_id", sort=False)["oracle_model_id"].first().to_dict()
    assert r["_per_query"].set_index("query_id")["oracle_model_id"].to_dict() == per_q


# --------------------------------------------------------------------------- #
# ER-01: baseline D labels                                                    #
# --------------------------------------------------------------------------- #
def test_oracle_classifier_labels_ignore_pool_order():
    rng = np.random.default_rng(0)
    n = 200
    emb = rng.standard_normal((n, 4))
    cluster = emb[:, 0] > 0
    ids = ["m_exp", "m_bad", "m_cheap"]
    # cluster A: m_exp and m_cheap tie at 1.0 (m_cheap is cheaper, later column);
    # cluster B: m_bad is uniquely best (gives the classifier a second class)
    true = np.where(cluster[:, None], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    cost = np.tile([0.5, 0.1, 0.01], (n, 1))
    idx = [f"q{i}" for i in range(n)]
    ev = emb[emb[:, 0] > 1.0]

    def routed(order):
        o_ids = [ids[k] for k in order]
        tr = pd.DataFrame(true[:, order], index=idx, columns=o_ids)
        tc = pd.DataFrame(cost[:, order], index=idx, columns=o_ids)
        mat = oracle_classifier_matrix(emb, tr, ev, o_ids, train_cost_df=tc)
        return [o_ids[k] for k in mat.argmax(axis=1)]

    assert set(routed([0, 1, 2])) == {"m_cheap"}
    assert routed([0, 1, 2]) == routed([2, 1, 0])


# --------------------------------------------------------------------------- #
# ER-02 / ER-03: eligibility in the cost-aware oracle                         #
# --------------------------------------------------------------------------- #
def test_per_query_cost_aware_oracle_respects_eligibility():
    true = np.array([[1.0, 0.9, 0.2], [0.5, 0.4, 0.3]])
    cost = np.full((2, 3), 0.01)
    eligible = np.array([[False, True, True], [True, True, True]])
    r = routing_evaluation(true.copy(), true, cost, ["A", "B", "C"], lam=0.5,
                           eligible=eligible, query_ids=["q0", "q1"])
    pq = r["_per_query"]
    assert pq.loc[0, "cost_aware_oracle_model_id"] == "B"   # A is ineligible on q0
    assert r["cost_aware"]["cost_aware_oracle_model_mix"] == {"A": 1, "B": 1}
    assert pq["cost_aware_oracle_model_id"].value_counts().to_dict() == \
        r["cost_aware"]["cost_aware_oracle_model_mix"]


def test_all_ineligible_row_is_excluded_and_counted():
    true = np.array([[1.0, 0.5], [0.3, 0.9]])
    cost = np.array([[0.1, 0.2], [0.1, 0.2]])
    eligible = np.array([[False, False], [True, True]])
    r = routing_evaluation(true.copy(), true, cost, ["A", "B"], lam=1.0,
                           eligible=eligible, query_ids=["q0", "q1"])
    ca = r["cost_aware"]
    assert np.isfinite(ca["mean_utility_regret"])
    assert ca["n_queries_no_eligible"] == 1 and r["n_queries_no_eligible"] == 1
    assert ca["n_queries"] == ca["n_queries_no_eligible"] + ca["n_queries_evaluated_cost_aware"]
    # q1 only: U = [0.3 - 0.1, 0.9 - 0.2] -> oracle and router both pick B
    assert ca["mean_selected_utility"] == pytest.approx(0.7)
    assert ca["mean_utility_regret"] == pytest.approx(0.0)
    pq = r["_per_query"].set_index("query_id")
    assert pq.loc["q0", "has_eligible_candidate"] == 0
    assert pd.isna(pq.loc["q0", "cost_aware_oracle_model_id"])

    none_ok = routing_evaluation(true.copy(), true, cost, ["A", "B"], lam=1.0,
                                 eligible=np.zeros((2, 2), bool))["cost_aware"]
    assert none_ok["n_queries_evaluated_cost_aware"] == 0
    assert none_ok["mean_utility_regret"] is None
    assert none_ok["mean_selected_utility"] is None


# --------------------------------------------------------------------------- #
# ER-05: non-finite outcomes fail loudly and consistently                     #
# --------------------------------------------------------------------------- #
def test_nan_outcome_raises_value_error_everywhere():
    true_df, cost_df = _mats()
    true_df.iloc[1, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        oracle_labels(true_df, cost_df)
    with pytest.raises(ValueError, match="non-finite"):
        routing_evaluation(np.zeros((3, 3)), true_df.to_numpy(), cost_df.to_numpy(), M)


# --------------------------------------------------------------------------- #
# ER-06: vectorised oracle_labels == the old per-row implementation           #
# --------------------------------------------------------------------------- #
def _reference_oracle_labels(true_df, cost_df=None, tol=1e-9):
    """The pre-vectorisation per-row implementation, kept verbatim as a reference."""
    def _dense_rank_desc(scores):
        order = np.argsort(-scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.int64)
        cur, prev = 0, None
        for idx in order:
            if prev is None or (prev - scores[idx]) > tol:
                cur += 1
                prev = scores[idx]
            ranks[idx] = cur
        return ranks

    models = list(true_df.columns)
    true = true_df.to_numpy(np.float64)
    cost = (cost_df.reindex(index=true_df.index, columns=models).to_numpy(np.float64)
            if cost_df is not None else None)
    oracle_score = true.max(axis=1)
    flag = true >= oracle_score[:, None] - tol
    n_ties = flag.sum(axis=1)
    order_cost = cost if cost is not None else np.zeros_like(true)
    tiebreak = np.where(flag, order_cost, np.inf)
    oracle_model = []
    for i in range(len(true)):
        cands = np.flatnonzero(flag[i])
        oracle_model.append(models[cands[np.lexsort(([models[c] for c in cands],
                                                      tiebreak[i, cands]))[0]]])
    rows = []
    for i, qid in enumerate(true_df.index):
        ranks = _dense_rank_desc(true[i])
        for j, m in enumerate(models):
            rec = {"query_id": qid, "model_id": m, "actual_score": float(true[i, j]),
                   "oracle_score": float(oracle_score[i]), "oracle_flag": int(flag[i, j]),
                   "oracle_rank": int(ranks[j]), "oracle_model_id": oracle_model[i],
                   "n_oracle_ties": int(n_ties[i])}
            if cost is not None:
                rec["cost"] = float(cost[i, j])
            rows.append(rec)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", [0, 1])
def test_vectorised_oracle_labels_match_reference(seed):
    rng = np.random.default_rng(seed)
    ids = ["m2", "m0", "m1", "m3"]
    q = [f"q{i}" for i in range(60)]
    true = pd.DataFrame(rng.integers(0, 4, size=(60, 4)) / 3.0, index=q, columns=ids)
    cost = pd.DataFrame(rng.integers(1, 3, size=(60, 4)) * 0.01, index=q, columns=ids)
    for c in (cost, None):
        pd.testing.assert_frame_equal(oracle_labels(true, c), _reference_oracle_labels(true, c),
                                      check_dtype=False)


def test_vectorised_oracle_labels_match_reference_on_mats():
    true_df, cost_df = _mats()
    pd.testing.assert_frame_equal(oracle_labels(true_df, cost_df),
                                  _reference_oracle_labels(true_df, cost_df), check_dtype=False)


@pytest.mark.parametrize("row, expected", [
    ([1.0, 1.0 - 0.6e-9, 1.0 - 1.2e-9], [1, 1, 2]),
    ([1.0, 1.0 - 0.9e-9, 1.0 - 1.8e-9, 1.0 - 2.7e-9], [1, 1, 2, 2]),
])
def test_oracle_rank_uses_group_anchor_ties(row, expected):
    cols = [f"m{j}" for j in range(len(row))]
    true_df = pd.DataFrame([row], index=["q"], columns=cols)
    assert oracle_labels(true_df, tol=1e-9)["oracle_rank"].tolist() == expected
    assert _reference_oracle_labels(true_df, tol=1e-9)["oracle_rank"].tolist() == expected
