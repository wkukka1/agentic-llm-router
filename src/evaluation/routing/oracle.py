"""Oracle-routing evaluation -- kept SEPARATE from predictive (NLL / calibration).

A NIRT / ZOIB model is a **response model**: it estimates ``E[Y | q, m]``. The
**router** then picks ``m_hat(q) = argmax_m E[Y | q, m]``
(:func:`router.nirt.routing_decision.routing_decision`). The **oracle**
``m*(q) = argmax_m y(q, m)`` is an **evaluation target only** -- it is derived
from *observed* outcomes and must never enter training, embeddings, features,
hyper-parameter selection, or candidate selection (see the leakage rules in
``docs/routing_evaluation.md``).

Pipeline this module scores::

    response model  ->  P(Y | q, m)  ->  routing decision  ->  selected model
                                                                    |
                                             actual test outcome  <-+
                                                    |
                                          compare against the oracle

`lower NLL != better router`: a model can predict probabilities better yet route
worse. So routing is reported with its own metrics -- oracle hit rate, regret
(mean / median / p90), zero-regret rate, within-tolerance rate, selected vs
oracle quality -- and, separately, a **cost-aware** oracle
``U(q, m) = y(q, m) - lambda * C(m)``.

Everything here is pure numpy / pandas (the optional oracle-*classifier* baseline
lazily imports scikit-learn). No torch, no training of the response model.

Tie policy
----------
``oracle_flag`` is tie-*inclusive*: ``oracle_flag = int(actual >= oracle_score -
tol)``, so every model that attains a query's best score is flagged. A single
``oracle_model_id`` is still reported for convenience, chosen deterministically
(cheapest flagged model, then lexicographically smallest ``model_id``) -- never
left to pandas / dict ordering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd

from router.nirt.routing_decision import routing_decision

from ..nirt.routing import oracle_choice

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from router.routing.base import Router

_TOL = 1e-9


def evaluate_router(
    router,
    true_df: pd.DataFrame,
    cost_df: Optional[pd.DataFrame] = None,
    *,
    lam: float = 0.0,
    tolerance: float = 0.01,
    **kwargs,
) -> dict:
    """Score ``router`` against the oracle on a dense outcome matrix.

    ``true_df`` is ``[query_id x model_id]`` of the **observed** target and
    ``cost_df`` the matching per-cell cost (both restricted to ``router``'s
    pool). Replaces the old ``Router.evaluate()`` method -- a serving class
    must not depend on ground truth, so this lives here instead. Delegates to
    :func:`routing_evaluation` -- returns oracle hit rate, regret quantiles,
    selected vs oracle quality / cost, and (with ``cost_df``) the cost-aware
    oracle block.
    """
    scores = router.aligned_scores(list(true_df.index))
    model_ids = router.model_ids
    true = true_df.reindex(columns=model_ids).to_numpy(np.float64)
    cost = (
        cost_df.reindex(index=true_df.index, columns=model_ids).to_numpy(np.float64)
        if cost_df is not None
        else None
    )
    return routing_evaluation(
        scores.to_numpy(np.float64),
        true,
        cost,
        model_ids,
        lam=lam,
        tolerance=tolerance,
        query_ids=list(true_df.index),
        strategy=router.name,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# oracle labels (evaluation-only)                                             #
# --------------------------------------------------------------------------- #
def _dense_rank_desc(scores: np.ndarray, tol: float = _TOL) -> np.ndarray:
    """1-indexed dense rank of ``scores`` (1 = best); tied scores share a rank."""
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.int64)
    cur = 0
    prev = None
    for pos, idx in enumerate(order):
        if prev is None or (prev - scores[idx]) > tol:
            cur += 1
            prev = scores[idx]
        ranks[idx] = cur
    return ranks


def oracle_labels(
    true_df: pd.DataFrame,
    cost_df: Optional[pd.DataFrame] = None,
    *,
    tol: float = _TOL,
) -> pd.DataFrame:
    """Long oracle-label table for one candidate pool / split.

    ``true_df`` is a dense ``[query_id x model_id]`` matrix of the **actual**
    observed target (normally the chance-corrected graded score ``y_soft`` --
    the same target the router is trained/evaluated against). The oracle is
    computed **within this pool only** -- pass the E0 / E1 / E2 model subset in
    ``true_df.columns`` and you get that pool's oracle, not a global one.

    Columns: ``query_id, model_id, actual_score[, cost], oracle_score,
    oracle_flag, oracle_rank, oracle_model_id, n_oracle_ties``.
    """
    models = list(true_df.columns)
    true = true_df.to_numpy(np.float64)
    cost = (
        cost_df.reindex(index=true_df.index, columns=models).to_numpy(np.float64)
        if cost_df is not None else None
    )
    oracle_score = true.max(axis=1)
    flag = (true >= oracle_score[:, None] - tol)
    n_ties = flag.sum(axis=1)

    # deterministic single oracle model: cheapest flagged, then smallest id
    order_cost = cost if cost is not None else np.zeros_like(true)
    tiebreak = np.where(flag, order_cost, np.inf)
    oracle_col = np.empty(len(true), dtype=np.int64)
    for i in range(len(true)):
        cands = np.flatnonzero(flag[i])
        best = cands[np.lexsort(([models[c] for c in cands], tiebreak[i, cands]))[0]]
        oracle_col[i] = best
    oracle_model = [models[c] for c in oracle_col]

    rows = []
    for i, qid in enumerate(true_df.index):
        ranks = _dense_rank_desc(true[i], tol)
        for j, m in enumerate(models):
            rec = {
                "query_id": qid,
                "model_id": m,
                "actual_score": float(true[i, j]),
                "oracle_score": float(oracle_score[i]),
                "oracle_flag": int(flag[i, j]),
                "oracle_rank": int(ranks[j]),
                "oracle_model_id": oracle_model[i],
                "n_oracle_ties": int(n_ties[i]),
            }
            if cost is not None:
                rec["cost"] = float(cost[i, j])
            rows.append(rec)
    return pd.DataFrame(rows)


def soft_oracle_targets(true_df: pd.DataFrame, tau: float = 0.1) -> pd.DataFrame:
    """``p*(m | q) = softmax(y(q, m) / tau)`` -- a distributional oracle that keeps
    near-tie information instead of forcing one hard winner. For the optional
    soft-distillation baseline (§E). Rows sum to 1."""
    z = true_df.to_numpy(np.float64) / max(float(tau), 1e-6)
    w = np.exp(z - z.max(axis=1, keepdims=True))
    return pd.DataFrame(w / w.sum(axis=1, keepdims=True),
                        index=true_df.index, columns=true_df.columns)


def _candidate_counts(eligible: Optional[np.ndarray], n_q: int, n_m: int) -> dict:
    if eligible is None:
        return {"mean": float(n_m), "min": int(n_m), "max": int(n_m)}
    c = np.asarray(eligible, bool).sum(axis=1)
    return {"mean": float(c.mean()), "min": int(c.min()), "max": int(c.max())}


# --------------------------------------------------------------------------- #
# routing evaluation                                                         #
# --------------------------------------------------------------------------- #
def routing_evaluation(
    pred: np.ndarray,
    true: np.ndarray,
    cost: Optional[np.ndarray],
    model_ids: Sequence[str],
    *,
    lam: float = 0.0,
    tolerance: float = 0.01,
    model_costs: Optional[np.ndarray] = None,
    eligible: Optional[np.ndarray] = None,
    query_ids: Optional[Sequence[str]] = None,
    strategy: str = "nirt_quality",
) -> dict:
    """Score one predicted ``[Q, M]`` matrix as a router against the oracle.

    Quality routing uses ``argmax_m pred``. If ``lam > 0`` the selection is
    cost-aware (``argmax_m [pred - lam * C(m)]``). Regardless of ``lam``, the
    cost-aware **oracle** block is reported whenever ``cost`` is available.

    ``oracle_hit_rate`` (exact: selected == the deterministic cost-broken oracle)
    and ``mean_regret`` are *different* -- with near-tied models you can miss the
    oracle (hit = 0) at ~0 regret. ``oracle_hit_rate_any_best`` is the tie-aware
    variant (selected attains the best score).
    """
    pred = np.asarray(pred, np.float64)
    true = np.asarray(true, np.float64)
    n_q, n_m = true.shape
    model_ids = list(model_ids)
    qi = np.arange(n_q)
    C = (np.asarray(model_costs, np.float64) if model_costs is not None
         else (cost.mean(axis=0) if cost is not None else None))

    selected = routing_decision(pred, lam=lam, model_costs=C, eligible=eligible)
    sel_quality = true[qi, selected]

    o_idx = oracle_choice(true, cost if cost is not None else np.zeros_like(true))
    oracle_quality = true.max(axis=1)
    regret = np.maximum(oracle_quality - sel_quality, 0.0)

    hit_exact = float(np.mean(selected == o_idx))
    hit_any_best = float(np.mean(sel_quality >= oracle_quality - _TOL))

    out: dict = {
        "strategy": strategy,
        "lam": float(lam),
        "tolerance": float(tolerance),
        "n_queries": int(n_q),
        "n_candidates": _candidate_counts(eligible, n_q, n_m),
        "oracle_hit_rate": hit_exact,
        "oracle_hit_rate_any_best": hit_any_best,
        "mean_regret": float(regret.mean()),
        "median_regret": float(np.median(regret)),
        "p90_regret": float(np.percentile(regret, 90)),
        "max_regret": float(regret.max()),
        "zero_regret_rate": float(np.mean(regret <= _TOL)),
        "within_tolerance_rate": float(np.mean(regret <= tolerance)),
        "mean_selected_quality": float(sel_quality.mean()),
        "mean_oracle_quality": float(oracle_quality.mean()),
        "selected_model_mix": {
            model_ids[k]: int((selected == k).sum())
            for k in range(n_m) if (selected == k).any()
        },
    }

    if cost is not None:
        cost = np.asarray(cost, np.float64)
        sel_cost = cost[qi, selected]
        out["mean_selected_cost"] = float(sel_cost.mean())
        out["mean_selected_cost_per_1k"] = float(sel_cost.mean() * 1000)
        # --- cost-aware oracle: U(q, m) = y(q, m) - lam * C(m) ----------------
        util = true - float(lam) * C[None, :]
        if eligible is not None:
            util = np.where(np.asarray(eligible, bool), util, -np.inf)
        ca_idx = util.argmax(axis=1)
        ca_oracle_util = util[qi, ca_idx]
        sel_util = util[qi, selected]
        out["cost_aware"] = {
            "lam": float(lam),
            "cost_aware_oracle_model_mix": {
                model_ids[k]: int((ca_idx == k).sum())
                for k in range(n_m) if (ca_idx == k).any()
            },
            "mean_cost_aware_oracle_utility": float(ca_oracle_util.mean()),
            "mean_selected_utility": float(sel_util.mean()),
            "mean_utility_regret": float(np.maximum(ca_oracle_util - sel_util, 0.0).mean()),
            "quality_oracle_mean_quality": float(oracle_quality.mean()),
            "quality_oracle_mean_cost_per_1k": float(cost[qi, o_idx].mean() * 1000),
            "cost_aware_oracle_mean_quality": float(true[qi, ca_idx].mean()),
            "cost_aware_oracle_mean_cost_per_1k": float(cost[qi, ca_idx].mean() * 1000),
        }

    if query_ids is not None:
        out["_per_query"] = per_query_table(
            pred, true, cost, model_ids, list(query_ids), selected, o_idx,
            lam=lam, model_costs=C,
        )
    return out


def per_query_table(
    pred: np.ndarray,
    true: np.ndarray,
    cost: Optional[np.ndarray],
    model_ids: Sequence[str],
    query_ids: Sequence[str],
    selected: np.ndarray,
    oracle_idx: np.ndarray,
    *,
    lam: float = 0.0,
    model_costs: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """One row per query: what the router picked, what the oracle picked, the gap."""
    model_ids = list(model_ids)
    qi = np.arange(len(query_ids))
    true = np.asarray(true, np.float64)
    oracle_quality = true.max(axis=1)
    sel_q = true[qi, selected]
    rows = {
        "query_id": list(query_ids),
        "selected_model_id": [model_ids[k] for k in selected],
        "selected_predicted_score": np.asarray(pred, np.float64)[qi, selected],
        "selected_actual_score": sel_q,
        "oracle_model_id": [model_ids[k] for k in oracle_idx],
        "oracle_score": oracle_quality,
        "oracle_hit": (selected == oracle_idx).astype(int),
        "oracle_hit_any_best": (sel_q >= oracle_quality - _TOL).astype(int),
        "oracle_regret": np.maximum(oracle_quality - sel_q, 0.0),
    }
    if cost is not None:
        cost = np.asarray(cost, np.float64)
        rows["selected_cost"] = cost[qi, selected]
        if model_costs is not None:
            C = np.asarray(model_costs, np.float64)
            util = true - float(lam) * C[None, :]
            ca_idx = util.argmax(axis=1)
            rows["cost_aware_oracle_model_id"] = [model_ids[k] for k in ca_idx]
            rows["selected_utility"] = util[qi, selected]
            rows["cost_aware_oracle_utility"] = util[qi, ca_idx]
    return pd.DataFrame(rows)


def per_query_model_table(
    pred: np.ndarray, true: np.ndarray, cost: Optional[np.ndarray],
    model_ids: Sequence[str], query_ids: Sequence[str],
) -> pd.DataFrame:
    """Full ``[Q x M]`` long table (predicted_score, actual_score, model_rank,
    oracle_rank) for later ranking analysis. Optional -- large."""
    lab = oracle_labels(
        pd.DataFrame(true, index=list(query_ids), columns=list(model_ids)),
        pd.DataFrame(cost, index=list(query_ids), columns=list(model_ids)) if cost is not None else None,
    )
    pred_long = pd.DataFrame(pred, index=list(query_ids), columns=list(model_ids)) \
        .stack().rename("predicted_score").reset_index()
    pred_long.columns = ["query_id", "model_id", "predicted_score"]
    m = lab.merge(pred_long, on=["query_id", "model_id"], how="left")
    m["model_rank"] = m.groupby("query_id")["predicted_score"].rank(
        ascending=False, method="dense"
    ).astype("Int64")
    return m


# --------------------------------------------------------------------------- #
# strategy comparison (§9)                                                    #
# --------------------------------------------------------------------------- #
def compare_routing_strategies(
    preds: dict[str, np.ndarray],
    true: np.ndarray,
    cost: Optional[np.ndarray],
    model_ids: Sequence[str],
    *,
    lam: float = 0.0,
    tolerance: float = 0.01,
    query_ids: Optional[Sequence[str]] = None,
    include_hard_oracle: bool = True,
    include_random: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Score several routers side by side.

    ``preds`` maps a name -> predicted ``[Q, M]`` matrix. Each is evaluated as
    (A) a quality router and, if ``cost`` is given, (B) a cost-aware router at
    ``lam``. (C) the hard-oracle upper bound (route on ``true``) and a random
    router are added for context. Returns ``(summary_df, per_strategy_dict)``.
    """
    true = np.asarray(true, np.float64)
    n_m = true.shape[1]
    strategies: dict[str, dict] = {}

    def _run(name, pmat, strat_lam):
        strategies[name] = routing_evaluation(
            pmat, true, cost, model_ids, lam=strat_lam, tolerance=tolerance,
            query_ids=query_ids, strategy=name,
        )

    for name, pmat in preds.items():
        _run(f"{name} (quality)", pmat, 0.0)
        if cost is not None and lam:
            _run(f"{name} (cost-aware lam={lam})", pmat, lam)

    if include_hard_oracle:
        # route on the truth; break exact-quality ties toward the cheaper model
        # so the upper bound is also the cheapest way to reach max quality.
        tb = true.astype(np.float64).copy()
        if cost is not None:
            c = np.asarray(cost, np.float64)
            tb = tb - 1e-9 * (c - c.min()) / (np.ptp(c) + 1e-12)
        _run("hard oracle (upper bound)", tb, 0.0)
    if include_random:
        rng = np.random.default_rng(0)
        _run("random", rng.random(true.shape), 0.0)

    cols = ["strategy", "oracle_hit_rate", "oracle_hit_rate_any_best", "mean_regret",
            "median_regret", "p90_regret", "zero_regret_rate", "within_tolerance_rate",
            "mean_selected_quality", "mean_oracle_quality"]
    if cost is not None:
        cols += ["mean_selected_cost_per_1k"]
    summary = pd.DataFrame([
        {c: (s.get("strategy") if c == "strategy" else s.get(c)) for c in cols}
        for s in strategies.values()
    ])
    for s in strategies.values():
        s.pop("_per_query", None)
    return summary, strategies


# --------------------------------------------------------------------------- #
# per-family fixed lookup -- zero-parameter routing floor                     #
# --------------------------------------------------------------------------- #
def per_task_argmax_baseline(
    train_true_df: pd.DataFrame,
    eval_query_ids: Sequence[str],
    model_ids: Sequence[str],
    family_of: dict,
) -> np.ndarray:
    """Zero-parameter routing floor: for each benchmark family, pick the single
    model with the best mean **train** score, then apply that fixed lookup to
    every eval query by its family. A learned router that can't beat this has
    learned nothing beyond "which family is this" -- it is a materially harder
    bar than "always the single best-average model" (the fixed-model row in
    :func:`compare_routing_strategies`), which has no per-family granularity.

    ``family_of`` maps ``query_id -> family label`` (e.g.
    ``evaluation.nirt.ood.family_of_query(data)``, or a raw ``dataset`` column
    turned into a dict). Eval queries whose family is unknown, or was never
    observed at train time, fall back to the single best train-average model
    overall.

    Returns an ``[len(eval_query_ids), len(model_ids)]`` one-hot score matrix
    (1.0 at the picked model) -- the same pluggable contract as
    :func:`oracle_classifier_matrix`, ready for ``routing_evaluation`` /
    ``compare_routing_strategies``.
    """
    model_ids = list(model_ids)
    train_true_df = train_true_df.reindex(columns=model_ids)

    train_family = pd.Series(
        {qid: family_of.get(qid) for qid in train_true_df.index}, name="family"
    ).dropna()
    global_best = train_true_df.mean(axis=0).idxmax()

    best_by_family: dict = {}
    for fam, qids in train_family.groupby(train_family).groups.items():
        best_by_family[fam] = train_true_df.loc[list(qids)].mean(axis=0).idxmax()

    col = {m: j for j, m in enumerate(model_ids)}
    out = np.zeros((len(eval_query_ids), len(model_ids)), dtype=np.float64)
    for i, qid in enumerate(eval_query_ids):
        pick = best_by_family.get(family_of.get(qid), global_best)
        out[i, col[pick]] = 1.0
    return out


# --------------------------------------------------------------------------- #
# optional baseline D -- direct hard-oracle classifier                        #
# --------------------------------------------------------------------------- #
def oracle_classifier_matrix(
    train_q_emb: np.ndarray,
    train_true_df: pd.DataFrame,
    eval_q_emb: np.ndarray,
    model_ids: Sequence[str],
    *,
    seed: int = 42,
) -> np.ndarray:
    """A separate baseline: multinomial logistic regression that predicts the
    oracle model id from the query embedding, trained **only on the train split**
    (labels = ``argmax_m y_train(q, m)``). Returns an ``[Q_eval, M]``
    class-probability matrix that slots straight into ``routing_evaluation`` /
    ``compare_routing_strategies``. This is NOT the NIRT response model.
    """
    from sklearn.linear_model import LogisticRegression

    model_ids = list(model_ids)
    y_train = train_true_df.reindex(columns=model_ids).to_numpy(np.float64).argmax(axis=1)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(np.asarray(train_q_emb, np.float64), y_train)

    proba = clf.predict_proba(np.asarray(eval_q_emb, np.float64))
    full = np.zeros((proba.shape[0], len(model_ids)), dtype=np.float64)
    full[:, clf.classes_] = proba
    return full


# --------------------------------------------------------------------------- #
# compare several routers against the oracle                                  #
# --------------------------------------------------------------------------- #
def compare_routers(
    routers: Sequence["Router"],
    true_df: pd.DataFrame,
    cost_df: Optional[pd.DataFrame] = None,
    *,
    lam: float = 0.0,
    tolerance: float = 0.01,
    include_hard_oracle: bool = True,
    include_random: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Score several routers side by side against the oracle on one outcome matrix.

    Each router's predicted-quality matrix (over ``true_df``'s queries, aligned to
    ``true_df.columns``) is handed to :func:`compare_routing_strategies`, which
    adds the hard-oracle upper bound and a random floor. Returns
    ``(summary_df, detail)``. Replaces the old ``router.routing.compare_routers``
    -- comparing against ground truth is an evaluation concern.
    """
    model_ids = list(true_df.columns)
    query_ids = list(true_df.index)
    preds: dict[str, np.ndarray] = {}
    for r in routers:
        mat = r.predict_scores(query_ids).reindex(index=query_ids, columns=model_ids)
        preds[r.name] = mat.to_numpy(float)

    cost = (
        cost_df.reindex(index=query_ids, columns=model_ids).to_numpy(float)
        if cost_df is not None
        else None
    )
    return compare_routing_strategies(
        preds,
        true_df.to_numpy(float),
        cost,
        model_ids,
        lam=lam,
        tolerance=tolerance,
        query_ids=query_ids,
        include_hard_oracle=include_hard_oracle,
        include_random=include_random,
    )
