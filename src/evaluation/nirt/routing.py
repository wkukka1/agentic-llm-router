"""Routing evaluation: turn a per-(query, model) predictor into a policy and
measure achieved quality / accuracy and token-cost savings.

A policy picks ``m* = argmax_m U(q, m)`` with ``U = pred - lam * C(m)``, ``C(m)``
the per-model mean cost -- the same formula and units
:func:`router.nirt.routing_decision.routing_decision` (the served router's own
decision rule) uses, via :func:`~router.nirt.routing_decision.cost_aware_utility`.
``lam`` is therefore in raw per-query cost units (USD), directly comparable to
``Router.route(lam=...)``'s ``lam`` -- NOT a ``[0, 1]``-normalized scale. ``lam
= 0`` is quality-only routing.

Everything is evaluated on the dense ``(query, model)`` matrices of a split:
``true`` (graded RouterBench ``performance``) and ``cost`` (USD per query, the
only cost signal RouterBench carries -- no token counts).

    from training.data.facade import load_training_data
    from router.nirt.routing import eval_matrices, routing_report

    d = load_training_data(load_config())
    true_df, cost_df = eval_matrices(d, split="test")
    report = routing_report({"nirt": nirt_pred, "classical_irt": irt_pred},
                            true_df.to_numpy(), cost_df.to_numpy(),
                            model_ids=list(true_df.columns), train_quality=train_q)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from router.nirt.routing_decision import routing_decision


# --------------------------------------------------------------------------- #
# matrices                                                                    #
# --------------------------------------------------------------------------- #
def dense_matrices(obs: pd.DataFrame):
    """``(true, cost)`` DataFrames [query x model] from a NIRT observation frame,
    keeping only queries with every model's target *and* cost observed."""
    from router.nirt.frames import pivot_qm

    true = pivot_qm(obs, "target")
    cost = pivot_qm(obs, "cost").reindex(index=true.index, columns=true.columns)
    keep = true.notna().all(axis=1) & cost.notna().all(axis=1)
    return true.loc[keep], cost.loc[keep]


def eval_matrices(data, split: str = "test", models: Optional[list[str]] = None):
    """Return aligned ``(true, cost)`` DataFrames [query x model] for ``split``."""
    obs = data.nirt_observations()
    obs = obs[obs["split"] == split]
    if models is not None:
        obs = obs[obs["model_id"].isin(models)]
    return dense_matrices(obs)


def align(matrix: pd.DataFrame, like: pd.DataFrame) -> np.ndarray:
    """Reindex ``matrix`` to ``like``'s (index, columns) and return a numpy array."""
    return matrix.reindex(index=like.index, columns=like.columns).to_numpy(np.float64)


def train_quality(data, models: list[str], metric: str = "quality") -> dict[str, float]:
    """Per-model mean training correctness -- for the 'best fixed model' baseline.

    ``metric="quality"`` -> mean graded target; ``"accuracy"`` -> mean(target>=0.5).
    """
    obs = data.nirt_observations()
    obs = obs[(obs["split"] == "train") & (obs["model_id"].isin(models))]
    y = obs["target"].to_numpy(np.float64)
    if metric == "accuracy":
        y = (y >= 0.5).astype(np.float64)
    return pd.Series(y).groupby(obs["model_id"].to_numpy()).mean().to_dict()


# --------------------------------------------------------------------------- #
# policies                                                                    #
# --------------------------------------------------------------------------- #
def _policy_row(name: str, choice: np.ndarray, true: np.ndarray, cost: np.ndarray) -> dict:
    qi = np.arange(len(choice))
    q = true[qi, choice]
    c = cost[qi, choice]
    return {
        "policy": name,
        "quality": float(q.mean()),
        "accuracy": float((q >= 0.5).mean()),
        "cost": float(c.mean()),
        "cost_per_1k_queries": float(c.mean() * 1000),
    }


def route(pred: np.ndarray, cost: np.ndarray, lam: float = 0.0) -> np.ndarray:
    """Model index chosen per query for ``U = pred - lam * C(m)``.

    ``C(m)`` is the per-model mean cost (not the per-cell realized cost -- a
    live router could not know a query's exact serving cost in advance), and
    non-finite predictions/costs are masked -- both via
    :func:`router.nirt.routing_decision.routing_decision`, the served
    router's own decision rule, so an offline ``lam`` sweep here selects
    exactly what ``Router.route(lam=lam)`` would."""
    C = np.asarray(cost, np.float64).mean(axis=0)
    return routing_decision(pred, lam=lam, model_costs=C)


def oracle_choice(
    true: np.ndarray, cost: np.ndarray, *, tol: float = 1e-9,
    model_ids: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Per-query index of the oracle model: the max observed score, ties broken by cost.

    Cost is **only a tie-breaker** among the models attaining the max score -- it is
    never traded off against score (that is the cost-aware utility, reported
    separately). Ties left after cost (equal score *and* equal cost) are broken by
    one of two modes:

    * ``model_ids`` given -- the **canonical** rule: max score -> min cost ->
      lexicographically smallest ``model_id``. Independent of column order; new
      code should always pass ``model_ids``.
    * ``model_ids=None`` -- **legacy compatibility**: max score -> min cost -> first
      column. Kept so ``routing_report`` and the pool-expansion battery reproduce
      their historical numbers.

    A plain ``true.argmax(1)`` is also "best per query", but it breaks ties by
    column order, so on the many queries where several models are equally good it
    overpays -- inflating the oracle cost (and deflating the oracle's cost-saving
    ceiling). Breaking the tie by cost matches RouterBench's own oracle
    (cheapest correct model) and does not change the oracle's quality.
    """
    true = np.asarray(true, np.float64)
    at_max = true >= true.max(axis=1, keepdims=True) - tol
    masked_cost = np.where(at_max, np.asarray(cost, np.float64), np.inf)
    if model_ids is None:
        return masked_cost.argmin(axis=1)
    ids = np.asarray([str(m) for m in model_ids])
    if len(ids) != true.shape[1]:
        raise ValueError(f"model_ids has {len(ids)} entries for {true.shape[1]} columns")
    id_rank = np.argsort(np.argsort(ids, kind="stable"), kind="stable")
    id_key = np.broadcast_to(id_rank, true.shape)
    return np.lexsort((id_key, masked_cost), axis=-1)[:, 0]


def routing_report(
    preds: dict[str, np.ndarray],
    true: np.ndarray,
    cost: np.ndarray,
    *,
    model_ids: list[str],
    train_quality: Optional[dict[str, float]] = None,
    reference: str = "gpt-4-1106-preview",
    lam: float = 0.0,
) -> pd.DataFrame:
    """One row per policy: oracle, fixed-model baselines, and each predictor.

    Adds savings columns relative to ``reference`` (a strong fixed model) and the
    quality gap to the oracle. The oracle is the *cheapest* model that attains
    each query's max true score (:func:`oracle_choice`) -- same quality as
    ``true.argmax(1)`` but no column-order tie-break overpay.
    """
    n_q, n_m = true.shape
    rows: list[dict] = []

    # -- reference / bound policies --------------------------------------
    rows.append(_policy_row("oracle (best per query)", oracle_choice(true, cost), true, cost))
    if train_quality:
        best_fixed = max(train_quality, key=train_quality.get)
        j = model_ids.index(best_fixed)
        rows.append(_policy_row(f"fixed: {best_fixed} (best on train)",
                                np.full(n_q, j), true, cost))
    cheapest = int(np.argmin(cost.mean(0)))
    rows.append(_policy_row(f"fixed: {model_ids[cheapest]} (cheapest)",
                            np.full(n_q, cheapest), true, cost))
    if reference in model_ids:
        rows.append(_policy_row(f"fixed: {reference}",
                                np.full(n_q, model_ids.index(reference)), true, cost))
    rows.append({"policy": "random model", "quality": float(true.mean()),
                 "accuracy": float((true >= 0.5).mean()),
                 "cost": float(cost.mean()),
                 "cost_per_1k_queries": float(cost.mean() * 1000)})

    # -- learned policies ----------------------------------------------
    for name, pred in preds.items():
        rows.append(_policy_row(f"route: {name} (lam={lam})",
                                route(pred, cost, lam), true, cost))

    df = pd.DataFrame(rows)

    ref_row = df[df["policy"] == f"fixed: {reference}"]
    oracle_q = df.iloc[0]["quality"]
    if not ref_row.empty:
        rc, ra = float(ref_row["cost"].iloc[0]), float(ref_row["accuracy"].iloc[0])
        df["cost_savings_vs_ref"] = 1.0 - df["cost"] / rc
        df["d_accuracy_vs_ref"] = df["accuracy"] - ra
    df["d_quality_vs_oracle"] = df["quality"] - oracle_q
    return df


def pareto(
    pred: np.ndarray,
    true: np.ndarray,
    cost: np.ndarray,
    lams=(0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
) -> pd.DataFrame:
    """Quality / accuracy / cost for a predictor across a sweep of ``lam``."""
    out = []
    for lam in lams:
        r = _policy_row(f"lam={lam}", route(pred, cost, lam), true, cost)
        r["lam"] = lam
        out.append(r)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Reward(alpha) and AIQ  -- scalars comparable to IRT-Router / RouterBench     #
# --------------------------------------------------------------------------- #
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def linear_cost(value, pool_mean_costs) -> np.ndarray:
    """Map a cost onto ``[0, 1]`` by the candidate pool's per-model mean-cost range."""
    c = np.asarray(pool_mean_costs, dtype=np.float64)
    lo, hi = float(c.min()), float(c.max())
    v = np.asarray(value, dtype=np.float64)
    return np.zeros_like(v) if hi <= lo else np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def reward(quality, policy_mean_cost, alpha: float, *, pool_mean_costs) -> float:
    """IRT-Router scoring: ``alpha * quality - (1 - alpha) * linear(cost)``."""
    return float(alpha * np.asarray(quality)
                 - (1 - alpha) * linear_cost(policy_mean_cost, pool_mean_costs))


def add_reward_columns(report_df: pd.DataFrame, pool_mean_costs, alphas=(0.5, 0.7, 0.8, 0.9)) -> pd.DataFrame:
    df = report_df.copy()
    for a in alphas:
        df[f"reward@{a}"] = [
            reward(q, c, a, pool_mean_costs=pool_mean_costs)
            for q, c in zip(df["quality"], df["cost"])
        ]
    return df


def aiq(pareto_df: pd.DataFrame, *, pool_mean_costs, pool_quality) -> dict:
    """Area-under-the-cost/quality-curve, in the spirit of RouterBench's AIQ.

    * router frontier ``q_r(c)`` -- the pareto points, sorted by cost, with a
      running-max quality (you can always spend more and do no worse).
    * baseline line -- linear interpolation between the cheapest model
      ``(min cost, its quality)`` and the best-quality model ``(its cost, its quality)``.
    * ``aiq_absolute``   = mean of ``q_r`` over the pool's cost range (trapezoid / range).
    * ``aiq_improvement``= mean of ``q_r - q_baseline`` over that range (>0 = the router
      beats naive "pay more, linearly get more").
    """
    c = np.asarray(pool_mean_costs, dtype=np.float64)
    q_pool = np.asarray(pool_quality, dtype=np.float64)
    lo, hi = float(c.min()), float(c.max())
    if hi <= lo:
        return {"aiq_absolute": float(q_pool.mean()), "aiq_improvement": 0.0}

    pts = pareto_df[["cost", "quality"]].to_numpy(np.float64)
    pts = pts[np.argsort(pts[:, 0])]
    xs = np.clip(pts[:, 0], lo, hi)
    ys = np.maximum.accumulate(pts[:, 1])
    grid = np.unique(np.concatenate([[lo, hi], xs]))
    q_r = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])

    best = int(np.argmax(q_pool))
    cheap = int(np.argmin(c))
    q_base = np.interp(grid, [c[cheap], c[best]], [q_pool[cheap], q_pool[best]],
                       left=q_pool[cheap], right=q_pool[best])

    span = hi - lo
    return {
        "aiq_absolute": float(_trapz(q_r, grid) / span),
        "aiq_improvement": float(_trapz(q_r - q_base, grid) / span),
        "cost_range": [lo, hi],
    }
