"""Router-vs-oracle comparison, per query.

Pure numpy / pandas -- no data loading, no torch. ``build_dataset.py`` feeds it
the observed outcome / cost matrices and each router's selections.

The oracle here is the *cost-aware, tolerance-based* one: among the models whose
observed score is within ``tol`` of the query's best score, the cheapest. It is
evaluation-only -- it is derived from observed outcomes and must never feed back
into training, features or candidate selection.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

_EPS = 1e-9
OK, OVERSPEND, QUALITY_LOSS = "ok", "overspend", "quality_loss"


def _require_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(
            f"{name} has non-finite cells; restrict to fully observed queries/models first"
        )


def oracle_pick(
    true: np.ndarray, cost: np.ndarray, model_ids: Sequence[str], *, tol: float
) -> np.ndarray:
    """``[Q]`` column index of the oracle model per query.

    The cheapest model whose score is within ``tol`` of the row's best; exact cost
    ties go to the lexicographically smallest ``model_id`` (independent of column
    order).
    """
    true = np.asarray(true, np.float64)
    cost = np.asarray(cost, np.float64)
    _require_finite("true", true)
    _require_finite("cost", cost)
    within = true >= true.max(axis=1, keepdims=True) - tol - _EPS
    masked_cost = np.where(within, cost, np.inf)
    id_rank = np.argsort(np.argsort(np.asarray(model_ids, dtype=object)))
    order = np.lexsort((np.broadcast_to(id_rank, cost.shape), masked_cost), axis=1)
    return order[:, 0]


def compare_to_oracle(
    true: np.ndarray,
    cost: np.ndarray,
    selected: np.ndarray,
    model_ids: Sequence[str],
    *,
    tol: float = 0.02,
    min_savings: float = 0.25,
) -> pd.DataFrame:
    """One row per query: the router's pick against the oracle's, and how it failed.

    ``failure`` is

    * ``quality_loss`` -- the router's pick scores more than ``tol`` below the
      query's best score (whatever it cost);
    * ``overspend`` -- the pick is within ``tol`` of the best, but the oracle's
      pick is at least ``min_savings`` (a fraction of the router's cost) cheaper;
    * ``ok`` -- neither.

    ``savings_gap`` is ``router_cost - oracle_cost`` (negative when the oracle
    spent more, as in a quality loss the router "saved" its way into);
    ``savings_gap_frac`` is that over the router's cost.
    """
    true = np.asarray(true, np.float64)
    cost = np.asarray(cost, np.float64)
    selected = np.asarray(selected, dtype=np.int64)
    ids = np.asarray(list(model_ids), dtype=object)
    qi = np.arange(len(selected))

    oracle = oracle_pick(true, cost, model_ids, tol=tol)
    best = true.max(axis=1)
    router_score, router_cost = true[qi, selected], cost[qi, selected]
    oracle_score, oracle_cost = true[qi, oracle], cost[qi, oracle]

    savings_gap = router_cost - oracle_cost
    frac = np.divide(savings_gap, router_cost, out=np.zeros_like(savings_gap),
                     where=router_cost > 0)

    quality_loss = router_score < best - tol - _EPS
    overspend = ~quality_loss & (frac >= min_savings) & (savings_gap > 0)
    failure = np.where(quality_loss, QUALITY_LOSS, np.where(overspend, OVERSPEND, OK))

    return pd.DataFrame({
        "best_score": best,
        "oracle_model": ids[oracle],
        "oracle_score": oracle_score,
        "oracle_cost": oracle_cost,
        "router_model": ids[selected],
        "router_score": router_score,
        "router_cost": router_cost,
        "savings_gap": savings_gap,
        "savings_gap_frac": frac,
        "failure": failure,
    })


_ROUTER_COLS = {
    "router_model": "model", "router_score": "score", "router_cost": "cost",
    "savings_gap": "gap", "savings_gap_frac": "gap_frac", "failure": "failure",
}
_ORACLE_COLS = ["best_score", "oracle_model", "oracle_score", "oracle_cost"]


def build_gap_table(
    true_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    selections: dict[str, np.ndarray],
    *,
    tol: float = 0.02,
    min_savings: float = 0.25,
) -> pd.DataFrame:
    """Wide per-query table: the shared oracle columns, then ``<router>_model /
    _score / _cost / _gap / _gap_frac / _failure`` for each router in
    ``selections`` (name -> ``[Q]`` selected column index), and ``failed_by``
    (``"none"``, one router name, or names joined with ``+``) plus ``n_good_models``
    (how many models scored within ``tol`` of the query's best).

    ``true_df`` / ``cost_df`` are dense ``[query_id x model_id]`` frames with the same
    index and columns.
    """
    if not selections:
        raise ValueError("build_gap_table needs at least one router in `selections`")
    model_ids = list(true_df.columns)
    true = true_df.to_numpy(np.float64)
    cost = cost_df.reindex_like(true_df).to_numpy(np.float64)
    per_router = {
        name: compare_to_oracle(true, cost, selected, model_ids, tol=tol, min_savings=min_savings)
        for name, selected in selections.items()
    }
    first = next(iter(per_router.values()))
    table = first[_ORACLE_COLS].copy()
    table["n_good_models"] = (true >= true.max(axis=1, keepdims=True) - tol - _EPS).sum(axis=1)
    for name, cmp in per_router.items():
        table = table.join(cmp[list(_ROUTER_COLS)].rename(columns={
            src: f"{name}_{dst}" for src, dst in _ROUTER_COLS.items()
        }))
    failed = np.column_stack([per_router[n]["failure"].to_numpy() != OK for n in selections])
    names = np.asarray(list(selections), dtype=object)
    table["failed_by"] = ["+".join(names[row]) if row.any() else "none" for row in failed]
    table.index = true_df.index
    table.index.name = "query_id"
    return table


def by_model_table(table: pd.DataFrame, routers: Sequence[str]) -> pd.DataFrame:
    """Long table of ``router, oracle_model, router_model, failure, n,
    total_savings_gap`` -- which model the oracle used against which the router
    paid for, and how often each combination fails."""
    parts = []
    for name in routers:
        g = (
            table.assign(router_model=table[f"{name}_model"], failure=table[f"{name}_failure"],
                         gap=table[f"{name}_gap"])
            .groupby(["oracle_model", "router_model", "failure"], as_index=False)
            .agg(n=("gap", "size"), total_savings_gap=("gap", "sum"))
        )
        parts.append(g.assign(router=name))
    out = pd.concat(parts, ignore_index=True)
    return out[["router", "oracle_model", "router_model", "failure", "n", "total_savings_gap"]]


def failure_rates_by(table: pd.DataFrame, routers: Sequence[str], *, by: str) -> pd.DataFrame:
    """Per router and per value of column ``by`` (e.g. ``dataset``): query count,
    overspend / quality-loss / total failure rate and the mean savings gap."""
    parts = []
    for name in routers:
        fail = table[f"{name}_failure"]
        g = (
            pd.DataFrame({
                by: table[by],
                "overspend": (fail == OVERSPEND).astype(float),
                "quality_loss": (fail == QUALITY_LOSS).astype(float),
                "gap": table[f"{name}_gap"],
            })
            .groupby(by)
            .agg(n=("gap", "size"), overspend_rate=("overspend", "mean"),
                 quality_loss_rate=("quality_loss", "mean"), mean_savings_gap=("gap", "mean"))
            .reset_index()
        )
        g["fail_rate"] = g["overspend_rate"] + g["quality_loss_rate"]
        parts.append(g.assign(router=name))
    out = pd.concat(parts, ignore_index=True)
    return out[["router", by, "n", "overspend_rate", "quality_loss_rate", "fail_rate",
                "mean_savings_gap"]]


def reading_sample(
    table: pd.DataFrame,
    routers: Sequence[str],
    *,
    per_group: int = 3,
    max_prompts: int = 300,
    max_chars: int = 600,
    seed: int = 0,
) -> str:
    """Markdown for reading failed prompts by hand.

    Failures are grouped by ``(router, failure type, oracle model)``; up to
    ``per_group`` prompts are drawn at random from each group (seeded), biggest
    groups first, until ``max_prompts`` have been written. Each prompt shows both
    picks side by side. ``table`` needs ``query`` (the prompt text) and ``dataset``.
    """
    lines = [
        "# Failed prompts, grouped by router | failure | oracle model",
        "",
        "Biggest groups first; each prompt shows what the router paid for against what "
        "the oracle needed.",
        "",
    ]
    written = 0
    for name in routers:
        failed = table[table[f"{name}_failure"] != OK]
        groups = sorted(failed.groupby([f"{name}_failure", "oracle_model"]),
                        key=lambda kv: -len(kv[1]))
        for (failure, oracle_model), grp in groups:
            if written >= max_prompts:
                break
            lines += [f"## {name} | {failure} | oracle={oracle_model} (n={len(grp)})", ""]
            for qid, row in grp.sample(n=min(per_group, len(grp)), random_state=seed).iterrows():
                if written >= max_prompts:
                    break
                text = str(row["query"])
                text = text if len(text) <= max_chars else text[:max_chars] + "..."
                lines += [
                    f"### {qid}  [{row['dataset']}]",
                    f"router: {row[f'{name}_model']} (${row[f'{name}_cost']:.4f}, "
                    f"score {row[f'{name}_score']:.2f})  |  oracle: {oracle_model} "
                    f"(${row['oracle_cost']:.4f}, score {row['oracle_score']:.2f})",
                    "",
                    *[f"> {ln}" for ln in text.splitlines() or [""]],
                    "",
                ]
                written += 1
    return "\n".join(lines)
