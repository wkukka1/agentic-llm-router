"""The fixed pool-expansion evaluation battery (§1 of the workstream plan).

``run_battery(phase, ...)`` re-runs, identically, for one phase:

* **1a** pool description            -> ``pool``
* **1b** prediction quality          -> ``prediction``   (ZOIB + Bernoulli control)
* **1c** routing policy table + frontier + AIQ
* **1d** derived headline numbers    -> ``routing["derived"]``
* **1e** cold-start status           -> ``cold_start``
* **1f** ablation (pool w/o the newly added models) -> ``ablation``

plus a **provenance** block. Everything is written under
``artifacts/pool_expansion/<phase>/`` and one row is appended to
``artifacts/pool_expansion/ledger.json`` (rendered to
``docs/pool_expansion_results.md``).

The predictor is frozen -- this module only ever *reads* checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from router.config import Config, load_config
from router.provenance import file_digest, git_dirty, git_sha
from training.data.facade import load_training_data
from training.data.splits import SPLIT_FILES
from training.nirt.baseline.checkpoint import load_run
from training.nirt.baseline.continuous_eval import evaluate_continuous
from training.nirt.baseline.data import checkpoint_matrix as _pred_matrix
from training.nirt.baseline.eval import evaluate_checkpoint

from ..nirt.routing import (
    add_reward_columns,
    aiq,
    align,
    eval_matrices,
    oracle_choice,
    pareto,
    route,
    routing_report,
    train_quality,
)

REFERENCE = "gpt-4-1106-preview"

LEDGER_COLUMNS = [
    "phase",
    "n_warm_models",
    "sources",
    "gpt4_cheapest_cost_ratio",
    "zoib_test_nll",
    "zoib_test_mae",
    "zoib_mean_cal_ece",
    "oracle_cost_saving_ceiling",
    "zoib_saving_at_minus1pt",
    "zoib_saving_at_minus3pt",
    "zoib_saving_at_minus1pt_test_ceiling",
    "zoib_saving_at_minus3pt_test_ceiling",
    "zoib_aiq_improvement",
    "oracle_offbest_fraction",
    "coldstart_beats_global_mean",
    "notes",
]


# --------------------------------------------------------------------------- #
# prediction matrices (frozen checkpoints)  --  _pred_matrix is                 #
# router.nirt.baseline.data.checkpoint_matrix (imported above)                  #
# --------------------------------------------------------------------------- #
def _nirt_matrix(run_name: str, data, split: str):
    from router.nirt.predict import predict_matrix
    from router.nirt.checkpoint import load_run as nirt_load_run

    model, _, midx = nirt_load_run(run_name)
    return predict_matrix(model, midx, data, split, pathway="irt")


def prediction_matrices(
    d, cfg: Config, split: str, *, zoib: str, bernoulli: str, nirt_run: Optional[str],
    lcb_level: float = 0.8,
) -> tuple[dict[str, pd.DataFrame], Optional[str]]:
    """Unaligned ``[query x model]`` prediction frames for every routed policy, plus
    the NIRT run's error (``None`` if it scored or wasn't requested).

    Computed once per battery: the frames don't depend on the pool, so
    ``routing_block`` and every ablation just ``align`` them to their matrices.
    """
    preds = {
        "baseline NIRT (Bernoulli)": _pred_matrix(bernoulli, cfg, split, "proba"),
        "ZOIB  E[Y]": _pred_matrix(zoib, cfg, split, "mean"),
        f"ZOIB  LCB(level={lcb_level})": _pred_matrix(zoib, cfg, split, "lower", level=lcb_level),
    }
    nirt_error = None
    if nirt_run:
        try:
            preds[f"NIRT ({nirt_run})"] = _nirt_matrix(nirt_run, d, split)
        except Exception as exc:  # noqa: BLE001 - NIRT run is optional context; recorded, not swallowed
            nirt_error = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                f"NIRT run {nirt_run!r} failed on {split!r} and is omitted from the "
                f"policy table: {nirt_error}",
                stacklevel=2,
            )
    return preds, nirt_error


def _pool_matrices(d, split: str, pool_models: Optional[Sequence[str]] = None, keep_query_pred=None):
    """``eval_matrices`` for the pool, failing fast when the pool can't be evaluated
    on ``split`` (a pinned model with no observations would otherwise empty the
    dense matrices and turn every downstream number into NaN)."""
    models = list(pool_models) if pool_models else None
    obs = d.nirt_observations()
    obs_split = obs[obs["split"] == split]
    if models:
        present = set(obs_split["model_id"])
        missing = [m for m in models if m not in present]
        if missing:
            raise ValueError(f"pool models with no {split!r} observations: {missing}")
    true_df, cost_df = eval_matrices(d, split=split, models=models)
    if keep_query_pred is not None:
        mask = [bool(keep_query_pred(q)) for q in true_df.index]
        true_df, cost_df = true_df.loc[mask], cost_df.loc[mask]
    if true_df.empty:
        sub = obs_split[obs_split["model_id"].isin(models)] if models else obs_split
        sparsest = sub.groupby("model_id")["query_id"].nunique().nsmallest(3).to_dict()
        raise ValueError(
            f"no {split!r} query has a target and cost for every pool model "
            f"(sparsest models, by #queries: {sparsest})"
        )
    return true_df, cost_df


# --------------------------------------------------------------------------- #
# 1a -- pool description                                                      #
# --------------------------------------------------------------------------- #
def pool_description(d, split: str, *, pool_models: Optional[Sequence[str]] = None) -> dict:
    obs = d.nirt_observations()
    true_df, cost_df = _pool_matrices(d, split, pool_models)
    models = list(true_df.columns)

    n_obs_all = obs.groupby("model_id").size().to_dict()
    src = (
        obs.groupby("model_id")["source"].agg(lambda s: sorted(set(s))).to_dict()
    )
    warm = set(d.warm_model_ids())

    rows = []
    for m in models:
        acc = float((true_df[m].to_numpy() >= 0.5).mean())
        cost_1k = float(cost_df[m].to_numpy().mean() * 1000)
        rows.append({
            "model_id": m,
            "sources": src.get(m, []),
            "n_correctness_obs": int(n_obs_all.get(m, 0)),
            "test_accuracy": acc,
            "mean_cost_per_1k": cost_1k,
            "warm": m in warm,
        })
    rows.sort(key=lambda r: r["mean_cost_per_1k"])

    costs = np.array([r["mean_cost_per_1k"] for r in rows])
    accs = np.array([r["test_accuracy"] for r in rows])
    max_cost = costs.max()
    cheap_mask = costs <= max_cost / 30.0
    return {
        "split": split,
        "models": rows,
        "n_models": len(rows),
        "n_warm_models": int(sum(r["warm"] for r in rows)),
        "gpt4_cheapest_cost_ratio": float(max_cost / costs.min()) if costs.min() > 0 else None,
        "n_priced_at_or_below_1_30th": int(cheap_mask.sum()),
        "mean_accuracy_of_cheap_models": float(accs[cheap_mask].mean()) if cheap_mask.any() else None,
        "best_single_test_accuracy": float(accs.max()),
    }


def shot_breakdown(d, split: str, *, pool_models: Sequence[str], suffix: str = ":5shot") -> Optional[dict]:
    """Per-model test accuracy on the base subset vs the ``suffix`` subset
    (E2: 0-shot vs 5-shot). ``None`` if the split has no ``suffix`` queries."""
    true_df, cost_df = _pool_matrices(d, split, pool_models)
    is_suf = np.array([str(q).endswith(suffix) for q in true_df.index])
    if not is_suf.any():
        return None
    base, suf = true_df.loc[~is_suf], true_df.loc[is_suf]
    cbase, csuf = cost_df.loc[~is_suf], cost_df.loc[is_suf]
    rows = []
    for m in true_df.columns:
        a0 = float((base[m].to_numpy() >= 0.5).mean())
        a5 = float((suf[m].to_numpy() >= 0.5).mean())
        rows.append({
            "model_id": m,
            "acc_base": a0, "acc_suffix": a5, "d_acc": a5 - a0,
            "cost_per_1k_base": float(cbase[m].mean() * 1000),
            "cost_per_1k_suffix": float(csuf[m].mean() * 1000),
        })
    rows.sort(key=lambda r: r["cost_per_1k_base"])
    return {
        "suffix": suffix,
        "n_base_queries": int((~is_suf).sum()),
        "n_suffix_queries": int(is_suf.sum()),
        "per_model": rows,
        "mean_d_acc": float(np.mean([r["d_acc"] for r in rows])),
        "note": "acc_base = 0-shot subset, acc_suffix = 5-shot subset, same test queries.",
    }


# --------------------------------------------------------------------------- #
# 1b -- prediction quality                                                    #
# --------------------------------------------------------------------------- #
def prediction_quality(zoib: str, bernoulli: str, cfg: Config, split: str) -> dict:
    z = evaluate_continuous(zoib, phase0_cfg=cfg, split=split, write=False)
    b = evaluate_checkpoint(bernoulli, phase0_cfg=cfg, split=split, write=False)
    zm = z["metrics"]
    return {
        "zoib": {
            "nll_mean": zm["nll_mean"],
            "nll_median": zm["nll_median"],
            "mae": zm["mae"],
            "rmse": zm["rmse"],
            "mean_calibration_ece": zm["mean_calibration_ece"],
            "coverage_50": zm.get("coverage_50"),
            "coverage_90": zm.get("coverage_90"),
            "bce_diag_bce": zm["bce_diag_bce"],
            "bce_diag_accuracy": zm["bce_diag_accuracy"],
            "boundary_calibration": z["boundary_calibration"],
            "theta_effective_rank": z["theta_spectrum"]["effective_rank"],
            "theta_dim": z["theta_spectrum"]["shape"][1],
            "per_family": z["per_family"],
            "response_flags": z["parameter_summary"]["response_flags"],
        },
        "bernoulli": {
            "bce": b["prediction"]["bce"],
            "accuracy": b["prediction"]["accuracy"],
            "brier": b["prediction"]["brier"],
            "auc": b["prediction"]["auc"],
            "ece": b["calibration"]["ece"],
            "delta_bce_vs_model_mean": b["prediction"]["delta_bce_vs_model_mean"],
            "theta_effective_rank": b["theta_spectrum"]["effective_rank"],
            "per_family": b["per_family"],
            "pathology_flags": b["parameter_summary"]["pathology_flags"],
        },
    }


# --------------------------------------------------------------------------- #
# 1c + 1d -- routing                                                          #
# --------------------------------------------------------------------------- #
# Dense enough that the matched-accuracy band optimum (§1d) is not an artifact of
# where a coarse grid happens to fall relative to the -1/-3 pt cutoff.
_FRONTIER_LAMS = (
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0,
)


def _frontier_saving(frontier: pd.DataFrame, ref_cost_1k: float, best_acc: float, margin: float):
    """Largest cost-saving on the frontier whose accuracy >= best_acc - margin."""
    q = frontier[frontier["accuracy"] >= best_acc - margin]
    if q.empty:
        return None
    r = q.loc[q["cost_per_1k_queries"].idxmin()]
    return {
        "lam": float(r["lam"]),
        "accuracy": float(r["accuracy"]),
        "d_accuracy_vs_best_single": float(r["accuracy"] - best_acc),
        "cost_per_1k_queries": float(r["cost_per_1k_queries"]),
        "cost_saving_vs_ref": float(1.0 - r["cost_per_1k_queries"] / ref_cost_1k),
    }


def _val_selected_saving(
    val_frontier: pd.DataFrame, test_frontier: pd.DataFrame, ref_cost_1k: float,
    *, val_best_acc: float, test_best_acc: float, margin: float,
):
    """Operating point chosen on validation, measured on test.

    Picks the cheapest *validation* frontier point whose accuracy is within
    ``margin`` of the best single model's validation accuracy, then reports the
    *test* frontier row at that lambda -- a choice a deployed router could have made
    in advance. :func:`_frontier_saving` on the test frontier is the ceiling.
    """
    q = val_frontier[val_frontier["accuracy"] >= val_best_acc - margin]
    if q.empty:
        return None
    v = q.loc[q["cost_per_1k_queries"].idxmin()]
    lam = float(v["lam"])
    t = test_frontier[np.isclose(test_frontier["lam"].to_numpy(np.float64), lam)]
    if t.empty:
        raise ValueError(f"lam={lam} chosen on validation is not on the test frontier grid")
    t = t.iloc[0]
    return {
        "lam": lam,
        "val_accuracy": float(v["accuracy"]),
        "accuracy": float(t["accuracy"]),
        "d_accuracy_vs_best_single": float(t["accuracy"] - test_best_acc),
        "cost_per_1k_queries": float(t["cost_per_1k_queries"]),
        "cost_saving_vs_ref": float(1.0 - t["cost_per_1k_queries"] / ref_cost_1k),
    }


def _closest_lam(frontier: pd.DataFrame, target_acc: float) -> float:
    return float(frontier.loc[(frontier["accuracy"] - target_acc).abs().idxmin(), "lam"])


def derived_headline(
    true: np.ndarray, cost: np.ndarray, zoib_pred: np.ndarray,
    *, model_ids: list[str], tq_acc: dict[str, float], frontier: pd.DataFrame,
    val_frontier: Optional[pd.DataFrame] = None, val_best_acc: Optional[float] = None,
    reference: str = REFERENCE,
) -> dict:
    """§1d: oracle ceiling, matched-accuracy savings, oracle mix, escalation.

    Matched-accuracy savings (``router_saving_at_minus*pt``) and the -3 pt lambda
    are chosen on ``val_frontier`` and measured on test; ``None`` when no validation
    frontier is given. The test-chosen values are kept as ``*_test_ceiling``.

    The oracle is the *cheapest* model that attains each query's max true score
    (:func:`router.nirt.routing.oracle_choice`) -- same quality as
    ``true.argmax(1)`` but no column-order tie-break overpay. The naive-argmax
    oracle's cost is still recorded (``naive_argmax_oracle``) so the difference
    the tie-break makes stays visible.
    """
    n = true.shape[0]
    qi = np.arange(n)
    o_choice = oracle_choice(true, cost)
    oracle_cost_1k = float(cost[qi, o_choice].mean() * 1000)

    best_single = max(tq_acc, key=tq_acc.get)
    jb = model_ids.index(best_single)
    best_cost_1k = float(cost[:, jb].mean() * 1000)
    best_acc = float((true[:, jb] >= 0.5).mean())
    ref_cost_1k = (
        float(cost[:, model_ids.index(reference)].mean() * 1000)
        if reference in model_ids else best_cost_1k
    )

    ceiling = 1.0 - oracle_cost_1k / best_cost_1k if best_cost_1k > 0 else 0.0
    mix = {
        model_ids[k]: int((o_choice == k).sum())
        for k in range(len(model_ids)) if (o_choice == k).any()
    }

    # reference: the naive column-order-tie-break oracle (what route_compare used
    # before the fix). Overpays on ties; recorded so the delta stays visible.
    na_choice = true.argmax(1)
    na_cost_1k = float(cost[qi, na_choice].mean() * 1000)
    naive_argmax_oracle = {
        "cost_per_1k_queries": na_cost_1k,
        "cost_saving_ceiling": 1.0 - na_cost_1k / best_cost_1k if best_cost_1k > 0 else 0.0,
        "offbest_fraction": float((na_choice != jb).mean()),
        "model_mix": {
            model_ids[k]: int((na_choice == k).sum())
            for k in range(len(model_ids)) if (na_choice == k).any()
        },
    }

    lam0_choice = route(zoib_pred, cost, 0.0)
    have_val = val_frontier is not None and val_best_acc is not None and not val_frontier.empty
    # lambda that lands closest to the -3pt operating point: validation-chosen
    # (deployable) and test-chosen (ceiling)
    lam3 = _closest_lam(val_frontier, val_best_acc - 0.03) if have_val else None
    lam3_ceiling = _closest_lam(frontier, best_acc - 0.03)

    def _saving(margin):
        if not have_val:
            return None
        return _val_selected_saving(
            val_frontier, frontier, ref_cost_1k,
            val_best_acc=val_best_acc, test_best_acc=best_acc, margin=margin,
        )

    def _escalation(lam):
        return None if lam is None else float((route(zoib_pred, cost, lam) != jb).mean())

    return {
        "best_single_model": best_single,
        "best_single_test_accuracy": best_acc,
        "reference_model": reference,
        "oracle_cost_per_1k_queries": oracle_cost_1k,
        "best_single_cost_per_1k_queries": best_cost_1k,
        "reference_cost_per_1k_queries": ref_cost_1k,
        "oracle_cost_saving_ceiling": ceiling,
        "oracle_offbest_fraction": float((o_choice != jb).mean()),
        "oracle_model_mix": mix,
        "oracle_note": "oracle = cheapest model attaining each query's max true score "
                       "(ties broken by cost, matching RouterBench). naive_argmax_oracle "
                       "is the old column-order tie-break for comparison.",
        "naive_argmax_oracle": naive_argmax_oracle,
        "lambda_selection": "validation" if have_val else "unavailable",
        "router_saving_at_minus1pt": _saving(0.01),
        "router_saving_at_minus3pt": _saving(0.03),
        # historical Phase 2 report operating point (ZOIB -57% @ -3.7pt, lam 0.5)
        "router_saving_at_minus3_7pt": _saving(0.037),
        "router_saving_at_minus1pt_test_ceiling": _frontier_saving(frontier, ref_cost_1k, best_acc, 0.01),
        "router_saving_at_minus3pt_test_ceiling": _frontier_saving(frontier, ref_cost_1k, best_acc, 0.03),
        "router_saving_at_minus3_7pt_test_ceiling": _frontier_saving(frontier, ref_cost_1k, best_acc, 0.037),
        "zoib_escalation_rate_lam0": float((lam0_choice != jb).mean()),
        "zoib_escalation_rate_at_minus3pt": _escalation(lam3),
        "zoib_escalation_rate_at_minus3pt_test_ceiling": _escalation(lam3_ceiling),
        "minus3pt_lambda": lam3,
        "minus3pt_lambda_test_ceiling": lam3_ceiling,
    }


_VALIDATION = "validation"


def _validation_frontier(d, val_zoib_ey, model_ids: list[str], tq_acc: dict[str, float], keep_query_pred):
    """``(frontier, best_acc, reason)`` on the validation split for the same pool.

    ``best_acc`` is the validation accuracy of the train-selected best single model.
    These pick lambda without test labels; ``(None, None, reason)`` when the pool
    can't be evaluated on validation.
    """
    if val_zoib_ey is None:
        return None, None, "no validation ZOIB predictions supplied"
    try:
        vt_df, vc_df = _pool_matrices(d, _VALIDATION, model_ids, keep_query_pred)
    except ValueError as exc:
        return None, None, str(exc)
    vt_df, vc_df = vt_df[model_ids], vc_df[model_ids]
    vt, vc = vt_df.to_numpy(np.float64), vc_df.to_numpy(np.float64)
    fr = pareto(align(val_zoib_ey, vt_df), vt, vc, lams=_FRONTIER_LAMS)
    jb = model_ids.index(max(tq_acc, key=tq_acc.get))
    return fr, float((vt[:, jb] >= 0.5).mean()), None


def routing_block(
    d, split: str, *, preds: dict[str, pd.DataFrame], val_zoib_ey: Optional[pd.DataFrame] = None,
    nirt_run: Optional[str] = None, nirt_error: Optional[str] = None,
    pool_models: Optional[Sequence[str]] = None, reference: str = REFERENCE,
    keep_query_pred=None,
) -> dict:
    """1c + 1d for one pool. ``preds`` are :func:`prediction_matrices` frames on
    ``split`` (must include ``"ZOIB  E[Y]"``); ``val_zoib_ey`` is ZOIB E[Y] on the
    validation split, used only to choose the headline lambda."""
    true_df, cost_df = _pool_matrices(d, split, pool_models, keep_query_pred)
    model_ids = list(true_df.columns)
    true, cost = true_df.to_numpy(np.float64), cost_df.to_numpy(np.float64)
    tq_acc = train_quality(d, model_ids, metric="accuracy")

    aligned = {name: align(m, true_df) for name, m in preds.items()}
    rep = routing_report(
        aligned, true, cost, model_ids=model_ids, train_quality=tq_acc, reference=reference, lam=0.0,
    )
    rep = add_reward_columns(rep, pool_mean_costs=cost.mean(axis=0))

    zoib_ey = aligned["ZOIB  E[Y]"]
    fr = pareto(zoib_ey, true, cost, lams=_FRONTIER_LAMS)
    frontier_aiq = aiq(
        fr, pool_mean_costs=cost.mean(axis=0),
        pool_quality=[true_df[m].mean() for m in model_ids],
    )
    val_fr, val_best_acc, val_reason = _validation_frontier(
        d, val_zoib_ey, model_ids, tq_acc, keep_query_pred
    )
    derived = derived_headline(
        true, cost, zoib_ey, model_ids=model_ids, tq_acc=tq_acc,
        frontier=fr, val_frontier=val_fr, val_best_acc=val_best_acc, reference=reference,
    )
    if val_reason:
        derived["lambda_selection_reason"] = val_reason
    return {
        "split": split,
        "n_queries": int(true.shape[0]),
        "model_ids": model_ids,
        "nirt_run": nirt_run,
        "nirt_error": nirt_error,
        "policies": rep.to_dict(orient="records"),
        "zoib_frontier": fr.to_dict(orient="records"),
        "zoib_frontier_validation": val_fr.to_dict(orient="records") if val_fr is not None else None,
        "zoib_aiq": frontier_aiq,
        "derived": derived,
    }


# --------------------------------------------------------------------------- #
# 1e -- cold-start                                                            #
# --------------------------------------------------------------------------- #
def cold_start_block(zoib_projected: str, cfg: Config, split: str) -> dict:
    p = Path(zoib_projected)
    if not (p / "model.pt").exists():
        return {"status": "skipped", "reason": f"no checkpoint at {zoib_projected}"}
    r = evaluate_continuous(p, phase0_cfg=cfg, split=split, write=False)
    cs = r.get("cold_start")
    if not cs or "pooled" not in cs:
        return {"status": "skipped", "reason": (cs or {}).get("skipped", "no cold-start observations")}
    pooled = cs["pooled"]
    nirt_bce = pooled["baseline_nirt"].get("bce")
    gm_bce = pooled["global_mean"].get("bce")
    return {
        "status": "ran",
        "cold_models": cs["cold_models"],
        "projected_theta_effective_rank": r["theta_spectrum"]["effective_rank"],
        "pooled": pooled,
        "per_model": cs["per_model"],
        "beats_global_mean_on_bce": bool(nirt_bce is not None and gm_bce is not None and nirt_bce < gm_bce),
    }


# --------------------------------------------------------------------------- #
# 1f -- ablation                                                              #
# --------------------------------------------------------------------------- #
def ablation_pools(added_models: Sequence[str], full_pool: Sequence[str]) -> tuple[list[str], list[str]]:
    """(models actually removed, reduced pool) -- removes *exactly* the named
    models that are in the pool, and nothing else, order preserved."""
    added = [m for m in added_models if m in full_pool]
    reduced = [m for m in full_pool if m not in set(added)]
    return added, reduced


def _ablation_case(d, split, *, remove, full_pool, full_derived, routing_kw) -> dict:
    _, reduced = ablation_pools(remove, full_pool)
    rd = routing_block(d, split, pool_models=reduced, **routing_kw)["derived"]
    return {
        "removed_models": list(remove),
        "reduced_pool": reduced,
        "reduced_oracle_cost_saving_ceiling": rd["oracle_cost_saving_ceiling"],
        "delta_oracle_cost_saving_ceiling": (
            full_derived["oracle_cost_saving_ceiling"] - rd["oracle_cost_saving_ceiling"]
        ),
        "delta_zoib_saving_at_minus3pt": _delta_saving(
            full_derived.get("router_saving_at_minus3pt"), rd.get("router_saving_at_minus3pt")
        ),
        "reduced_derived": rd,
    }


def ablation_block(
    d, split: str, *, added_models: Sequence[str], full_pool: Sequence[str],
    full_derived: dict, preds: dict[str, pd.DataFrame],
    val_zoib_ey: Optional[pd.DataFrame] = None, reference: str = REFERENCE,
    drop_query_suffix: Optional[str] = None,
) -> dict:
    # The prediction frames are pool-independent, so every re-run reuses them.
    routing_kw = dict(preds=preds, val_zoib_ey=val_zoib_ey, reference=reference)
    # Query-subset ablation (E2): re-run routing with a query subset removed
    # (e.g. all ":5shot" items) -- the "added" thing that phase is observations,
    # not models.
    if drop_query_suffix:
        rd = routing_block(
            d, split, pool_models=full_pool,
            keep_query_pred=lambda q, s=drop_query_suffix: not str(q).endswith(s),
            **routing_kw,
        )["derived"]
        return {
            "status": "ran",
            "mode": "query_subset",
            "removed_query_suffix": drop_query_suffix,
            "reduced_derived": rd,
            "reduced_oracle_cost_saving_ceiling": rd["oracle_cost_saving_ceiling"],
            "delta_oracle_cost_saving_ceiling":
                full_derived["oracle_cost_saving_ceiling"] - rd["oracle_cost_saving_ceiling"],
            "delta_zoib_saving_at_minus3pt": _delta_saving(
                full_derived.get("router_saving_at_minus3pt"), rd.get("router_saving_at_minus3pt")
            ),
        }

    added, _ = ablation_pools(added_models, full_pool)
    if not added:
        return {"status": "n/a", "reason": "no models added this phase"}

    kw = dict(full_pool=full_pool, full_derived=full_derived, routing_kw=routing_kw)
    out = {
        "status": "ran",
        "added_models": added,
        # all newly added models removed at once
        "all_removed": _ablation_case(d, split, remove=added, **kw),
    }
    # per-model leave-one-out attribution (only meaningful with >1 added model)
    if len(added) > 1:
        out["leave_one_out"] = {
            m: _ablation_case(d, split, remove=[m], **kw) for m in added
        }
    # convenience mirrors of the headline deltas (all-removed case)
    out["delta_oracle_cost_saving_ceiling"] = out["all_removed"]["delta_oracle_cost_saving_ceiling"]
    out["delta_zoib_saving_at_minus3pt"] = out["all_removed"]["delta_zoib_saving_at_minus3pt"]
    return out


def _delta_saving(full, reduced):
    fv = full["cost_saving_vs_ref"] if full else None
    rv = reduced["cost_saving_vs_ref"] if reduced else None
    if fv is None or rv is None:
        return {"full": fv, "reduced": rv, "delta": None}
    return {"full": fv, "reduced": rv, "delta": fv - rv}


# --------------------------------------------------------------------------- #
# provenance                                                                  #
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> Optional[str]:
    return file_digest(path)


def provenance(cfg: Config, *, phase: str, pool_models: list[str], checkpoints: dict) -> dict:
    root = cfg.root
    sha, dirty = git_sha(root), git_dirty(root)

    artefacts: dict[str, Optional[str]] = {}
    # the config the battery actually ran under (file + resolved content) ...
    if cfg.source_path is not None:
        artefacts[f"config/{cfg.source_path.name}"] = _sha256(cfg.source_path)
    artefacts["config_resolved"] = hashlib.sha256(
        json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    # ... plus the phase configs the frozen checkpoints were trained from
    for p in ("phase1.yaml", "phase2.yaml"):
        artefacts[f"config/{p}"] = _sha256(root / "configs" / p)

    splits_dir = cfg.path("splits")
    for key, fname in SPLIT_FILES.items():
        digest = _sha256(splits_dir / fname)
        if digest is not None or key != "ood":  # ood is optional; others record None if absent
            artefacts[f"splits/{fname}"] = digest
    for label, ck in checkpoints.items():
        artefacts[f"checkpoint/{label}"] = _sha256(cfg.resolve(ck) / "model.pt")

    def _seed(name):
        try:
            import yaml

            return yaml.safe_load((root / "configs" / name).read_text(encoding="utf-8")).get("seed")
        except Exception:  # noqa: BLE001
            return None

    return {
        "phase": phase,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        "git_dirty": dirty,
        "config_path": str(cfg.source_path) if cfg.source_path is not None else None,
        "splits_dir": str(splits_dir),
        "seed": {"phase1": _seed("phase1.yaml"), "phase2": _seed("phase2.yaml")},
        "pool_models": pool_models,
        "n_pool_models": len(pool_models),
        "artefact_hashes": artefacts,
    }


# --------------------------------------------------------------------------- #
# orchestration                                                               #
# --------------------------------------------------------------------------- #
def run_battery(
    phase: str,
    *,
    cfg: Optional[Config] = None,
    split: str = "test",
    zoib: str = "artifacts/phase2/zoib",
    bernoulli: str = "artifacts/phase1/baseline",
    zoib_projected: str = "artifacts/phase2/zoib_projected",
    nirt_run: Optional[str] = "nirt-2d-projected",
    reference: str = REFERENCE,
    lcb_level: float = 0.8,
    added_models: Sequence[str] = (),
    sources: Sequence[str] = ("routerbench",),
    pool_models: Optional[Sequence[str]] = None,
    ablation_drop_query_suffix: Optional[str] = None,
    notes: str = "",
    write: bool = True,
) -> dict:
    cfg = cfg or load_config()
    d = load_training_data(cfg)

    # The phase's routing pool is exactly what its ZOIB checkpoint was trained on
    # -- pinned from the checkpoint so the battery stays self-consistent even
    # after a later phase grows the global NIRT dataset.
    if pool_models is None:
        _, _zblob, _ = load_run(zoib)
        pool_models = sorted(_zblob["model_index"], key=_zblob["model_index"].get)
    pool_models = list(pool_models)

    pool = pool_description(d, split, pool_models=pool_models)

    pred = prediction_quality(zoib, bernoulli, cfg, split)
    # every checkpoint is scored once here; routing + all ablations reuse the frames
    preds, nirt_error = prediction_matrices(
        d, cfg, split, zoib=zoib, bernoulli=bernoulli, nirt_run=nirt_run, lcb_level=lcb_level,
    )
    val_zoib_ey = (
        preds["ZOIB  E[Y]"] if split == _VALIDATION else _pred_matrix(zoib, cfg, _VALIDATION, "mean")
    )
    routing = routing_block(
        d, split, preds=preds, val_zoib_ey=val_zoib_ey, nirt_run=nirt_run, nirt_error=nirt_error,
        pool_models=pool_models, reference=reference,
    )
    cold = cold_start_block(zoib_projected, cfg, split)
    abl = ablation_block(
        d, split, added_models=added_models, full_pool=pool_models,
        full_derived=routing["derived"], preds=preds, val_zoib_ey=val_zoib_ey,
        reference=reference, drop_query_suffix=ablation_drop_query_suffix,
    )
    shots = shot_breakdown(d, split, pool_models=pool_models)
    prov = provenance(
        cfg, phase=phase, pool_models=pool_models,
        checkpoints={"zoib": zoib, "bernoulli": bernoulli, "zoib_projected": zoib_projected},
    )

    result = {
        "phase": phase,
        "split": split,
        "sources": sorted(set(sources)),
        "provenance": prov,
        "pool": pool,
        "prediction": pred,
        "routing": routing,
        "cold_start": cold,
        "ablation": abl,
        "shot_breakdown": shots,
        "notes": notes,
    }

    if write:
        out = cfg.root / "artifacts" / "pool_expansion" / phase
        out.mkdir(parents=True, exist_ok=True)
        (out / "battery.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        _subfiles = ["pool", "prediction", "routing", "cold_start", "ablation", "provenance"]
        if result.get("shot_breakdown"):
            _subfiles.append("shot_breakdown")
        for k in _subfiles:
            (out / f"{k}.json").write_text(json.dumps(result[k], indent=2, default=float), encoding="utf-8")
        append_ledger(result, cfg=cfg)
        render_ledger_md(cfg=cfg)
    return result


# --------------------------------------------------------------------------- #
# ledger                                                                      #
# --------------------------------------------------------------------------- #
def ledger_row(result: dict) -> dict:
    pool, routing, pred = result["pool"], result["routing"], result["prediction"]
    der = routing["derived"]
    def _saving(key):
        s = der.get(key)
        return s["cost_saving_vs_ref"] if s else None

    notes = result.get("notes", "")
    if routing.get("nirt_error"):
        failed = f"NIRT run {routing.get('nirt_run')!r} failed: {routing['nirt_error']}"
        notes = f"{notes}; {failed}" if notes else failed
    return {
        "phase": result["phase"],
        "n_warm_models": pool["n_warm_models"],
        "sources": ",".join(result["sources"]),
        "gpt4_cheapest_cost_ratio": pool["gpt4_cheapest_cost_ratio"],
        "zoib_test_nll": pred["zoib"]["nll_mean"],
        "zoib_test_mae": pred["zoib"]["mae"],
        "zoib_mean_cal_ece": pred["zoib"]["mean_calibration_ece"],
        "oracle_cost_saving_ceiling": der["oracle_cost_saving_ceiling"],
        "zoib_saving_at_minus1pt": _saving("router_saving_at_minus1pt"),
        "zoib_saving_at_minus3pt": _saving("router_saving_at_minus3pt"),
        "zoib_saving_at_minus1pt_test_ceiling": _saving("router_saving_at_minus1pt_test_ceiling"),
        "zoib_saving_at_minus3pt_test_ceiling": _saving("router_saving_at_minus3pt_test_ceiling"),
        "zoib_aiq_improvement": routing["zoib_aiq"].get("aiq_improvement"),
        "oracle_offbest_fraction": der["oracle_offbest_fraction"],
        "coldstart_beats_global_mean": (
            result["cold_start"].get("beats_global_mean_on_bce")
            if result["cold_start"].get("status") == "ran" else None
        ),
        "notes": notes,
    }


def append_ledger(result: dict, *, cfg: Optional[Config] = None) -> Path:
    cfg = cfg or load_config()
    path = cfg.root / "artifacts" / "pool_expansion" / "ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(path.read_text()) if path.exists() else []
    row = ledger_row(result)
    rows = [r for r in rows if r.get("phase") != row["phase"]]
    rows.append(row)
    rows.sort(key=lambda r: r["phase"])
    path.write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    return path


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_ledger_md(*, cfg: Optional[Config] = None) -> Path:
    cfg = cfg or load_config()
    ledger = cfg.root / "artifacts" / "pool_expansion" / "ledger.json"
    rows = json.loads(ledger.read_text()) if ledger.exists() else []
    out = cfg.root / "docs" / "pool_expansion_results.md"

    lines = [
        "# Candidate-pool expansion -- results ledger",
        "",
        "Auto-rendered from `artifacts/pool_expansion/ledger.json` by "
        "`evaluation.pool_expansion.render_ledger_md`. One row per phase; the predictor "
        "(frozen ZOIB head + Phase 1 Bernoulli control) is never retuned.",
        "",
        "`oracle_cost_saving_ceiling` = `1 - oracle_cost / best-single-model_cost` "
        "(oracle = cheapest model attaining each query's max true score). "
        "`zoib_saving_at_minus{1,3}pt` = test cost-saving vs the reference model at the "
        "lambda chosen on the **validation** split (cheapest validation frontier point "
        "within {1,3} points of the best single model's validation accuracy). "
        "`zoib_saving_at_minus{1,3}pt_test_ceiling` = the same rule applied to the test "
        "frontier directly -- an optimistic ceiling, not a deployable number "
        "(`n/a` = no frontier point qualifies, or no validation split).",
        "",
    ]
    header = "| " + " | ".join(LEDGER_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in LEDGER_COLUMNS) + " |"
    lines += [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in LEDGER_COLUMNS) + " |")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
