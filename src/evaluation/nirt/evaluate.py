"""Evaluate a trained NIRT / IRT-Router model against ground truth.

    from router.nirt.checkpoint import load_run
    from evaluation.nirt.evaluate import evaluate_split, cold_start_eval

    model, cfg, model_index = load_run("nirt-1d-projected")
    evaluate_split(model, model_index, "test")          # prediction quality
    cold_start_eval(model, model_index, split="test")   # place a held-out LLM from its profile

Label-free prediction (``predict_dataset``, ``predict_matrix``) lives in
:mod:`router.nirt.predict` -- this module is evaluation-only because
everything here needs the observed outcome ``true``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from router.config import load_config
from router.nirt.predict import predict_dataset, predict_matrix
from router.nirt.routing_decision import regret as _regret, routing_decision
from training.data.facade import load_training_data
from training.nirt.metrics import marginal_baselines, prediction_metrics


def ranking_metrics(pred: np.ndarray, true: np.ndarray, tol: float = 1e-9) -> dict:
    """Per-query ranking quality of ``pred`` against observed ``true`` [Q x M].

    RouterBench targets are graded and heavily tied, so "top-1" is defined as
    *picking a best-scoring model* (``true >= row_max - tol``), not matching a
    single arg-max index.
    """
    from scipy.stats import spearmanr

    n_q = len(true)
    argmax_pred = routing_decision(pred)   # lam=0 -- quality-only, same as pred.argmax(1)
    qi = np.arange(n_q)
    row_max = true.max(1)

    optimal = float((true[qi, argmax_pred] >= row_max - tol).mean())
    regret = float(_regret(pred, true, selected=argmax_pred).mean())

    rhos, pair_acc = [], []
    for r_p, r_t in zip(pred, true):
        if np.std(r_p) > 1e-9 and np.std(r_t) > 1e-9:
            rhos.append(spearmanr(r_p, r_t).statistic)
        d_p = r_p[:, None] - r_p[None, :]
        d_t = r_t[:, None] - r_t[None, :]
        m = np.triu(np.ones_like(d_p, dtype=bool), k=1) & (d_t != 0)
        if m.any():
            pair_acc.append(float((np.sign(d_p[m]) == np.sign(d_t[m])).mean()))
    return {
        "optimal_rate": optimal,
        "regret": regret,
        "spearman_mean": float(np.nanmean(rhos)) if rhos else float("nan"),
        "pairwise_accuracy": float(np.mean(pair_acc)) if pair_acc else float("nan"),
    }


def evaluate_split(
    model,
    model_index: dict,
    split: str,
    *,
    phase0_cfg=None,
    data=None,
    pathway: str = "irt",
    query_pathway: str | None = None,
    query_features: str | None = None,
    train_ds=None,
) -> dict:
    """Prediction metrics on ``split`` plus the marginal baselines for context."""
    if data is None:
        data = load_training_data(phase0_cfg or load_config())

    ds = data.nirt_dataset(split=split, pathway=pathway, query_pathway=query_pathway,
                           query_features=query_features)
    y, p = predict_dataset(model, ds, model_index)
    metrics = prediction_metrics(y, p)

    if train_ds is None:
        train_ds = data.nirt_dataset(split="train", pathway=pathway, query_pathway=query_pathway,
                                     query_features=query_features)
    base = marginal_baselines(train_ds.targets, train_ds.model_ids, y, ds.model_ids)

    metrics["split"] = split
    metrics["baselines"] = base
    metrics["delta_bce_vs_model_mean"] = metrics["bce"] - base["model_mean"]["bce"]
    metrics["delta_bce_vs_global_mean"] = metrics["bce"] - base["global_mean"]["bce"]
    return metrics


# --------------------------------------------------------------------------- #
# cold-start: place a held-out LLM from its profile embedding alone            #
# --------------------------------------------------------------------------- #
def nearest_profile_model(profile_store, target_id: str, candidates) -> str:
    """The candidate whose profile embedding is closest (cosine) to ``target_id``."""
    t = np.asarray(profile_store.get(target_id), dtype=np.float64)
    t = t / (np.linalg.norm(t) + 1e-12)
    best, best_sim = None, -np.inf
    for c in candidates:
        v = np.asarray(profile_store.get(c), dtype=np.float64)
        sim = float(v @ t / (np.linalg.norm(v) + 1e-12))
        if sim > best_sim:
            best, best_sim = c, sim
    return best


def _predict_for_model(model, e_q: np.ndarray, profile_vec: np.ndarray) -> np.ndarray:
    import torch

    Em = np.repeat(np.asarray(profile_vec, np.float32)[None], len(e_q), axis=0)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(
            model(torch.from_numpy(np.ascontiguousarray(e_q, np.float32)), torch.from_numpy(Em))
        ).numpy().astype(np.float64)


def cold_start_eval(
    model,
    model_index: dict,
    *,
    split: str = "test",
    pathway: str = "irt",
    phase0_cfg=None,
    data=None,
) -> dict:
    """Score cold-start LLMs (held out of training) on ``split`` from their profile
    embedding alone, and compare to no-NIRT baselines.

    Needs a ``model_params="projected"`` run -- a ``free`` model has no parameters
    for an unseen LLM.
    """
    if getattr(model, "model_params", None) != "projected":
        raise ValueError(
            "cold_start_eval needs a 'projected' run; a 'free' model cannot place an "
            "LLM it never trained on."
        )
    if data is None:
        data = load_training_data(phase0_cfg or load_config())

    q_store = data.query_embeddings(pathway)
    p_store = data.profile_embeddings(pathway)
    if q_store is None or p_store is None:
        raise FileNotFoundError(f"embeddings for pathway '{pathway}' not built")

    warm_ids = sorted(model_index, key=model_index.get)
    cold_ids = [m for m in data.cold_start_model_ids() if m in p_store._index]

    long = data.correctness(split=split, models="cold", score_kind="effective")
    long = long[long["model_id"].isin(cold_ids) & long["query_id"].isin(q_store._index)]
    cold_ids = [m for m in cold_ids if (long["model_id"] == m).any()]
    if not cold_ids:
        raise ValueError("no cold-start models with correctness observations in this split")

    # warm predicted + true matrices for the split (for placement / baselines)
    from .routing import eval_matrices

    true_w, cost_w = eval_matrices(data, split=split)
    warm_pred = predict_matrix(model, model_index, data, split, pathway).reindex(
        index=true_w.index, columns=true_w.columns
    )
    train_ds = data.nirt_dataset(split="train", pathway=pathway)
    global_rate = float(np.asarray(train_ds.targets, np.float64).mean())

    per_model, pooled_true, pooled_pred = {}, [], []
    base_gm, base_wm, base_nw = [], [], []
    aug_pred_cols, aug_true_cols = {}, {}

    for m in cold_ids:
        sub = long[long["model_id"] == m]
        qids = sub["query_id"].to_numpy()
        y = pd.to_numeric(sub["score"], errors="coerce").to_numpy(np.float64)
        e_q = q_store.gather(qids)
        p = _predict_for_model(model, e_q, p_store.get(m))

        nw = nearest_profile_model(p_store, m, warm_ids)
        in_w = np.array([q in warm_pred.index for q in qids])
        wm_pred = np.full(len(qids), global_rate)
        nw_pred = np.full(len(qids), global_rate)
        if in_w.any():
            idx = warm_pred.index.get_indexer(qids[in_w])
            wm_pred[in_w] = warm_pred.to_numpy()[idx].mean(1)
            nw_pred[in_w] = warm_pred[nw].to_numpy()[idx]

        per_model[m] = {
            "n": int(len(y)),
            "prediction": prediction_metrics(y, p),
            "nearest_warm_model": nw,
            "by_family": _family_breakdown(qids, y, p, data),
        }
        pooled_true.append(y); pooled_pred.append(p)
        base_gm.append((y, np.full(len(y), global_rate)))
        base_wm.append((y, wm_pred)); base_nw.append((y, nw_pred))
        aug_pred_cols[m] = pd.Series(p, index=qids)
        aug_true_cols[m] = pd.Series(y, index=qids)

    def _pool(pairs):
        yy = np.concatenate([a for a, _ in pairs]); pp = np.concatenate([b for _, b in pairs])
        return prediction_metrics(yy, pp)

    pooled = {
        "nirt": prediction_metrics(np.concatenate(pooled_true), np.concatenate(pooled_pred)),
        "global_mean": _pool(base_gm),
        "warm_mean": _pool(base_wm),
        "nearest_warm": _pool(base_nw),
    }

    # -- placement: rank of the cold model among the warm pool --------------
    aug_pred = warm_pred.copy()
    aug_true = true_w.copy()
    for m in cold_ids:
        aug_pred[m] = aug_pred.index.to_series().map(aug_pred_cols[m])
        aug_true[m] = aug_true.index.to_series().map(aug_true_cols[m])
    keep = aug_pred.notna().all(1) & aug_true.notna().all(1)
    ap, at = aug_pred.loc[keep].to_numpy(), aug_true.loc[keep].to_numpy()
    cold_pos = [list(aug_pred.columns).index(m) for m in cold_ids]
    placement = {}
    for m, j in zip(cold_ids, cold_pos):
        rank_pred = (ap > ap[:, [j]]).sum(1)
        rank_true = (at > at[:, [j]]).sum(1)
        placement[m] = {
            "mean_abs_rank_error": float(np.abs(rank_pred - rank_true).mean()),
            "mean_rank_pred": float(rank_pred.mean()),
            "mean_rank_true": float(rank_true.mean()),
        }
    rank_full = ranking_metrics(ap, at)

    return {
        "split": split,
        "pathway": pathway,
        "orientation": getattr(model, "orientation", "?"),
        "cold_models": cold_ids,
        "warm_models": warm_ids,
        "per_model": per_model,
        "pooled_prediction": pooled,
        "placement": placement,
        "augmented_ranking": rank_full,
        "notes": "warm_mean/nearest_warm are no-NIRT references; global_mean == "
                 "predict the overall training correctness rate.",
    }


def _family_breakdown(qids, y, p, data) -> dict:
    """Predicted vs observed correctness split by benchmark family, via
    :func:`training.data.families.family_labels` -- the canonical family map
    (``configs/phase0.yaml``'s ``profiles.task_families``), not a private
    keyword list that could diverge from it (XD-03)."""
    from training.data.families import family_labels

    labels = np.asarray(family_labels(list(qids), data.cfg, queries_df=data.queries))
    out = {}
    for fam in sorted(set(labels)):
        if fam == "other":    # unmapped -- not a recognised benchmark family
            continue
        mask = labels == fam
        if mask.sum() >= 20:
            out[fam] = {"n": int(mask.sum()), "true": round(float(y[mask].mean()), 4),
                        "pred": round(float(p[mask].mean()), 4)}
    return out
