"""Evaluate a trained Phase 1 baseline checkpoint.

    from router.nirt.baseline_eval import evaluate_checkpoint
    evaluate_checkpoint("artifacts/phase1/baseline", split="test")

Prediction metrics, calibration, per-model / per-family breakdowns, theta
spectrum, ICC curves, and (for ``model_params="projected"`` runs) cold-start
metrics -- written under the checkpoint dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Config, load_config, section
from .baseline_data import _join, batched_forward, build_arrays
from .calibration import calibration_report, plot_reliability, save_calibration
from .checkpoint import load_run
from .diagnostics import (
    icc_curve,
    parameter_summary,
    plot_icc,
    plot_theta_spectrum,
    save_theta_spectrum,
    theta_spectrum,
)
from .metrics import marginal_baselines, prediction_metrics


def profile_matrix(cfg: Config, model_index: dict, pathway: str) -> np.ndarray:
    """``(M, profile_dim)`` profile embeddings in ``model_index`` order."""
    from ..data.phase1 import load_phase1

    store = load_phase1(cfg).profile_embeddings(pathway)
    order = [m for m in sorted(model_index, key=model_index.get) if m in store]
    return np.ascontiguousarray(store.gather(order), dtype=np.float32)


def theta_matrix(model, cfg: Config, model_index: dict, pathway: str) -> np.ndarray:
    """Full ``(M, K)`` ability matrix -- ``theta`` table (free) or ``W_theta @ e_m``."""
    import torch

    if model.model_params == "free":
        return model.theta.weight.detach().numpy()
    return model.theta_table(torch.from_numpy(profile_matrix(cfg, model_index, pathway))).numpy()


def family_labels(query_ids, cfg: Config, queries_df=None):
    keymap = {str(k).lower(): f
              for f, keys in (section(cfg, "profiles").get("task_families", {}) or {}).items()
              for k in keys}
    if queries_df is None:
        import pandas as pd

        queries_df = pd.read_parquet(cfg.path("processed") / "queries.parquet",
                                     columns=["query_id", "dataset"])
    ds = queries_df.set_index("query_id")["dataset"].to_dict()
    return [next((f for k, f in keymap.items() if k in (ds.get(q, "") or "").lower()), "other")
            for q in query_ids]


def _per_group(y, p, keys) -> dict:
    keys = np.asarray(keys)
    out = {}
    for k in np.unique(keys):
        m = keys == k
        if m.sum() >= 20:
            pm = prediction_metrics(y[m], p[m])
            out[str(k)] = {"n": int(m.sum()), **{j: pm[j] for j in ("bce", "brier", "accuracy", "pos_rate")}}
    return out


def _iccs(query_ids, a, b, theta, n: int) -> dict:
    """easy / medium / hard / high- and low-discrimination representative queries."""
    seen = {}
    for i, q in enumerate(query_ids):
        seen.setdefault(q, i)
    idx = np.fromiter(seen.values(), dtype=int)
    order = np.argsort(b[idx])
    a_norm = np.linalg.norm(a[idx], axis=1)
    picks = {"easy": idx[order[0]], "medium": idx[order[len(order) // 2]], "hard": idx[order[-1]],
             "high_discrimination": idx[a_norm.argmax()], "low_discrimination": idx[a_norm.argmin()]}
    theta_mean = theta.mean(axis=0)
    curves = {}
    for label, i in list(picks.items())[:n]:
        dim = int(np.argmax(np.abs(a[i])))
        lo, hi = float(theta[:, dim].min()), float(theta[:, dim].max())
        pad = 0.5 * (hi - lo + 1e-6)
        curves[label] = icc_curve(a[i], float(b[i]), dim_index=dim, theta_hold=theta_mean,
                                  sweep=np.linspace(lo - pad, hi + pad, 81))
    return curves


def evaluate_checkpoint(
    directory: str | Path, *, phase0_cfg: Optional[Config] = None,
    split: str = "test", write: bool = True, n_icc: int = 6,
) -> dict:
    d = Path(directory)
    cfg = phase0_cfg or load_config()
    model, blob, s = load_run(d)
    mi = blob["model_index"]

    common = dict(pathway=s["pathway"], binary_threshold=s["binary_threshold"],
                  score_kind=s["score_kind"], use_relevance=model.use_relevance,
                  use_warmup=model.use_warmup, model_index=mi)
    tr = build_arrays(cfg, split="train", **common)
    ev = build_arrays(cfg, split=split, **common)

    fwd = batched_forward(model, ev, fields=("proba", "a_q", "b_q"))
    p, a_all, b_all = fwd["proba"], fwd["a_q"], fwd["b_q"]

    metrics = prediction_metrics(ev.y, p)
    base = marginal_baselines(tr.y, tr.model_ids, ev.y, ev.model_ids)
    metrics["delta_bce_vs_model_mean"] = metrics["bce"] - base["model_mean"]["bce"]
    calib = calibration_report(ev.y, p)
    theta = theta_matrix(model, cfg, mi, s["pathway"])
    spec = theta_spectrum(theta)
    iccs = _iccs(ev.query_ids, a_all, b_all, theta, n_icc)

    cold = None
    if model.model_params == "projected":
        cold = cold_start_baseline(model, cfg, blob, split=split, **s)

    result = {
        "split": split,
        "prediction": metrics,
        "marginal_baselines": {k: {kk: vv for kk, vv in v.items() if kk != "per_model"}
                               for k, v in base.items()},
        "calibration": {k: calib[k] for k in ("ece", "mce", "brier", "log_loss",
                                              "mean_prediction", "base_rate")},
        "per_model": _per_group(ev.y, p, ev.model_ids),
        "per_family": _per_group(ev.y, p, family_labels(ev.query_ids, cfg)),
        "theta_spectrum": spec,
        "parameter_summary": parameter_summary(
            theta=theta, a_q=a_all, b_q=b_all,
            discrimination_constrained=bool(model.disc_head.constrain)),
        "n_icc": len(iccs),
        "cold_start": cold,
    }

    if write:
        prev = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
        (d / "metrics.json").write_text(json.dumps({**prev, split: result}, indent=2, default=float),
                                        encoding="utf-8")
        save_calibration(calib, d)
        save_theta_spectrum(spec, d)
        plot_reliability(calib, d / "plots" / "calibration.png")
        plot_theta_spectrum(spec, d / "plots" / "theta_spectrum.png")
        icc_dir = d / "plots" / "icc"
        icc_dir.mkdir(parents=True, exist_ok=True)
        (icc_dir / "icc_curves.json").write_text(json.dumps(iccs, indent=2, default=float),
                                                 encoding="utf-8")
        plot_icc(iccs, icc_dir / "icc_all.png")
    return result


def cold_start_baseline(
    model, cfg: Config, blob: dict, *, split: str = "test",
    pathway: str = "irt", binary_threshold: float = 0.5, score_kind: str = "effective",
) -> dict:
    """Score held-out (cold-start) LLMs from their profile embedding alone
    (``theta_m = W_theta @ e_m``), vs predicting the training correctness rate.
    Needs a ``model_params="projected"`` run."""
    import pandas as pd
    import torch

    from ..data.phase1 import load_phase1
    from ..retrieval import load_warmup
    from ..taxonomy import load_relevance

    if model.model_params != "projected":
        return {"skipped": "cold-start needs a model_params='projected' run"}

    d = load_phase1(cfg)
    p_store, q_store = d.profile_embeddings(pathway), d.query_embeddings(pathway)
    cold_ids = [m for m in d.cold_start_model_ids() if m in p_store]
    long = d.correctness(split=split, models="cold", score_kind=score_kind)
    long = long[long["model_id"].isin(cold_ids) & long["query_id"].isin(q_store._index)]
    if long.empty:
        return {"skipped": f"no cold-start observations in split '{split}'", "cold_models": cold_ids}

    rel = load_relevance(cfg) if model.use_relevance else None
    wu = load_warmup(cfg, pathway) if model.use_warmup else None
    global_rate = float(build_arrays(cfg, split="train", pathway=pathway,
                                     binary_threshold=binary_threshold, score_kind=score_kind,
                                     use_relevance=False, use_warmup=False,
                                     model_index=blob["model_index"]).y.mean())

    keys = ("bce", "brier", "accuracy", "auc")
    nll = None
    per_model, pooled_y, pooled_p, pooled_base = {}, [], [], []
    for m in cold_ids:
        sub = long[long["model_id"] == m]
        qids = np.array([q for q in sub["query_id"].to_numpy() if q in q_store._index])
        if len(qids) == 0:
            per_model[m] = {"n": 0, "skipped": "no rows in the query store"}
            continue
        y_soft = np.clip(pd.to_numeric(sub[sub["query_id"].isin(set(qids))]["score"],
                                       errors="coerce").to_numpy(np.float64), 0, 1)
        y = (y_soft >= binary_threshold).astype(np.float64)
        e_q = torch.from_numpy(np.ascontiguousarray(q_store.gather(qids), np.float32))
        e_m = torch.from_numpy(np.tile(p_store.get(m).astype(np.float32), (len(qids), 1)))
        r_q = torch.from_numpy(_join(rel, qids, 1.0 / rel.dim)) if rel is not None else None
        nbr = torch.from_numpy(_join(wu, qids, 0.0)) if wu is not None else None
        with torch.no_grad():
            o = model(e_q, e_m, r_q, nbr)
            prob = model.response_head.as_probability(o.response).numpy().astype(np.float64)
            if model.response_model != "bernoulli":
                nll = float(model.response_head.nll(torch.from_numpy(y_soft.astype(np.float32)),
                                                    o.response).mean())
        pm = prediction_metrics(y, prob)
        if model.response_model != "bernoulli":
            pm["nll"] = nll
        per_model[m] = {"n": int(len(y)), **{k: pm.get(k) for k in keys}}
        pooled_y.append(y); pooled_p.append(prob); pooled_base.append(np.full(len(y), global_rate))

    if not pooled_y:
        return {"skipped": "no usable cold-start observations", "cold_models": cold_ids}
    yy = np.concatenate(pooled_y)
    return {
        "cold_models": cold_ids, "split": split, "per_model": per_model,
        "pooled": {
            "baseline_nirt": {k: prediction_metrics(yy, np.concatenate(pooled_p))[k] for k in keys},
            "global_mean": {k: prediction_metrics(yy, np.concatenate(pooled_base))[k] for k in keys[:3]},
        },
        "note": "cold-start theta_m = W_theta @ profile_embedding; "
                "global_mean = predict the overall training correctness rate.",
    }
