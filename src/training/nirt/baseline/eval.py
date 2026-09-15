"""Evaluate a trained Phase 1 baseline checkpoint.

    from router.nirt.baseline.eval import evaluate_checkpoint
    evaluate_checkpoint("artifacts/phase1/baseline", split="test")

Prediction metrics, calibration, per-model / per-family breakdowns, theta
spectrum, ICC curves, and (for ``model_params="projected"`` runs) cold-start
metrics -- written under the checkpoint dir (``metrics.json -> eval.<split>``,
so the trainer's own ``validation`` block is never overwritten).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from router.config import Config, load_config
from training.data.families import family_labels

from .calibration import calibration_report, plot_reliability, save_calibration
from .checkpoint import load_run
from .data import _join, batched_forward, build_arrays, graded_labels
from .diagnostics import (
    icc_curve,
    parameter_summary,
    plot_icc,
    plot_theta_spectrum,
    save_theta_spectrum,
    theta_spectrum,
)
from training.nirt.metrics import marginal_baselines, prediction_metrics, reliability_curve


def profile_matrix(cfg: Config, model_index: dict, pathway: str, *, data=None) -> np.ndarray:
    """``(M, profile_dim)`` profile embeddings in ``model_index`` order.

    Raises if the store lacks any checkpoint model -- silently dropping rows
    would misalign the matrix with ``model_index``."""
    from training.data.facade import load_training_data

    store = (data or load_training_data(cfg)).profile_embeddings(pathway)
    ordered = sorted(model_index, key=model_index.get)
    missing = [m for m in ordered if store is None or m not in store]
    if missing:
        raise KeyError(
            f"profile store for pathway '{pathway}' lacks {len(missing)} checkpoint model(s) "
            f"{missing[:10]}; rebuild the profile embeddings"
        )
    return np.ascontiguousarray(store.gather(ordered), dtype=np.float32)


def theta_matrix(model, cfg: Config, model_index: dict, pathway: str, *, data=None) -> np.ndarray:
    """Full ``(M, K)`` ability matrix -- ``theta`` table (free) or ``W_theta @ e_m``."""
    import torch

    if model.model_params == "free":
        return model.theta.weight.detach().numpy()
    return model.theta_table(
        torch.from_numpy(profile_matrix(cfg, model_index, pathway, data=data))).numpy()


def _per_group(y, p, keys) -> dict:
    keys = np.asarray(keys)
    out = {}
    for k in np.unique(keys):
        m = keys == k
        if m.sum() >= 20:
            pm = prediction_metrics(y[m], p[m])
            out[str(k)] = {"n": int(m.sum()), **{j: pm[j] for j in ("bce", "brier", "accuracy", "pos_rate")}}
    return out


def _graded_block(y_soft, p) -> dict:
    """MAE / RMSE / mean-calibration of ``p`` as the expected graded score --
    the same target (``y_soft``) and binning the continuous heads are scored on,
    so the Bernoulli row of :func:`compare_response_models` is comparable."""
    y = np.asarray(y_soft, np.float64)
    err = np.asarray(p, np.float64) - y
    rc = reliability_curve(y, np.clip(p, 0, 1), n_bins=15)
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mean_calibration_ece": rc["ece"]}


def _interaction_logit_fn(model, a_i, b_i, r_i):
    """``theta rows -> full model logit`` for one query, including the interaction
    residual, so an ICC describes the fitted model rather than the base IRT term."""
    import torch

    def fn(theta_rows):
        n = len(theta_rows)
        A = torch.as_tensor(np.tile(a_i, (n, 1)), dtype=torch.float32)
        B = torch.full((n,), float(b_i))
        R = (torch.as_tensor(np.tile(r_i, (n, 1)), dtype=torch.float32)
             if r_i is not None and model.use_relevance else None)
        with torch.no_grad():
            logit, _ = model.interaction(A, torch.as_tensor(theta_rows, dtype=torch.float32), B, R)
        return logit.numpy().astype(np.float64)

    return fn


def _iccs(query_ids, a, b, theta, n: int, *, model=None, r_q=None) -> dict:
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
    use_interaction = model is not None and bool(getattr(model, "use_interaction", False))
    curves = {}
    for label, i in list(picks.items())[:n]:
        dim = int(np.argmax(np.abs(a[i])))
        lo, hi = float(theta[:, dim].min()), float(theta[:, dim].max())
        pad = 0.5 * (hi - lo + 1e-6)
        fn = (_interaction_logit_fn(model, a[i], b[i], None if r_q is None else r_q[i])
              if use_interaction else None)
        curves[label] = icc_curve(a[i], float(b[i]), dim_index=dim, theta_hold=theta_mean,
                                  sweep=np.linspace(lo - pad, hi + pad, 81), logit_fn=fn)
    return curves


def _write_eval(directory: Path, split: str, result: dict) -> None:
    path = directory / "metrics.json"
    prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    evals = dict(prev.get("eval", {}) or {})
    evals[split] = result
    path.write_text(json.dumps({**prev, "eval": evals}, indent=2, default=float), encoding="utf-8")


def evaluate_checkpoint(
    directory: str | Path, *, phase0_cfg: Optional[Config] = None,
    split: str = "test", write: bool = True, n_icc: int = 6,
) -> dict:
    from training.data.facade import load_training_data

    d = Path(directory)
    cfg = phase0_cfg or load_config()
    model, blob, s = load_run(d)
    mi = blob["model_index"]
    data = load_training_data(cfg)          # once per evaluation, shared below

    common = dict(data=data, pathway=s["pathway"], binary_threshold=s["binary_threshold"],
                  score_kind=s["score_kind"], model_index=mi)
    # train arrays only feed the marginal baselines (y, model_ids): skip the joins
    tr = build_arrays(cfg, split="train", use_relevance=False, use_warmup=False, **common)
    ev = build_arrays(cfg, split=split, use_relevance=model.use_relevance,
                      use_warmup=model.use_warmup, **common)

    fwd = batched_forward(model, ev, fields=("proba", "a_q", "b_q"))
    p, a_all, b_all = fwd["proba"], fwd["a_q"], fwd["b_q"]

    metrics = prediction_metrics(ev.y, p)
    base = marginal_baselines(tr.y, tr.model_ids, ev.y, ev.model_ids)
    metrics["delta_bce_vs_model_mean"] = metrics["bce"] - base["model_mean"]["bce"]
    calib = calibration_report(ev.y, p)
    theta = theta_matrix(model, cfg, mi, s["pathway"], data=data)
    spec = theta_spectrum(theta)
    iccs = _iccs(ev.query_ids, a_all, b_all, theta, n_icc, model=model, r_q=ev.r_q)

    cold = None
    if model.model_params == "projected":
        cold = cold_start_baseline(model, cfg, blob, split=split, data=data, **s)

    result = {
        "split": split,
        "prediction": metrics,
        "graded": _graded_block(ev.y_soft, p),
        "marginal_baselines": {k: {kk: vv for kk, vv in v.items() if kk != "per_model"}
                               for k, v in base.items()},
        "calibration": {k: calib[k] for k in ("ece", "mce", "brier", "log_loss",
                                              "mean_prediction", "base_rate")},
        "per_model": _per_group(ev.y, p, ev.model_ids),
        "per_family": _per_group(ev.y, p, family_labels(ev.query_ids, cfg, queries_df=data.queries)),
        "theta_spectrum": spec,
        "parameter_summary": parameter_summary(
            theta=theta, a_q=a_all, b_q=b_all,
            discrimination_constrained=bool(model.disc_head.constrain)),
        "interaction_gamma": (float(model.interaction.gamma.detach())
                              if model.use_interaction else None),
        "n_icc": len(iccs),
        "cold_start": cold,
    }

    if write:
        _write_eval(d, split, result)
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
    data=None,
) -> dict:
    """Score held-out (cold-start) LLMs from their profile embedding alone
    (``theta_m = W_theta @ e_m``), vs predicting the training correctness rate.
    Needs a ``model_params="projected"`` run. Labels use :func:`graded_labels`,
    the same definition as :func:`build_arrays`."""
    import pandas as pd
    import torch

    from training.data.facade import load_training_data
    from training.retrieval import load_warmup
    from training.taxonomy import load_relevance

    if model.model_params != "projected":
        return {"skipped": "cold-start needs a model_params='projected' run"}

    d = data or load_training_data(cfg)
    p_store, q_store = d.profile_embeddings(pathway), d.query_embeddings(pathway)
    cold_ids = [m for m in d.cold_start_model_ids() if m in p_store]
    long = d.correctness(split=split, models="cold", score_kind=score_kind)
    long = long[long["model_id"].isin(cold_ids) & long["query_id"].isin(q_store._index)]
    if long.empty:
        return {"skipped": f"no cold-start observations in split '{split}'", "cold_models": cold_ids}

    rel = load_relevance(cfg) if model.use_relevance else None
    wu = load_warmup(cfg, pathway) if model.use_warmup else None
    global_rate = float(build_arrays(cfg, split="train", data=d, pathway=pathway,
                                     binary_threshold=binary_threshold, score_kind=score_kind,
                                     use_relevance=False, use_warmup=False,
                                     model_index=blob["model_index"]).y.mean())

    continuous = model.response_model != "bernoulli"
    keys = ("bce", "brier", "accuracy", "auc")
    per_model, pooled_y, pooled_p, pooled_base, pooled_nll = {}, [], [], [], []
    for m in cold_ids:
        sub = long[long["model_id"] == m]
        y_soft, y, finite = graded_labels(
            pd.to_numeric(sub["score"], errors="coerce").to_numpy(np.float64), binary_threshold)
        qids = sub["query_id"].to_numpy()[finite]
        y_soft, y = y_soft[finite], y[finite].astype(np.float64)
        if len(qids) == 0:
            per_model[m] = {"n": 0, "skipped": "no finite-score rows in the query store"}
            continue
        e_q = torch.from_numpy(np.ascontiguousarray(q_store.gather(qids), np.float32))
        e_m = torch.from_numpy(np.tile(p_store.get(m).astype(np.float32), (len(qids), 1)))
        r_q = torch.from_numpy(_join(rel, qids, 1.0 / rel.dim)) if rel is not None else None
        nbr = torch.from_numpy(_join(wu, qids, 0.0)) if wu is not None else None
        with torch.no_grad():
            o = model(e_q, e_m, r_q, nbr)
            prob = model.response_head.as_probability(o.response).numpy().astype(np.float64)
            nll_rows = (model.response_head.nll(torch.from_numpy(y_soft), o.response).numpy()
                        if continuous else None)
        pm = prediction_metrics(y, prob)
        row = {"n": int(len(y)), **{k: pm.get(k) for k in keys}}
        if continuous:
            row["nll"] = float(nll_rows.mean())
            pooled_nll.append(nll_rows)
        per_model[m] = row
        pooled_y.append(y); pooled_p.append(prob); pooled_base.append(np.full(len(y), global_rate))

    if not pooled_y:
        return {"skipped": "no usable cold-start observations", "cold_models": cold_ids}
    yy = np.concatenate(pooled_y)
    nirt_block = {k: prediction_metrics(yy, np.concatenate(pooled_p))[k] for k in keys}
    if continuous:
        nirt_block["nll"] = float(np.concatenate(pooled_nll).mean())
    return {
        "cold_models": cold_ids, "split": split, "per_model": per_model,
        "pooled": {
            "baseline_nirt": nirt_block,
            "global_mean": {k: prediction_metrics(yy, np.concatenate(pooled_base))[k] for k in keys[:3]},
        },
        "note": "cold-start theta_m = W_theta @ profile_embedding; "
                "global_mean = predict the overall training correctness rate.",
    }
