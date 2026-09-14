"""Prediction-quality metrics for the NIRT baseline (pure numpy / scipy / sklearn).

v1 evaluates the model as a *predictor* of per-(query, model) correctness. Ranking
/ routing / cold-start evaluation are the next stage and live elsewhere.

``prediction_metrics`` works on graded targets (RouterBench ``performance`` is in
``{0, .25, .5, .75, 1}``): BCE / MSE / MAE are computed against the soft target,
AUC and accuracy binarise at 0.5.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

_EPS = 1e-7


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < _EPS or np.std(b) < _EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < _EPS or np.std(b) < _EPS:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        rho = spearmanr(a, b).statistic
        return float(rho)
    except Exception:  # pragma: no cover - scipy always present here
        ar = np.argsort(np.argsort(a))
        br = np.argsort(np.argsort(b))
        return _corr(ar.astype(float), br.astype(float))


def _auc(y_bin: np.ndarray, y_prob: np.ndarray) -> float:
    if y_bin.min() == y_bin.max():
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_bin, y_prob))
    except Exception:  # pragma: no cover
        return float("nan")


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error against the (soft) outcome rate."""
    return reliability_curve(y_true, y_prob, n_bins=n_bins)["ece"]


def brier_score(y_true: Sequence[float], y_prob: Sequence[float]) -> float:
    """``mean((p - y)^2)`` -- the Brier score (== MSE of the probability)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss(y_true: Sequence[float], y_prob: Sequence[float]) -> float:
    """Mean binary cross-entropy against a (possibly soft) target."""
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), _EPS, 1 - _EPS)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())


def reliability_curve(
    y_true: Sequence[float], y_prob: Sequence[float], n_bins: int = 10
) -> dict:
    """Equal-width reliability curve + ECE / MCE against the (soft) outcome rate."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    total = max(len(y_true), 1)
    rows, ece, mce = [], 0.0, 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                         "count": 0, "confidence": None, "accuracy": None, "gap": None})
            continue
        conf = float(y_prob[m].mean())
        acc = float(y_true[m].mean())
        gap = abs(acc - conf)
        w = m.sum() / total
        ece += w * gap
        mce = max(mce, gap)
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "count": int(m.sum()), "confidence": conf, "accuracy": acc, "gap": gap})
    return {"n_bins": n_bins, "bins": rows, "ece": float(ece), "mce": float(mce)}


def prediction_metrics(y_true: Sequence[float], y_prob: Sequence[float]) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = int(len(y_true))
    if n == 0:
        return {"n": 0}

    p = np.clip(y_prob, _EPS, 1 - _EPS)
    bce = float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())
    y_bin = (y_true >= 0.5).astype(int)
    brier = float(np.mean((y_prob - y_true) ** 2))
    return {
        "n": n,
        "bce": bce,
        "log_loss": bce,
        "mse": brier,
        "brier": brier,
        "mae": float(np.mean(np.abs(y_prob - y_true))),
        "accuracy": float(np.mean((y_prob >= 0.5) == y_bin)),
        "pearson_r": _corr(y_prob, y_true),
        "spearman_r": _spearman(y_prob, y_true),
        "auc": _auc(y_bin, y_prob),
        "ece": _ece(y_true, y_prob),
        "acc@0.5": float(np.mean((y_prob >= 0.5) == y_bin)),
        "pos_rate": float(y_bin.mean()),
        "pred_mean": float(y_prob.mean()),
    }


def _bce_mse(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    p = np.clip(y_prob, _EPS, 1 - _EPS)
    return {
        "bce": float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean()),
        "mse": float(np.mean((y_prob - y_true) ** 2)),
    }


def per_group_metrics(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    groups: Sequence[str],
    *,
    min_n: int = 20,
) -> dict:
    """``prediction_metrics`` computed within each ``groups`` label plus an
    ``overall`` row. Groups with fewer than ``min_n`` observations are pooled into
    ``"(small)"``. Used by the P1 capacity diagnostic to see which benchmark
    families the model underfits (``groups`` = per-observation family labels)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    g = np.asarray(groups).astype(str)
    out: dict[str, dict] = {"overall": prediction_metrics(y_true, y_prob)}
    counts = {lbl: int((g == lbl).sum()) for lbl in np.unique(g)}
    small = np.array([counts[lbl] < min_n for lbl in g])
    for lbl in sorted(counts):
        if counts[lbl] < min_n:
            continue
        m = g == lbl
        out[lbl] = prediction_metrics(y_true[m], y_prob[m])
    if small.any():
        out["(small)"] = prediction_metrics(y_true[small], y_prob[small])
    return out


def effective_rank(theta: np.ndarray) -> float:
    """Participation ratio of ``theta``'s covariance eigenvalues:
    ``(sum(lambda))^2 / sum(lambda^2)``. Ranges from 1 (all variance on one
    line -- the query head has collapsed to a 1-D difficulty scale) to ``K``
    (isotropic -- the full latent space is used). See the P7 monotone-collapse
    diagnosis (``theta_q`` near rank 1 forces single-crossing, family-blind
    routing). ``theta`` is ``[N, K]``; NaN if fewer than 2 rows or if the
    covariance is degenerate (all rows identical)."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[0] < 2:
        return float("nan")
    tc = theta - theta.mean(axis=0, keepdims=True)
    cov = (tc.T @ tc) / len(tc)
    ev = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    denom = float((ev ** 2).sum())
    if denom < 1e-18:
        return float("nan")
    return float((ev.sum() ** 2) / denom)


def bias_argmax_agreement(logit: np.ndarray, b_m: np.ndarray) -> float:
    """Fraction of queries where the full router's ``argmax_m logit(q, m)``
    agrees with the query-INDEPENDENT constant ``argmax_m(-b_m)`` -- "always
    route to the best-on-average model, ignore the query". High agreement
    (roughly >0.95) means the query term barely moves the routing decision --
    the other half of the P7 collapse diagnosis alongside :func:`effective_rank`.

    ``logit`` is a dense ``[Q, M]`` matrix (same model order as ``b_m``,
    ``[M]``)."""
    logit = np.asarray(logit, dtype=np.float64)
    b_m = np.asarray(b_m, dtype=np.float64)
    if logit.ndim != 2 or logit.shape[0] == 0 or logit.shape[1] != b_m.shape[0]:
        return float("nan")
    bias_only = int(np.argmax(-b_m))
    return float(np.mean(logit.argmax(axis=1) == bias_only))


def marginal_baselines(
    train_targets: Sequence[float],
    train_model_ids: Sequence[str],
    eval_targets: Sequence[float],
    eval_model_ids: Sequence[str],
) -> dict:
    """Reference predictors every NIRT run is measured against.

    * ``global_mean`` -- predict the overall training correctness rate for every
      (query, model).
    * ``model_mean`` -- predict each model's mean training correctness (ignores
      the query). This is the bar the query head has to clear.
    """
    tt = np.asarray(train_targets, dtype=np.float64)
    tm = np.asarray(train_model_ids)
    ev = np.asarray(eval_targets, dtype=np.float64)
    em = np.asarray(eval_model_ids)

    g = float(tt.mean())
    per_model: dict[str, float] = {}
    for mid in np.unique(tm):
        per_model[str(mid)] = float(tt[tm == mid].mean())

    global_pred = np.full_like(ev, g)
    model_pred = np.array([per_model.get(str(m), g) for m in em], dtype=np.float64)
    return {
        "global_mean": {"value": g, **_bce_mse(ev, global_pred)},
        "model_mean": {"per_model": per_model, **_bce_mse(ev, model_pred)},
    }
