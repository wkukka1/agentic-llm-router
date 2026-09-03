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
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    total = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.sum() / total * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


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
