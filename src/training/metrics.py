"""Scoring and calibration.

Beyond accuracy, two families matter because downstream consumers use them
directly:

* **Calibration** (ECE, Brier, temperature scaling) -- a head that is 89%
  accurate but claims 99% confidence is worse than useless to anything that
  thresholds on confidence.
* **Selective risk** -- accuracy at a given coverage tells you where to abstain.

Temperature scaling is monotonic, so it never changes accuracy; it only makes
the probabilities mean what they say.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from router.calibration import _EPSILON, apply_temperature


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Standard equal-width-bin ECE over the top-1 confidence.

    Bins are half-open on the right, ``(lo, hi]``, *except* the first, which is
    closed on both sides so that a confidence of exactly 0.0 lands somewhere.
    With an argmax over a probability simplex that cannot happen -- the top
    class is always at least ``1/n_classes`` -- but this function also gets
    handed scores that are not simplex-derived, and a sample silently belonging
    to no bin is the kind of thing that makes a calibration number quietly
    wrong rather than loudly broken.
    """
    if len(confidence) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (confidence >= lo) if i == 0 else (confidence > lo)
        mask &= confidence <= hi
        if not mask.any():
            continue
        error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def top_k_accuracy(proba: np.ndarray, y_true_idx: np.ndarray, k: int) -> float:
    """Share of rows whose true label is among the k highest-scoring.

    When ``k`` is at least the number of classes every row trivially qualifies,
    so this returns 1.0 rather than NaN. That case is degenerate but it is not
    undefined, and NaN propagates into leaderboards as a blank cell that reads
    like a failed run.
    """
    if proba.shape[1] <= k:
        return 1.0
    topk = np.argsort(-proba, axis=1)[:, :k]
    return float((topk == y_true_idx[:, None]).any(axis=1).mean())


def selective_accuracy(confidence: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Accuracy on the most-confident ``coverage`` fraction of predictions."""
    if len(confidence) == 0:
        return float("nan")
    keep = max(1, int(round(len(confidence) * coverage)))
    order = np.argsort(-confidence)[:keep]
    return float(correct[order].mean())


def measure_latency(predict_fn, texts: list[str], *, n_samples: int = 200, warmup: int = 5) -> dict[str, float]:
    """Single-prompt latency, which is the serving path -- not batched throughput."""
    sample = texts[: min(n_samples, len(texts))]
    for text in sample[:warmup]:
        predict_fn([text])

    timings: list[float] = []
    for text in sample:
        start = time.perf_counter()
        predict_fn([text])
        timings.append((time.perf_counter() - start) * 1000.0)

    arr = np.array(timings)
    return {
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_mean": float(arr.mean()),
    }


def evaluate(
    y_true: list[str],
    proba: np.ndarray,
    labels: list[str],
    *,
    coverages: tuple[float, ...] = (0.5, 0.7, 0.9),
) -> dict[str, Any]:
    """Full metric bundle for one model on one split."""
    index = {label: i for i, label in enumerate(labels)}
    y_true_idx = np.array([index[label] for label in y_true])
    y_pred_idx = proba.argmax(axis=1)
    y_pred = [labels[i] for i in y_pred_idx]

    confidence = proba.max(axis=1)
    correct = (y_pred_idx == y_true_idx).astype(float)

    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top2_accuracy": top_k_accuracy(proba, y_true_idx, 2),
        "top3_accuracy": top_k_accuracy(proba, y_true_idx, 3),
        "log_loss": float(log_loss(y_true_idx, proba, labels=list(range(len(labels))))),
        "ece": expected_calibration_error(confidence, correct),
        "mean_confidence": float(confidence.mean()),
    }
    for coverage in coverages:
        metrics[f"acc@coverage{int(coverage * 100)}"] = selective_accuracy(confidence, correct, coverage)

    # For a binary head the router thresholds a score, so ranking quality
    # matters more than accuracy at the default 0.5 cut.
    if len(labels) == 2:
        positive = proba[:, 1]
        binary_true = (y_true_idx == 1).astype(int)
        if 0 < binary_true.sum() < len(binary_true):
            metrics["roc_auc"] = float(roc_auc_score(binary_true, positive))
            metrics["pr_auc"] = float(average_precision_score(binary_true, positive))

    # Precision and recall separately, not just F1: they fail differently and
    # the router cares which. Low recall on a class means prompts of that kind
    # get routed elsewhere; low precision means other prompts get misrouted
    # *into* it. F1 alone hides which is happening.
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    metrics["per_class"] = {
        label: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(sup),
        }
        for label, p, r, f, sup in zip(labels, precision, recall, f1, support, strict=True)
    }
    metrics["per_class_f1"] = {label: float(score) for label, score in zip(labels, f1, strict=True)}
    metrics["macro_precision"] = float(precision.mean())
    metrics["macro_recall"] = float(recall.mean())
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    metrics["labels"] = labels
    return metrics



def fit_temperature(
    proba: np.ndarray,
    y_true_idx: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> float:
    """Find the temperature minimising validation NLL.

    ``T > 1`` softens an over-confident model; ``T < 1`` sharpens an
    under-confident one.
    """
    if len(y_true_idx) == 0:
        return 1.0

    def nll(temperature: float) -> float:
        scaled = apply_temperature(proba, temperature)
        picked = scaled[np.arange(len(y_true_idx)), y_true_idx]
        return float(-np.log(np.clip(picked, _EPSILON, None)).mean())

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(result.x) if result.success else 1.0
