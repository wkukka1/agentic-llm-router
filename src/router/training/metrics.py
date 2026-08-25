"""Scoring for the domain head.

Beyond accuracy, two families of metric are reported because the router
consumes them directly:

* **Calibration** (ECE, Brier) -- the router thresholds on confidence, so a
  model that is 90% accurate but claims 99% confidence is actively harmful.
* **Selective risk** -- accuracy at a given coverage tells you where to put the
  "escalate to a strong model / ask a follow-up" threshold from the design.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Standard equal-width-bin ECE over the top-1 confidence."""
    if len(confidence) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def top_k_accuracy(proba: np.ndarray, y_true_idx: np.ndarray, k: int) -> float:
    if proba.shape[1] < k:
        return float("nan")
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

    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    metrics["per_class_f1"] = {label: float(score) for label, score in zip(labels, per_class_f1, strict=True)}
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    metrics["labels"] = labels
    return metrics
