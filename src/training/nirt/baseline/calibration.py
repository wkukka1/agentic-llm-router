"""Phase 1 calibration: reliability curve + ECE, optional PNG.

``calibration_report`` -> the JSON written to ``calibration.json``;
``plot_reliability`` -> ``plots/calibration.png`` (reliability diagram + a
confidence histogram).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from training.nirt.metrics import brier_score, log_loss, reliability_curve

from .diagnostics import _write_json, pyplot, savefig


def calibration_report(y_true: Sequence[float], y_prob: Sequence[float], n_bins: int = 15) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    rc = reliability_curve(y_true, y_prob, n_bins=n_bins)
    return {
        "n": int(len(y_true)), "n_bins": n_bins,
        "ece": rc["ece"], "mce": rc["mce"],
        "brier": brier_score(y_true, y_prob), "log_loss": log_loss(y_true, y_prob),
        "mean_prediction": float(y_prob.mean()), "base_rate": float(y_true.mean()),
        "reliability_curve": rc["bins"],
    }


def save_calibration(report: dict, directory, name: str = "calibration.json") -> Path:
    return _write_json(report, directory, name)


def plot_reliability(report: dict, path, title: str = "Phase 1 baseline calibration") -> Optional[Path]:
    plt = pyplot()
    if plt is None:
        return None
    bins = [b for b in report["reliability_curve"] if b["count"]]
    conf = [b["confidence"] for b in bins]
    acc = [b["accuracy"] for b in bins]
    centers = [(b["lo"] + b["hi"]) / 2 for b in bins]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6, 7), height_ratios=[3, 1])
    ax0.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax0.plot(conf, acc, "o-", color="#1f77b4", label="model")
    ax0.set(xlabel="mean predicted P(correct)", ylabel="observed accuracy",
            xlim=(0, 1), ylim=(0, 1),
            title=f"{title}\nECE={report['ece']:.4f}  Brier={report['brier']:.4f}")
    ax0.legend()
    ax1.bar(centers, [b["count"] for b in bins], width=0.9 / max(len(bins), 1), color="#9ecae1")
    ax1.set(xlabel="predicted P(correct)", ylabel="count", xlim=(0, 1))
    return savefig(fig, path, plt)
