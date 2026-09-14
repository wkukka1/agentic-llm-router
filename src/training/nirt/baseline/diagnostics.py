"""Phase 1 IRT diagnostics.

* :func:`theta_spectrum` -- ``SVD(Theta)``: singular values, explained variance,
  effective rank, collapse flag. A sanity check, not a proof of correctness.
* :func:`icc_curve` -- ``P(correct | theta)`` sweeping one latent dimension.
* :func:`parameter_summary` -- distributions of ``theta_m`` / ``a_q`` / ``b_q``
  and pathological-value flags.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# plotting helpers (shared with calibration.py)                               #
# --------------------------------------------------------------------------- #
def pyplot():
    """``matplotlib.pyplot`` with the Agg backend, or ``None`` if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:  # pragma: no cover
        return None


def savefig(fig, path: str | Path, plt) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def _write_json(obj: dict, directory: str | Path, name: str) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# theta spectrum                                                              #
# --------------------------------------------------------------------------- #
def effective_rank(singular_values, eps: float = 1e-12) -> float:
    """``exp(H(p))`` where ``p`` = normalised singular values (Roy & Vetterli)."""
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def theta_spectrum(theta: np.ndarray) -> dict:
    T = np.asarray(theta, dtype=np.float64)
    sv = np.linalg.svd(T - T.mean(axis=0, keepdims=True), compute_uv=False)
    var = sv ** 2
    return {
        "shape": list(T.shape),
        "singular_values": sv.tolist(),
        "explained_variance_ratio": (var / var.sum()).tolist() if var.sum() else [0.0] * len(sv),
        "effective_rank": effective_rank(sv),
        "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float("inf"),
        "theta_mean_norm": float(np.linalg.norm(T.mean(axis=0))),
        "theta_row_norm_mean": float(np.linalg.norm(T, axis=1).mean()),
        "collapsed": bool(len(sv) > 1 and var[0] / var.sum() > 0.98),
    }


def save_theta_spectrum(spec: dict, directory, name: str = "theta_spectrum.json") -> Path:
    return _write_json(spec, directory, name)


def plot_theta_spectrum(spec: dict, path) -> Optional[Path]:
    plt = pyplot()
    if plt is None:
        return None
    sv = np.asarray(spec["singular_values"])
    x = np.arange(1, len(sv) + 1)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 4))
    ax0.bar(x, sv, color="#1f77b4")
    ax0.set(xlabel="component", ylabel="singular value",
            title=f"Theta spectrum (eff. rank {spec['effective_rank']:.2f} / {len(sv)})")
    ax1.plot(x, np.cumsum(spec["explained_variance_ratio"]), "o-", color="#ff7f0e")
    ax1.set(xlabel="component", ylabel="cumulative explained variance", ylim=(0, 1.02))
    ax1.axhline(0.9, ls="--", color="gray")
    return savefig(fig, path, plt)


# --------------------------------------------------------------------------- #
# ICC                                                                         #
# --------------------------------------------------------------------------- #
def icc_curve(a_q, b_q: float, *, dim_index: int = 0, theta_hold=None, sweep=None) -> dict:
    """``P(correct | theta)`` sweeping latent dimension ``dim_index`` (others pinned
    at ``theta_hold``, default 0; ``sweep`` defaults to [-4, 4])."""
    a = np.asarray(a_q, dtype=np.float64).ravel()
    hold = np.zeros(a.size) if theta_hold is None else np.asarray(theta_hold, np.float64).ravel()
    sweep = np.linspace(-4, 4, 81) if sweep is None else np.asarray(sweep, np.float64)
    theta = np.tile(hold, (len(sweep), 1))
    theta[:, dim_index] = sweep
    p = 1.0 / (1.0 + np.exp(-(theta @ a - float(b_q))))
    dp = np.diff(p)
    return {
        "dim_index": int(dim_index),
        "theta": sweep.tolist(),
        "p_correct": p.tolist(),
        "a_q": a.tolist(),
        "b_q": float(b_q),
        "discrimination_dim": float(a[dim_index]),
        "monotone_increasing": bool(np.all(dp >= -1e-9) if a[dim_index] >= 0 else np.all(dp <= 1e-9)),
        "p_range": [float(p.min()), float(p.max())],
    }


def plot_icc(curves: dict, path, title: str = "Item characteristic curves") -> Optional[Path]:
    plt = pyplot()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, c in curves.items():
        ax.plot(c["theta"], c["p_correct"],
                label=f"{label} (a={c['discrimination_dim']:.2f}, b={c['b_q']:.2f})")
    ax.set(xlabel=r"model ability $\theta$ (swept dim)",
           ylabel=r"$P(\mathrm{correct}\mid\theta)$", ylim=(-0.02, 1.02), title=title)
    ax.legend(fontsize=8)
    return savefig(fig, path, plt)


# --------------------------------------------------------------------------- #
# parameter summary                                                           #
# --------------------------------------------------------------------------- #
def _dist(x) -> dict:
    x = np.asarray(x, dtype=np.float64).ravel()
    q = np.percentile(x, [5, 50, 95])
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "p05": float(q[0]), "median": float(q[1]),
            "p95": float(q[2]), "max": float(x.max()), "n_nonfinite": int((~np.isfinite(x)).sum())}


def parameter_summary(*, theta, a_q, b_q, discrimination_constrained: bool = True) -> dict:
    theta, a_q, b_q = np.asarray(theta), np.asarray(a_q), np.asarray(b_q)
    theta_norm = np.linalg.norm(theta, axis=1)
    a_norm = np.linalg.norm(a_q, axis=1)
    flags = []
    if discrimination_constrained and np.any(a_q < -1e-6):
        flags.append("negative discrimination despite softplus constraint")
    if not (np.isfinite(theta).all() and np.isfinite(a_q).all() and np.isfinite(b_q).all()):
        flags.append("non-finite parameters")
    if a_norm.std() < 1e-6:
        flags.append("all discrimination vectors near-identical")
    if theta_norm.max() > 50:
        flags.append("exploding theta norm")
    return {
        "theta": _dist(theta), "theta_row_norm": _dist(theta_norm),
        "discrimination": _dist(a_q), "discrimination_row_norm": _dist(a_norm),
        "difficulty": _dist(b_q), "pathology_flags": flags,
    }
