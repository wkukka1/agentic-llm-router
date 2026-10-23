"""Elo-anchored model manifold: a K-dim coordinate ``z_m`` per LLM from its
public Arena category Elos + benchmark scores (``configs/model_manifold.yaml``).

Model: each standardized feature ``x_mf ~= W_f . z_m`` over the *observed*
cells only (masked alternating least squares -- a factor model with missing
data, the setting of observational scaling laws, arXiv 2405.10938, where 2-3
components carry ~97% of model x benchmark variance).

Axis 0 is pinned to the anchor feature (standardized Arena ``full`` Elo):
``W[anchor] = e_0`` and, for every model Arena rated, ``z_m0 = x_m,anchor``
exactly. Models without an Arena rating get ``z_m0`` imputed from their other
cells. Axes ``1..K-1`` are rotated to orthogonal principal directions of what
the features say *beyond* overall Elo. :func:`project_model` places an unseen
model with the same row solve, so a cold-start model needs only its public
numbers.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def build_feature_matrix(
    manifold_cfg: dict,
    elo_long: pd.DataFrame,
    open_llm_rows: Sequence[dict] = (),
    *,
    drop_overlapping: bool = False,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Raw ``(model_ids, X (M x F, NaN = missing), columns)``.

    Columns are ``elo:<category>`` (anchor first) then ``olb:<benchmark>``.
    ``alias_of`` entries copy their target's row; unknown keys stay NaN."""
    arena = manifold_cfg.get("arena") or {}
    anchor = arena.get("anchor_category", "full")
    cats = [anchor] + [c for c in (arena.get("categories") or []) if c != anchor]
    olb = manifold_cfg.get("open_llm_leaderboard") or {}
    bench = list(olb.get("columns") or [])
    if drop_overlapping:
        bench = [b for b in bench if not (olb.get("overlaps") or {}).get(b)]

    elo = elo_long.pivot_table(index="arena_key", columns="category", values="rating")
    by_repo = {r.get("fullname"): r for r in open_llm_rows}
    models = manifold_cfg.get("models") or {}

    def row_for(entry: dict) -> list[float]:
        key, repo = entry.get("arena_key"), entry.get("hf_repo_id")
        vals = []
        for c in cats:
            ok = key is not None and key in elo.index and c in elo.columns
            vals.append(float(elo.at[key, c]) if ok and pd.notna(elo.at[key, c]) else np.nan)
        r = by_repo.get(repo) if repo else None
        for b in bench:
            v = None if r is None else r.get(b)
            vals.append(float(v) if v is not None else np.nan)
        return vals

    ids = sorted(models)
    rows = []
    for mid in ids:
        entry = models[mid] or {}
        target = entry.get("alias_of")
        if target is not None:
            entry = models.get(target) or {}
        rows.append(row_for(entry))
    columns = [f"elo:{c}" for c in cats] + [f"olb:{b}" for b in bench]
    return ids, np.asarray(rows, dtype=np.float64), columns


def _solve_row(W: np.ndarray, x: np.ndarray, obs: np.ndarray, anchor: int, ridge: float) -> np.ndarray:
    """``z`` for one model from its observed cells (axis 0 pinned when the
    anchor is observed)."""
    k = W.shape[1]
    z = np.zeros(k)
    if obs[anchor]:
        z[0] = x[anchor]
        if k > 1:
            cols = obs.copy()
            cols[anchor] = False
            if cols.any():
                A = W[cols, 1:]
                rhs = x[cols] - W[cols, 0] * z[0]
                z[1:] = np.linalg.solve(A.T @ A + ridge * np.eye(k - 1), A.T @ rhs)
        return z
    if obs.any():
        A = W[obs]
        z = np.linalg.solve(A.T @ A + ridge * np.eye(k), A.T @ x[obs])
    return z


def fit_manifold(
    X: np.ndarray,
    k: int,
    *,
    columns: Optional[Sequence[str]] = None,
    anchor: int = 0,
    ridge: float = 1e-2,
    n_iter: int = 500,
    tol: float = 1e-8,
    min_observed: int = 3,
) -> dict:
    """Fit the anchored factor model to raw ``X`` (NaN = missing).

    Returns ``{"mean", "std", "W" (F x k), "Z" (M x k), "columns", "anchor",
    "ridge", "rmse", "explained"}``. Columns observed for fewer than
    ``min_observed`` models are dropped (they cannot pin a loading)."""
    X = np.asarray(X, dtype=np.float64)
    columns = list(columns) if columns is not None else [f"f{i}" for i in range(X.shape[1])]
    keep = (~np.isnan(X)).sum(0) >= min_observed
    if not keep[anchor]:
        raise ValueError("anchor column has fewer than min_observed values")
    anchor = int(keep[:anchor].sum())
    X, columns = X[:, keep], [c for c, kk in zip(columns, keep) if kk]
    k = int(k)
    if k < 1 or k > X.shape[1]:
        raise ValueError(f"k must be in [1, {X.shape[1]}], got {k}")

    mean = np.nanmean(X, 0)
    std = np.nanstd(X, 0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    obs = ~np.isnan(Xs)
    Xf = np.where(obs, Xs, 0.0)
    M, F = Xs.shape

    # init: axis 0 = anchor (0 where unrated), the rest = PCs of the mean-filled residual
    Z = np.zeros((M, k))
    Z[:, 0] = Xf[:, anchor]
    if k > 1:
        resid = Xf - np.outer(Z[:, 0], Xf.T @ Z[:, 0] / max(Z[:, 0] @ Z[:, 0], 1e-12))
        U, S, _ = np.linalg.svd(resid, full_matrices=False)
        Z[:, 1:] = U[:, : k - 1] * S[: k - 1] / np.sqrt(M)
    W = np.zeros((F, k))
    prev = np.inf
    for _ in range(n_iter):
        for f in range(F):
            if f == anchor:
                W[f] = 0.0
                W[f, 0] = 1.0
                continue
            o = obs[:, f]
            A = Z[o]
            W[f] = np.linalg.solve(A.T @ A + ridge * np.eye(k), A.T @ Xs[o, f])
        for i in range(M):
            Z[i] = _solve_row(W, Xf[i], obs[i], anchor, ridge)
        err = float(np.mean((Xf - Z @ W.T)[obs] ** 2))
        if abs(prev - err) < tol:
            break
        prev = err

    if k > 1:
        # orthogonal rotation of the non-anchor axes onto their principal
        # directions; Z W^T is unchanged
        _, _, Vt = np.linalg.svd(Z[:, 1:], full_matrices=False)
        R = Vt.T
        Z[:, 1:] = Z[:, 1:] @ R
        W[:, 1:] = W[:, 1:] @ R

    recon = Z @ W.T
    sse = float(np.sum((Xf - recon)[obs] ** 2))
    sst = float(np.sum(Xf[obs] ** 2))
    return {
        "mean": mean, "std": std, "W": W, "Z": Z, "columns": columns, "anchor": anchor,
        "ridge": float(ridge), "keep": keep,
        "rmse": float(np.sqrt(sse / max(obs.sum(), 1))),
        "explained": 1.0 - sse / max(sst, 1e-12),
    }


def project_model(x_raw: Sequence[float], manifold: dict) -> np.ndarray:
    """``z`` for one model's raw feature row (same column order as the
    ``build_feature_matrix`` output, NaN = missing) under a fitted manifold."""
    x = np.asarray(x_raw, dtype=np.float64)[manifold["keep"]]
    xs = (x - manifold["mean"]) / manifold["std"]
    obs = ~np.isnan(xs)
    return _solve_row(manifold["W"], np.where(obs, xs, 0.0), obs, manifold["anchor"],
                      manifold["ridge"])


def arena_ratings(cfg, manifold_path: str = "configs/model_manifold.yaml") -> dict[str, float]:
    """Pool id -> pinned Arena overall Elo for every model ``manifold_path`` maps to
    an Arena key. Unrated models are absent, never imputed."""
    import yaml

    from training.data.public_benchmarks import fetch_arena_category_elo

    mcfg = yaml.safe_load(cfg.resolve(manifold_path).read_text(encoding="utf-8"))
    arena = mcfg["arena"]
    elo_path = fetch_arena_category_elo(cfg, snapshot_date=arena["snapshot_date"],
                                        filename=arena["filename"], revision=arena["revision"])
    ids, X, _ = build_feature_matrix(mcfg, pd.read_csv(elo_path))
    return {m: float(x) for m, x in zip(ids, X[:, 0]) if np.isfinite(x)}
