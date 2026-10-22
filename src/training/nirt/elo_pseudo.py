"""Elo -> quality pseudo-labels for pretraining NIRT's profile manifold.

The real correctness data covers ~28 models on ~110k queries. Arena Elo covers
most of those models, per category, and the query store holds ~270k encoded
prompts (correctness + Chatbot Arena + GPT-4-Judge). This module turns the two
into synthetic ``(query, model, p)`` rows:

    p(q, m) = sigmoid(alpha_c + beta_c * elo_std(m, c)),   c = category(q)

``(alpha_c, beta_c)`` per task family come from regressing each model's
**train-split** mean score (on the logit scale) on its standardised Arena Elo in
the family's matching category -- the go/no-go regression of
``scripts/nirt/elo_diagnostic.py``. A query's family is its dataset's family
when it has one, else the nearest family centroid in embedding space.

Pretraining NIRT on these rows (``scripts/nirt/profile_manifold.py``) lays the
model-side manifold out in Elo order; fine-tuning on the real data then
corrects it. Models with no Arena rating get no rows. Pure functions, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

# task family -> the Arena category that matches it best
FAMILY_CATEGORY = {"math": "math", "code": "coding", "knowledge": "full",
                   "reasoning": "hard_6", "language_zh": "chinese", "chat": "full",
                   "other": "full"}

__all__ = [
    "FAMILY_CATEGORY",
    "pool_category_elo",
    "dataset_families",
    "fit_elo_quality_map",
    "assign_families",
    "pseudo_observations",
]


def pool_category_elo(manifold_cfg: Mapping, elo_rows: pd.DataFrame) -> pd.DataFrame:
    """``model_id x category`` Arena rating for every pool id the manifold config
    maps to an Arena key (``models`` and ``extra_models``; ``alias_of`` resolved).
    Unrated models are absent."""
    entries = {**(manifold_cfg.get("models") or {}), **(manifold_cfg.get("extra_models") or {})}
    table = elo_rows.pivot_table(index="arena_key", columns="category", values="rating")
    rows = {}
    for mid, entry in entries.items():
        entry = entry or {}
        if entry.get("alias_of"):
            entry = entries.get(entry["alias_of"]) or {}
        key = entry.get("arena_key")
        if key in table.index:
            rows[str(mid)] = table.loc[key]
    return pd.DataFrame(rows).T.sort_index()


def dataset_families(task_families: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """dataset name -> family, from the phase-0 ``profiles.task_families``."""
    return {str(ds): str(fam) for fam, dss in task_families.items() for ds in dss}


def _logit(p: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_elo_quality_map(perf: pd.DataFrame, elo: pd.DataFrame, *,
                        min_models: int = 4) -> dict[str, dict]:
    """Per family, ``logit(mean score) ~ alpha + beta * z`` where ``z`` is the
    family category's Elo standardised over the rated models.

    ``perf``: ``model_id x family`` mean train score (NaN where unobserved).
    A family with fewer than ``min_models`` rated, observed models falls back to
    the ``other`` fit (pooled over every family, ``full`` Elo).
    Returns ``family -> {category, alpha, beta, mu, sd, n, r2}``.
    """
    def one(y: pd.Series, cat: str) -> dict | None:
        x = elo[cat].reindex(y.index) if cat in elo else pd.Series(np.nan, index=y.index)
        ok = y.notna() & x.notna()
        if ok.sum() < min_models:
            return None
        xs, ys = x[ok].to_numpy(float), _logit(y[ok].to_numpy(float))
        mu, sd = float(xs.mean()), float(xs.std()) or 1.0
        z = (xs - mu) / sd
        beta, alpha = np.polyfit(z, ys, 1)
        resid = ys - (alpha + beta * z)
        r2 = 1.0 - float(resid.var() / ys.var()) if ys.var() > 0 else float("nan")
        return {"category": cat, "alpha": float(alpha), "beta": float(beta),
                "mu": mu, "sd": sd, "n": int(ok.sum()), "r2": r2}

    pooled = perf.mean(axis=1, skipna=True)
    out = {"other": one(pooled, "full")}
    if out["other"] is None:
        raise ValueError("fit_elo_quality_map: too few rated models with train scores")
    for fam in perf.columns:
        fit = one(perf[fam], FAMILY_CATEGORY.get(fam, "full"))
        out[str(fam)] = fit if fit is not None else {**out["other"], "fallback": True}
    return out


def assign_families(E: np.ndarray, E_ref: np.ndarray, fam_ref: Sequence[str]) -> np.ndarray:
    """Nearest family centroid (cosine) for each row of ``E``, centroids taken
    over the labelled rows ``E_ref`` / ``fam_ref``."""
    fam_ref = np.asarray(fam_ref).astype(str)
    fams = np.unique(fam_ref)

    def unit(x):
        x = np.asarray(x, dtype=np.float32)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    Ur = unit(E_ref)
    C = unit(np.stack([Ur[fam_ref == f].mean(0) for f in fams]))
    out = np.empty(len(E), dtype=object)
    for s in range(0, len(E), 65536):          # bounded memory on 270k rows
        out[s : s + 65536] = fams[(unit(E[s : s + 65536]) @ C.T).argmax(1)]
    return out.astype(str)


def pseudo_observations(query_ids: Sequence[str], query_family: Sequence[str],
                        elo: pd.DataFrame, fmap: Mapping[str, dict], *,
                        models_per_query: int | None = None,
                        seed: int = 0) -> pd.DataFrame:
    """Synthetic ``(query_id, model_id, target)`` rows, ``target = p(q, m)``.

    Every rated model of ``elo`` is a candidate; ``models_per_query`` samples that
    many per query (all when ``None``) so a fixed row budget covers more queries.
    Columns match ``nirt_observations`` (``metric_type='pseudo_elo'``,
    ``source='elo_pseudo'``, ``cost`` NaN) so ``NIRTDataset`` takes it as is."""
    rng = np.random.default_rng(seed)
    qids = np.asarray(query_ids).astype(str)
    fam = np.asarray(query_family).astype(str)
    models = np.asarray(elo.index).astype(str)
    # p per (family, model): one table, then gathered per row
    fams = sorted(set(fam))
    P = np.empty((len(fams), len(models)), dtype=np.float32)
    for i, f in enumerate(fams):
        fit = fmap.get(f) or fmap["other"]
        x = elo[fit["category"]] if fit["category"] in elo else elo["full"]
        x = x.fillna(elo["full"]).to_numpy(float)          # missing category -> overall Elo
        z = (x - fit["mu"]) / fit["sd"]
        P[i] = 1.0 / (1.0 + np.exp(-(fit["alpha"] + fit["beta"] * z)))
    fi = np.searchsorted(np.asarray(fams), fam)

    k = len(models) if not models_per_query else min(int(models_per_query), len(models))
    if k == len(models):
        mj = np.tile(np.arange(len(models)), len(qids))
    else:
        mj = np.argsort(rng.random((len(qids), len(models))), axis=1)[:, :k].ravel()
    qi = np.repeat(np.arange(len(qids)), k)
    return pd.DataFrame({
        "query_id": qids[qi],
        "model_id": models[mj],
        "target": P[fi[qi], mj],
        "family": fam[qi],
        "metric_type": "pseudo_elo",
        "source": "elo_pseudo",
        "cost": np.nan,
    })
