"""Out-of-distribution split: hold whole benchmark *families* out of training.

The random content-hash train/val/test split (``router.data.splits``) keeps the
distribution fixed. IRT-Router / RouterBench instead test generalisation to
unseen task types. Here we hold out the ``math`` + ``code`` families
(``grade-school-math`` / ``mbpp`` -- generative reasoning + program synthesis,
structurally unlike the multiple-choice-heavy remainder), train on everything
else, and evaluate only on the held-out families.

    from router.nirt.ood import ood_datasets, ood_matrices
    train_ds, val_ds, ood_ds = ood_datasets(data, holdout=("math", "code"))

Family membership reuses the coarse map in ``configs/phase0.yaml ->
profiles.task_families`` (via ``training.data.families``).
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from training.data.families import family_of, task_family_map

DEFAULT_HOLDOUT = ("math", "code")


def ood_families(nirt_cfg: Optional[dict]) -> tuple[str, ...]:
    ev = (nirt_cfg or {}).get("evaluation", {}) or {}
    return tuple(ev.get("ood_holdout_families", DEFAULT_HOLDOUT))


def family_of_query(data) -> dict[str, str]:
    """``query_id -> coarse family`` (only queries whose dataset maps to one)."""
    fam_map = task_family_map(data.cfg)
    ds = data.responses[["query_id", "dataset"]].drop_duplicates()
    out: dict[str, str] = {}
    for qid, dname in zip(ds["query_id"], ds["dataset"]):
        fam = family_of(dname, fam_map)
        if fam is not None:
            out[str(qid)] = fam
    return out


def split_observations(
    data,
    holdout: Iterable[str] = DEFAULT_HOLDOUT,
    *,
    score_kind: str = "effective",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(train_obs, ood_obs)`` -- NIRT observation frames.

    ``ood_obs`` = every warm-model observation whose query is in a held-out
    family. ``train_obs`` = the rest, restricted to the original train/validation
    splits (so the random test split stays untouched for other experiments).
    """
    from training.data.nirt import build_nirt_observations

    holdout = set(holdout)
    obs = build_nirt_observations(data.cfg, data=data, score_kind=score_kind)
    fam = family_of_query(data)
    obs = obs.assign(family=obs["query_id"].map(fam))

    is_ood = obs["family"].isin(holdout)
    ood_obs = obs[is_ood].reset_index(drop=True)
    train_obs = obs[~is_ood & obs["split"].isin(["train", "validation"])].reset_index(drop=True)
    return train_obs, ood_obs


def ood_datasets(
    data,
    holdout: Iterable[str] = DEFAULT_HOLDOUT,
    *,
    pathway: str = "irt",
    query_pathway: Optional[str] = None,
    score_kind: str = "effective",
):
    """``(train_ds, val_ds, ood_ds)`` :class:`NIRTDataset`s.

    ``val_ds`` is the ``split=="validation"`` slice of the non-held-out data
    (early-stopping signal); ``ood_ds`` is the held-out families.

    ``query_pathway`` (default: ``pathway``) swaps only the query store -- for the
    OOD pass this should be a store built from a leakage-safe (families excluded)
    FAISS bank.
    """
    from router.data.nirt import NIRTDataset

    train_obs, ood_obs = split_observations(data, holdout, score_kind=score_kind)
    q_store = data.query_embeddings(query_pathway or pathway)
    p_store = data.profile_embeddings(pathway)
    if q_store is None or p_store is None:
        raise FileNotFoundError(f"embeddings for pathway '{pathway}' not built")

    tr = train_obs[train_obs["split"] == "train"]
    va = train_obs[train_obs["split"] == "validation"]
    return (
        NIRTDataset(tr, q_store, p_store),
        NIRTDataset(va, q_store, p_store),
        NIRTDataset(ood_obs, q_store, p_store),
    )


def ood_matrices(data, holdout: Iterable[str] = DEFAULT_HOLDOUT, *, score_kind: str = "effective"):
    """Dense ``(true, cost)`` DataFrames [query x model] over the held-out families."""
    from .routing import dense_matrices

    _, ood_obs = split_observations(data, holdout, score_kind=score_kind)
    return dense_matrices(ood_obs)


def family_summary(data, holdout: Iterable[str] = DEFAULT_HOLDOUT) -> pd.DataFrame:
    train_obs, ood_obs = split_observations(data, holdout)
    return pd.DataFrame({
        "train": train_obs.groupby("family").size(),
        "ood": ood_obs.groupby("family").size(),
    }).fillna(0).astype(int)
