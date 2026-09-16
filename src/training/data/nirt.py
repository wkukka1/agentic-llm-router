"""Build the NIRT observation table, and the training-side convenience
wrapper around :class:`router.data.nirt.NIRTDataset`.

``router.data.nirt`` owns the *dataset* (joins observations to embedding
stores by id -- serving needs that to score a live request). Everything here
*derives* the observation table from raw training data (correctness views,
splits) or resolves a bare ``Config`` into a loaded :class:`TrainingData` --
neither of which a live router does, so it lives on the training side of the
boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

import pandas as pd

from router.config import Config
from router.data.nirt import NIRTDataset

from . import schemas

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from .facade import TrainingData

NIRT_OBS_COLUMNS = [
    "query_id", "model_id", "target", "score_raw", "metric_type",
    "source", "split", "cost", "is_multiple_choice", "n_choices",
]


# --------------------------------------------------------------------------- #
# observation table                                                           #
# --------------------------------------------------------------------------- #
def build_nirt_observations(
    cfg: Config,
    *,
    data: Optional["TrainingData"] = None,
    score_kind: str = "effective",
    metrics: Optional[Iterable[str]] = None,
    models: str = "warm",
    drop_unsplit: bool = True,
) -> pd.DataFrame:
    """Lightweight ``(query, model, target)`` table for every correctness obs.

    ``target`` follows ``score_kind`` (``effective`` = chance-corrected where
    applicable, else raw). ``split`` is attached from ``data/splits``; rows whose
    query is in no split (only reachable via cold-start models) are dropped when
    ``drop_unsplit``.
    """
    from .facade import load_training_data

    d = data or load_training_data(cfg)
    long = d.correctness(split=None, metrics=metrics, models=models,
                         score_kind=score_kind)

    split_of: dict[str, str] = {}
    for name in ("train", "validation", "test", "ood"):
        entry = d.splits.get(name)
        if entry is None:
            continue
        for qid in entry["query_ids"]:
            split_of[qid] = name

    out = pd.DataFrame({
        "query_id": long["query_id"],
        "model_id": long["model_id"],
        "target": pd.to_numeric(long["score"], errors="coerce"),
        "score_raw": pd.to_numeric(long["score_raw"], errors="coerce"),
        "metric_type": long["metric_type"],
        "is_multiple_choice": long["is_multiple_choice"].astype(bool),
        "n_choices": long["n_choices"].astype("Int64"),
        "split": long["query_id"].map(split_of).astype("string"),
    })

    # cost + source live on the raw response rows, not the correctness view
    corr_rows = d.responses[d.responses["metric_type"].isin(schemas.MetricType.CORRECTNESS)]
    extra = (
        corr_rows.groupby(["query_id", "model_id"], as_index=False)
        .agg(cost=("cost", "mean"), source=("source", "first"))
    )
    out = out.merge(extra, on=["query_id", "model_id"], how="left")

    if drop_unsplit:
        out = out[out["split"].notna()]
    out = out[NIRT_OBS_COLUMNS].reset_index(drop=True)
    return out


def write_nirt_observations(df: pd.DataFrame, cfg: Config) -> Path:
    out = cfg.path("processed") / "nirt_observations.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def read_nirt_observations(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.path("processed") / "nirt_observations.parquet")


def observations(
    cfg: Config,
    *,
    data: Optional["TrainingData"] = None,
    build: bool = False,
    **build_kw,
) -> pd.DataFrame:
    """Read ``nirt_observations.parquet`` if present (and no ``build_kw`` is
    given); else build it in memory (never writes to disk) if ``build`` is
    true -- ``build_kw`` always forces a rebuild regardless of ``build``, same
    as a stale-but-present file being ignored; else raise.

    The one training-side accessor for "read or build the observation table"
    (XD-05 / XA-11) -- every call site used to have its own copy with a
    different fallback (raise / crash / build-without-saving / silently widen
    to the wrong, broader query set). ``router.data.nirt.NIRTDataset.
    from_config`` (the serving-side reader) is deliberately NOT built on this
    -- serving must never build data on the fly.
    """
    path = cfg.path("processed") / "nirt_observations.parquet"
    if path.exists() and not build_kw:
        return read_nirt_observations(cfg)
    if not build and not build_kw:
        raise FileNotFoundError(
            f"{path} not built; run scripts/data/build_nirt_dataset.py, or pass build=True"
        )
    from .facade import load_training_data

    d = data or load_training_data(cfg)
    return build_nirt_observations(cfg, data=d, **build_kw)


# --------------------------------------------------------------------------- #
# dataset construction -- resolve `data` + build-if-missing, then delegate    #
# --------------------------------------------------------------------------- #
def nirt_dataset_from_config(
    cfg: Config,
    *,
    data: Optional["TrainingData"] = None,
    split: Optional[str] = None,
    pathway: str = "irt",
    query_pathway: Optional[str] = None,
    query_features: Optional[str] = None,
    observations: Optional[pd.DataFrame] = None,
    return_ids: bool = False,
    **build_kw,
) -> NIRTDataset:
    """Training-side convenience over :meth:`NIRTDataset.from_config`: resolves
    ``data`` from ``cfg`` if not given, and builds the observation table if
    ``nirt_observations.parquet`` doesn't exist yet, before delegating."""
    from .facade import load_training_data

    d = data or load_training_data(cfg)
    if observations is None:
        # same rule as TrainingData.nirt_observations: explicit build kwargs
        # (score_kind=, models=, ...) always rebuild instead of being ignored
        observations = d.nirt_observations(**build_kw)
    return NIRTDataset.from_config(
        cfg, data=d, split=split, pathway=pathway, query_pathway=query_pathway,
        query_features=query_features, observations=observations,
        return_ids=return_ids,
    )
