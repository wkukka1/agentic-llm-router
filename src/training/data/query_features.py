"""Structured per-query difficulty features.

Capacity-workstream finding (2026-09-04, ``docs/nirt_capacity.md``): NIRT does not
underfit -- trained to convergence it overfits, and RouterBench's worst single
generalisation loss is a **data confound**: 0-shot and 5-shot variants of a
question carry identical text (hence identical frozen `e_q`) but different
labels, and shot count is not a feature. One binary "is this the 5-shot variant"
feature recovered most of a 0-shot-only model's validation BCE (0.602 -> 0.569,
both regimes improve) -- bigger than every architecture / encoder /
regularisation change tried.

This module builds a small, leakage-free, deterministic numeric feature vector
per query -- a pure function of ``queries.parquet`` / ``responses.parquet`` and
the config's family map, no labels -- meant to be concatenated to the frozen
query embedding (see ``data.query_features`` in ``configs/nirt.yaml`` and
:class:`router.data.nirt.NIRTDataset`):

* one-hot coarse benchmark **family** (``profiles.task_families``, fixed order
  from the config so the feature dimension never depends on which queries are
  in a given build);
* ``is_5shot`` -- 1.0 iff the query_id carries RouterBench's ``:5shot`` suffix;
* ``prompt_len`` -- character length / 5000, clipped;
* ``prompt_words`` -- ``log1p(word count) / 8``;
* ``n_choices`` -- multiple-choice option count / 5 (0 for free-form / unknown).

    from router.data.query_features import build_query_features, load_query_features
    build_query_features(cfg)                    # -> query_features__default store
    load_query_features(cfg)                     # -> EmbeddingStore or None
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from router.config import Config
from .families import family_of, task_family_map

FEATURE_KIND = "query_features"
_5SHOT_SUFFIX = ":5shot"


def family_names(cfg: Config) -> list[str]:
    """Fixed, sorted family list from ``profiles.task_families`` -- the one-hot
    order, independent of which queries happen to be in a given build."""
    return sorted(set(task_family_map(cfg).values()))


def feature_names(cfg: Config) -> list[str]:
    """Column names, in the exact order :func:`build_query_features` writes them."""
    return family_names(cfg) + ["is_5shot", "prompt_len", "prompt_words", "n_choices"]


def _n_choices_lookup(cfg: Config) -> dict[str, float]:
    """``query_id -> n_choices`` (first non-null value seen), from the raw
    response rows -- independent of ``nirt_observations.parquet`` so this builder
    only needs the Phase 0 tables."""
    path = cfg.path("processed") / "responses.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["query_id", "n_choices"])
    df = df[df["n_choices"].notna()]
    if df.empty:
        return {}
    return (
        df.drop_duplicates("query_id", keep="first")
        .set_index("query_id")["n_choices"]
        .astype(float)
        .to_dict()
    )


def query_features_store_dir(cfg: Config, name: str = "default") -> Path:
    from router.embeddings import default_store_dir

    return default_store_dir(cfg, FEATURE_KIND, name)


def build_query_features(
    cfg: Config,
    *,
    query_ids: Optional[Sequence[str]] = None,
    name: str = "default",
    save: bool = True,
) -> "EmbeddingStore":  # noqa: F821 - imported lazily below
    """Build (and optionally save) the structured feature store.

    ``query_ids`` restricts the build to a subset (default: every row of
    ``queries.parquet``, matching the other embedding builders)."""
    from router.embeddings import EmbeddingStore

    queries = pd.read_parquet(
        cfg.path("processed") / "queries.parquet", columns=["query_id", "query", "dataset"]
    )
    if query_ids is not None:
        keep = set(map(str, query_ids))
        queries = queries[queries["query_id"].astype(str).isin(keep)]
    queries = queries.sort_values("query_id").reset_index(drop=True)

    fams = family_names(cfg)
    fam_map = task_family_map(cfg)
    fam_index = {f: i for i, f in enumerate(fams)}
    nch = _n_choices_lookup(cfg)

    ids = queries["query_id"].astype(str).tolist()
    dim = len(fams) + 4
    out = np.zeros((len(ids), dim), dtype=np.float32)
    for i, (qid, text, dataset) in enumerate(
        zip(ids, queries["query"].astype(str), queries["dataset"].astype(str))
    ):
        fam = family_of(dataset, fam_map)
        if fam in fam_index:
            out[i, fam_index[fam]] = 1.0
        out[i, len(fams)] = 1.0 if qid.endswith(_5SHOT_SUFFIX) else 0.0
        out[i, len(fams) + 1] = min(len(text) / 5000.0, 5.0)
        out[i, len(fams) + 2] = float(np.log1p(len(text.split()))) / 8.0
        out[i, len(fams) + 3] = float(nch.get(qid, 0.0)) / 5.0

    manifest = {
        "kind": FEATURE_KIND,
        "id_field": "query_id",
        "dim": dim,
        "count": len(ids),
        "complete": True,
        "feature_names": feature_names(cfg),
        "families": fams,
        "source": "deterministic function of queries.parquet/responses.parquet + "
                  "profiles.task_families -- no labels, leakage-free",
    }
    store = EmbeddingStore(ids, out, manifest, id_field="query_id")
    if save:
        store.save(query_features_store_dir(cfg, name))
    return store


def load_query_features(cfg: Config, name: str = "default"):
    """The feature :class:`~router.embeddings.EmbeddingStore`, or ``None`` if not built."""
    from router.embeddings import EmbeddingStore

    d = query_features_store_dir(cfg, name)
    return EmbeddingStore.load(d) if EmbeddingStore.exists(d) else None
