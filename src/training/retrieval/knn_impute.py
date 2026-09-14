"""kNN-imputed query embeddings for the Phase-2 NIRT response model.

Every NIRT query's embedding is *replaced* by the cosine-similarity-weighted mean
of its ``k`` nearest **training** queries (found in the retrieval-pathway FAISS
bank, self-excluded; averaged in the ``irt`` / BERT space), so the query head
:class:`~router.nirt.model.NIRTModel` only ever sees points near the training
manifold. The result is saved as an ordinary query
:class:`~router.embeddings.EmbeddingStore` under a synthetic pathway name
(``knn10w`` etc.), which the NIRT stack loads via ``data.query_pathway``.

This is the Phase-2 twin of :mod:`router.retrieval.warmup` (which does the same
blend as a *side channel* for the Phase-1 ``BaselineNIRT``); here it becomes the
query representation itself.

    from router.retrieval.knn_impute import build_knn_imputed_store
    store = build_knn_imputed_store(cfg, k=10)          # -> query__knn10w/

Leakage: the bank is training-queries-only and self-excluded, so a query is never
imputed from itself. For the OOD evaluation pass the held-out families are also
dropped from the bank (``build_query_bank(exclude_query_ids=...)``); point this
builder at that bank with ``bank_dir=``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from router.config import Config
from router.embeddings import EmbeddingStore, default_store_dir
from .query_bank import QueryBank


_WEIGHT_TAG = {"similarity": "w", "uniform": "u"}


def imputed_pathway_name(k: int, weighting: str = "similarity", *, ood: bool = False) -> str:
    """Synthetic query-store pathway name, e.g. ``knn10w`` / ``knn10w-ood``."""
    if weighting not in _WEIGHT_TAG:
        raise ValueError(f"weighting must be 'similarity' | 'uniform', got {weighting!r}")
    return f"knn{int(k)}{_WEIGHT_TAG[weighting]}{'-ood' if ood else ''}"


def imputed_store_dir(cfg: Config, k: int, weighting: str = "similarity", *, ood: bool = False) -> Path:
    return default_store_dir(cfg, "query", imputed_pathway_name(k, weighting, ood=ood))


def _observation_query_ids(cfg: Config) -> list[str]:
    import pandas as pd

    p = cfg.path("processed") / "nirt_observations.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not built; run scripts/data/build_nirt_dataset.py first"
        )
    return sorted(pd.read_parquet(p, columns=["query_id"])["query_id"].astype(str).unique().tolist())


def build_knn_imputed_store(
    cfg: Config,
    *,
    k: int,
    weighting: str = "similarity",
    bank: Optional[QueryBank] = None,
    bank_dir: Optional[str | Path] = None,
    retrieval_pathway: Optional[str] = None,
    target_pathway: str = "irt",
    query_ids: Optional[Sequence[str]] = None,
    out_pathway: Optional[str] = None,
    ood: bool = False,
    save: bool = True,
    batch_size: int = 4096,
    verbose: bool = True,
) -> EmbeddingStore:
    """Build (and optionally save) a kNN-imputed query embedding store.

    ``k`` nearest **training** queries per query, self-excluded; imputed vector =
    ``weighting``-weighted mean of the neighbours in the ``target_pathway`` space.
    Queries with no neighbour keep their original ``target_pathway`` embedding
    (counted in the manifest as ``fallback_count``).
    """
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    bank = bank or QueryBank.load(bank_dir if bank_dir is not None else cfg)
    retr_pw = retrieval_pathway or bank.manifest.get("pathway", "retrieval")
    retr_store = EmbeddingStore.load(default_store_dir(cfg, "query", retr_pw))
    tgt_store = EmbeddingStore.load(default_store_dir(cfg, "query", target_pathway))

    ids_in = [str(q) for q in (query_ids if query_ids is not None else _observation_query_ids(cfg))]
    ids = [q for q in ids_in if q in retr_store and q in tgt_store]
    if not ids:
        raise ValueError("no requested query ids present in both retrieval and target stores")

    retr_vecs = retr_store.gather(ids)
    out = np.zeros((len(ids), tgt_store.dim), dtype=np.float32)
    fallback = 0
    for i in range(0, len(ids), batch_size):
        chunk = ids[i : i + batch_size]
        vecs, found = bank.neighbor_weighted_mean(
            retr_vecs[i : i + len(chunk)], tgt_store, k=int(k),
            exclude_ids=chunk, weighting=weighting,
        )
        for r, qid in enumerate(chunk):
            if found[r] > 0:
                out[i + r] = vecs[r]
            else:
                out[i + r] = tgt_store.get(qid)
                fallback += 1
        if verbose and (i // batch_size) % 4 == 0:
            print(f"[knn-impute] {min(i + len(chunk), len(ids)):,}/{len(ids):,}")

    out_pathway = out_pathway or imputed_pathway_name(k, weighting, ood=ood)
    manifest = {
        "kind": "query",
        "pathway": out_pathway,
        "dim": int(tgt_store.dim),
        "id_field": "query_id",
        "count": len(ids),
        "normalized": False,
        "complete": True,
        "knn_k": int(k),
        "weighting": weighting,
        "retrieval_pathway": retr_pw,
        "target_pathway": target_pathway,
        "bank_split": bank.manifest.get("split", "train"),
        "bank_count": len(bank),
        "bank_holdout_families": bank.manifest.get("holdout_families"),
        "fallback_count": int(fallback),
        "source": (
            f"{weighting}-weighted mean irt-embedding of the {k} nearest TRAIN "
            f"queries (self-excluded); replaces e_q"
        ),
    }
    store = EmbeddingStore(ids, out, manifest, id_field="query_id")
    if verbose:
        print(f"[knn-impute] {out_pathway}: {len(ids):,} queries x {store.dim} "
              f"({fallback:,} fell back to raw e_q)")
    if save:
        store.save(default_store_dir(cfg, "query", out_pathway))
    return store


def load_or_build_imputed_store(
    cfg: Config, *, k: int, weighting: str = "similarity", ood: bool = False,
    rebuild: bool = False, **build_kw,
) -> EmbeddingStore:
    """Return the imputed store, building it only if missing / stale / ``rebuild``."""
    out_pathway = imputed_pathway_name(k, weighting, ood=ood)
    d = default_store_dir(cfg, "query", out_pathway)
    if not rebuild and EmbeddingStore.exists(d):
        store = EmbeddingStore.load(d)
        m = store.manifest
        if m.get("knn_k") == int(k) and m.get("weighting") == weighting:
            return store
    return build_knn_imputed_store(cfg, k=k, weighting=weighting, ood=ood,
                                   out_pathway=out_pathway, **build_kw)
