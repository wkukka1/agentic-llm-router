"""Query relevance vectors ``r_q``.

    r_q = softmax( cos(e_q, centroid_c) / tau )     c = 1..C

A distribution over the ``C`` ability clusters ("which dimensions is this query
about?"), a pure function of the frozen embedding + centroids (leakage-free,
computable for an unseen query). Conditions the NIRT discrimination head. Stored
as an :class:`~router.embeddings.EmbeddingStore` (``(n, C)``, id_field
``query_id``) at ``data/processed/embeddings/query_relevance__<pathway>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from router.config import Config
from .clustering import _unit, centroids_fingerprint, load_centroids, load_clusters, load_meta

DEFAULT_TAU = 0.1


def _pathway(cfg: Config, pathway: Optional[str]) -> str:
    return pathway or load_meta(cfg).get("pathway") or cfg.get("clustering.pathway", "retrieval")


def relevance_from_embeddings(embeddings, centroids, *, tau: float = DEFAULT_TAU) -> np.ndarray:
    """``(n, dim)`` embeddings + ``(C, dim)`` centroids -> ``(n, C)`` relevance,
    rows >= 0 summing to 1. ``tau`` = softmax temperature (smaller -> peakier)."""
    sims = _unit(np.asarray(embeddings, np.float64)) @ _unit(np.asarray(centroids, np.float64)).T
    z = sims / max(float(tau), 1e-6)
    w = np.exp(z - z.max(axis=1, keepdims=True))
    return (w / w.sum(axis=1, keepdims=True)).astype(np.float32)


def relevance_store_dir(cfg: Config, pathway: str) -> Path:
    from router.embeddings import default_store_dir

    return default_store_dir(cfg, "query_relevance", pathway)


def build_relevance(cfg: Config, *, pathway: Optional[str] = None, tau: float = DEFAULT_TAU,
                    query_ids: Optional[Sequence[str]] = None, save: bool = True):
    """Build + save ``r_q`` for every clustered NIRT query."""
    from router.embeddings import EmbeddingStore, default_store_dir

    pw = _pathway(cfg, pathway)
    centroids = load_centroids(cfg)
    store = EmbeddingStore.load(default_store_dir(cfg, "query", pw))
    ids = [q for q in (list(query_ids) if query_ids is not None
                       else load_clusters(cfg)["query_id"].tolist()) if q in store]
    R = relevance_from_embeddings(store.gather(ids), centroids, tau=tau)
    manifest = {"kind": "query_relevance", "pathway": pw, "dim": int(centroids.shape[0]),
                "id_field": "query_id", "count": len(ids), "tau": float(tau), "normalized": True,
                "complete": True, "source": "softmax(cos(e_q, centroid)/tau)",
                "centroids_fingerprint": centroids_fingerprint(centroids),
                "taxonomy_version": int(cfg.get("taxonomy.version", 1))}
    rel = EmbeddingStore(ids, R, manifest, id_field="query_id")
    if save:
        rel.save(relevance_store_dir(cfg, pw))
    return rel


def load_relevance(cfg: Config, pathway: Optional[str] = None):
    """The ``r_q`` :class:`EmbeddingStore`, or ``None`` if not built.

    Raises if the store was built from different centroids than the current
    ``clustering`` artifacts (re-clustering without rebuilding relevance would
    otherwise silently re-label every column)."""
    from router.embeddings import EmbeddingStore

    d = relevance_store_dir(cfg, _pathway(cfg, pathway))
    if not EmbeddingStore.exists(d):
        return None
    rel = EmbeddingStore.load(d)
    try:
        centroids = load_centroids(cfg)
    except FileNotFoundError:
        return rel
    stale = centroids.shape[0] != rel.dim
    fp = rel.manifest.get("centroids_fingerprint")
    if fp is not None and fp != centroids_fingerprint(centroids):
        stale = True
    if stale:
        raise ValueError(
            f"{d} was built from different cluster centroids than the current taxonomy "
            f"(dim {rel.dim} vs {centroids.shape[0]} clusters); rebuild it with "
            f"training.taxonomy.relevance.build_relevance / scripts/taxonomy/build_relevance.py"
        )
    return rel
