"""A tiny in-memory FAISS bank over a router's TRAINED models' profile
embeddings, for kNN-blending an ad hoc/unknown model's profile embedding
toward its nearest real neighbours.

This is the model-side twin of :mod:`training.retrieval.query_bank` /
:mod:`training.retrieval.knn_impute`, but the model pool is small enough
(tens of rows, not millions) that there is no on-disk persistence, no
train/val split, and no leakage bookkeeping to do -- a bank is just "the ids
a given router actually trained on", built fresh per call.

See that module's docstring for the caveat this shares: smoothing a
representation toward its nearest neighbours barely moved the query-side
accuracy and hurt OOD calibration (docs/nirt_model.md). Treat this as an
explicit opt-in a caller reaches for, not a default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from router._normalize import unit_rows as _unit
from .query_bank import QueryBank

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from router.embeddings.encoder import EmbeddingStore


def build_model_bank(store: "EmbeddingStore", model_ids: Sequence[str], *,
                     index_type: str = "flat_ip") -> QueryBank:
    """A :class:`~training.retrieval.query_bank.QueryBank` over ``store``'s
    rows for ``model_ids`` -- normally a router's own trained pool (models
    with real training observations), so an ad hoc model is only ever blended
    toward profiles the response model actually learned to use."""
    import faiss

    ids = [m for m in dict.fromkeys(str(m) for m in model_ids) if m in store]
    if not ids:
        raise ValueError("no requested model ids present in the profile embedding store")
    X = np.ascontiguousarray(store.gather(ids), dtype=np.float32)

    if index_type == "flat_ip":
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(_unit(X).astype(np.float32))
        normalized = True
    elif index_type == "flat_l2":
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
        normalized = False
    else:
        raise ValueError(f"unknown index_type {index_type!r}; use flat_ip | flat_l2")

    manifest = {
        "pathway": store.manifest.get("pathway"), "index_type": index_type,
        "normalized": normalized, "dim": int(X.shape[1]), "count": len(ids),
    }
    return QueryBank(index, ids, manifest)


def knn_impute_model(bank: QueryBank, raw_vec: np.ndarray, store: "EmbeddingStore", *,
                     k: int = 10, weighting: str = "similarity",
                     exclude_ids: Optional[Sequence[str]] = None) -> tuple[np.ndarray, int]:
    """Blend ``raw_vec`` (one ad hoc model's rendered profile embedding)
    toward its ``k`` nearest neighbours in ``bank`` (profile-embedding cosine
    similarity), reading their own vectors from ``store``.

    Returns ``(blended_vec, n_neighbours_found)`` -- 0 found (an empty or
    fully-excluded bank) means the caller should fall back to ``raw_vec``
    itself, exactly like :func:`training.retrieval.knn_impute.build_knn_imputed_store`
    does for queries.
    """
    vecs, found = bank.neighbor_weighted_mean(
        np.asarray(raw_vec, dtype=np.float32)[None, :], store, k=k,
        exclude_ids=exclude_ids, weighting=weighting,
    )
    return vecs[0], int(found[0])
