"""``EmbeddingStore`` builders for tests."""

from __future__ import annotations

import numpy as np

from router.embeddings.encoder import EmbeddingStore


def make_store(ids, data, id_field: str = "query_id", *, seed: int = 0) -> EmbeddingStore:
    """An ``EmbeddingStore`` over ``ids``.

    ``data`` is either the embedding matrix (an array with ``len(ids)`` rows) or an
    ``int`` giving the width of a fresh seeded standard-normal matrix.
    """
    ids = list(ids)
    if isinstance(data, (int, np.integer)):
        data = np.random.default_rng(seed).standard_normal((len(ids), int(data)))
    mat = np.asarray(data, dtype=np.float32)
    manifest = {"dim": mat.shape[1], "id_field": id_field,
                "count": len(ids), "complete": True}
    return EmbeddingStore(ids, mat, manifest, id_field=id_field)
