"""FAISS ``QueryBank`` builder for the retrieval / knn-impute tests.

``faiss`` is imported lazily so ``import helpers`` stays cheap; call sites still
guard with ``pytest.importorskip("faiss")``.
"""

from __future__ import annotations

import numpy as np


def make_faiss_bank(tmp_path, *, n: int = 200, dim: int = 16, seed: int = 0, save: bool = True):
    """A flat-IP ``QueryBank`` over ``n`` random unit vectors.

    Returns ``(bank, X, ids)`` where ``X`` is the *un-normalised* query matrix
    (callers feed raw rows of it back into ``bank.search`` / ``neighbor_mean``).
    """
    import faiss

    from training.retrieval.query_bank import QueryBank

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(dim)
    index.add(Xn)
    ids = [f"q{i}" for i in range(n)]
    manifest = {"split": "train", "pathway": "retrieval", "index_type": "flat_ip",
                "normalized": True, "dim": dim, "count": n, "default_k": 5}
    bank = QueryBank(index, ids, manifest)
    if save:
        bank.save(tmp_path)
    return bank, X, ids
