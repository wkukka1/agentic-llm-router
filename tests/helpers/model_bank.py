"""``training.retrieval.model_bank`` builder for tests.

``faiss`` is imported lazily (via ``build_model_bank``) so ``import helpers``
stays cheap; call sites still guard with ``pytest.importorskip("faiss")``.
"""

from __future__ import annotations

import numpy as np

from helpers.stores import make_store


def make_model_bank(*, n: int = 8, dim: int = 6, seed: int = 0, prefix: str = "m"):
    """A small ``model_id -> vector`` :class:`EmbeddingStore` plus the
    :class:`~training.retrieval.model_bank.QueryBank`-backed bank built over
    all of its ids.

    Returns ``(bank, store, ids)``.
    """
    from training.retrieval.model_bank import build_model_bank

    rng = np.random.default_rng(seed)
    ids = [f"{prefix}{i}" for i in range(n)]
    mat = rng.standard_normal((n, dim)).astype(np.float32)
    store = make_store(ids, mat, "model_id")
    bank = build_model_bank(store, ids)
    return bank, store, ids
