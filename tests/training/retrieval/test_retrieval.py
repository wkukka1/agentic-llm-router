"""FAISS query bank: build / save / load / kNN / self-exclusion."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import make_store
from helpers.faiss_bank import make_faiss_bank as _bank

pytest.importorskip("faiss")

from training.retrieval.query_bank import QueryBank


def test_save_load_roundtrip(tmp_path):
    bank, X, ids = _bank(tmp_path)
    loaded = QueryBank.load(tmp_path)
    assert loaded.ids == ids
    assert len(loaded) == len(ids)
    r = loaded.search(X[:3], k=5)
    assert r.ids.shape == (3, 5)


def test_knn_finds_self_without_exclusion(tmp_path):
    bank, X, ids = _bank(tmp_path)
    r = bank.search(X[10:11], k=1)
    assert r.ids[0, 0] == "q10"


def test_knn_excludes_self(tmp_path):
    bank, X, ids = _bank(tmp_path)
    r = bank.search(X[10:11], k=3, exclude_ids=["q10"])
    assert "q10" not in list(r.ids[0])
    assert sum(1 for x in r.ids[0] if x) == 3


def test_neighbor_mean_shape(tmp_path):
    bank, X, ids = _bank(tmp_path)
    store = make_store(ids, X, "query_id")
    nm = bank.neighbor_mean(X[:4], store, k=5, exclude_ids=ids[:4])
    assert nm.shape == (4, X.shape[1])
    assert np.isfinite(nm).all()
