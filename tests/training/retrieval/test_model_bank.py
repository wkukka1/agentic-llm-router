"""``training.retrieval.model_bank``: the model-side kNN-impute bank -- row
selection/normalization (``build_model_bank``) and the weighted blend +
zero-neighbour fallback (``knn_impute_model``)."""

from __future__ import annotations

import numpy as np
import pytest
from helpers.model_bank import make_model_bank
from helpers.stores import make_store

pytest.importorskip("faiss")

from training.retrieval.model_bank import build_model_bank, knn_impute_model


def test_build_model_bank_restricts_to_requested_ids():
    bank, store, ids = make_model_bank(n=8)
    sub = ids[:3]
    small_bank = build_model_bank(store, sub)
    assert set(small_bank.ids) == set(sub)
    assert small_bank.manifest["normalized"] is True


def test_build_model_bank_drops_ids_absent_from_store():
    bank, store, ids = make_model_bank(n=5)
    small_bank = build_model_bank(store, ids[:2] + ["not-in-store"])
    assert set(small_bank.ids) == set(ids[:2])


def test_build_model_bank_raises_when_nothing_matches():
    bank, store, ids = make_model_bank(n=5)
    with pytest.raises(ValueError, match="no requested model ids"):
        build_model_bank(store, ["nope-1", "nope-2"])


def test_knn_impute_model_matches_hand_calc():
    bank, store, ids = make_model_bank(n=20, dim=6, seed=1)
    raw = store.get(ids[0]) + 0.01 * np.random.default_rng(2).standard_normal(6).astype(np.float32)

    blended, found = knn_impute_model(bank, raw, store, k=3, weighting="similarity")
    assert found == 3

    from training.retrieval.query_bank import QueryBank

    res = bank.search(raw[None, :], k=3)
    w = np.clip(res.scores[0], 0, None)
    expect = (store.gather(list(res.ids[0])).astype(np.float64) * (w / w.sum())[:, None]).sum(0)
    np.testing.assert_allclose(blended, expect, rtol=1e-4, atol=1e-5)


def test_knn_impute_model_zero_neighbors():
    bank, store, ids = make_model_bank(n=5, dim=4)
    empty_store = make_store([], np.zeros((0, 4), np.float32), "model_id")
    empty_bank = build_model_bank(store, ids)  # bank is fine, just search an empty target store
    blended, found = knn_impute_model(empty_bank, store.get(ids[0]), empty_store, k=3)
    assert found == 0
    assert np.count_nonzero(blended) == 0
