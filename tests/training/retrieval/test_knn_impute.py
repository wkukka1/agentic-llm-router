"""kNN-imputed query representation: weighted neighbour mean, builder fallback,
leakage-safe bank exclusion, and the query_pathway store swap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")

from router.embeddings.encoder import EmbeddingStore
from router.retrieval.knn_impute import imputed_pathway_name
from router.retrieval.query_bank import QueryBank


def _bank_and_store(tmp_path, n=60, dim=8, seed=0):
    import faiss

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(dim)
    index.add(Xn)
    ids = [f"q{i}" for i in range(n)]
    bank = QueryBank(index, ids, {"split": "train", "pathway": "retrieval",
                                  "index_type": "flat_ip", "normalized": True,
                                  "dim": dim, "count": n, "default_k": 5})
    bank.save(tmp_path)
    # target space: a *different* embedding per id, same id set
    T = rng.standard_normal((n, dim + 3)).astype(np.float32)
    store = EmbeddingStore(ids, T, {"dim": dim + 3, "id_field": "query_id",
                                    "count": n, "complete": True}, id_field="query_id")
    return bank, X, T, store, ids


# ---------------------------------------------------------------- weighted mean
def test_uniform_matches_neighbor_mean(tmp_path):
    bank, X, T, store, ids = _bank_and_store(tmp_path)
    plain = bank.neighbor_mean(X[:5], store, k=4, exclude_ids=ids[:5])
    wtd, found = bank.neighbor_weighted_mean(X[:5], store, k=4, exclude_ids=ids[:5],
                                             weighting="uniform")
    assert (found == 4).all()
    np.testing.assert_allclose(plain, wtd, rtol=1e-5, atol=1e-6)


def test_similarity_weighting_matches_hand_calc(tmp_path):
    bank, X, T, store, ids = _bank_and_store(tmp_path)
    res = bank.search(X[10:11], k=3, exclude_ids=["q10"])
    nbr = list(res.ids[0])
    w = np.clip(res.scores[0], 0, None)
    expect = (store.gather(nbr).astype(np.float64) * (w / w.sum())[:, None]).sum(0)
    got, found = bank.neighbor_weighted_mean(X[10:11], store, k=3, exclude_ids=["q10"],
                                             weighting="similarity")
    assert found[0] == 3
    np.testing.assert_allclose(got[0], expect, rtol=1e-4, atol=1e-5)


def test_k1_returns_single_neighbor(tmp_path):
    bank, X, T, store, ids = _bank_and_store(tmp_path)
    res = bank.search(X[3:4], k=1, exclude_ids=["q3"])
    got, _ = bank.neighbor_weighted_mean(X[3:4], store, k=1, exclude_ids=["q3"])
    np.testing.assert_allclose(got[0], store.get(res.ids[0, 0]), rtol=1e-5, atol=1e-6)


def test_self_exclusion(tmp_path):
    bank, X, T, store, ids = _bank_and_store(tmp_path)
    # without exclusion the nearest neighbour of q7 is q7 itself
    assert bank.search(X[7:8], k=1).ids[0, 0] == "q7"
    got, _ = bank.neighbor_weighted_mean(X[7:8], store, k=1, exclude_ids=["q7"])
    assert not np.allclose(got[0], store.get("q7"))


def test_no_neighbor_returns_zero_and_found_zero(tmp_path):
    bank, X, T, store, ids = _bank_and_store(tmp_path)
    empty = EmbeddingStore([], np.zeros((0, T.shape[1]), np.float32),
                           {"dim": T.shape[1], "id_field": "query_id", "count": 0,
                            "complete": True}, id_field="query_id")
    got, found = bank.neighbor_weighted_mean(X[:3], empty, k=5, exclude_ids=ids[:3])
    assert (found == 0).all()
    assert np.count_nonzero(got) == 0


# ---------------------------------------------------------------- naming
def test_pathway_names():
    assert imputed_pathway_name(10, "similarity") == "knn10w"
    assert imputed_pathway_name(5, "uniform") == "knn5u"
    assert imputed_pathway_name(10, "similarity", ood=True) == "knn10w-ood"


# ---------------------------------------------------------------- bank exclusion
def test_build_query_bank_excludes_ids(tmp_path, monkeypatch):
    import router.embeddings as emb
    import router.retrieval.query_bank as qb

    ids_all = [f"q{i}" for i in range(20)]
    rng = np.random.default_rng(1)
    store = EmbeddingStore(ids_all, rng.standard_normal((20, 6)).astype(np.float32),
                           {"dim": 6, "id_field": "query_id", "count": 20,
                            "complete": True}, id_field="query_id")
    monkeypatch.setattr(qb, "_bank_query_ids", lambda cfg, split: ids_all)
    monkeypatch.setattr(emb.EmbeddingStore, "exists", classmethod(lambda cls, d: True))
    monkeypatch.setattr(emb.EmbeddingStore, "load", classmethod(lambda cls, d: store))
    monkeypatch.setattr(emb, "default_store_dir", lambda cfg, kind, pw: tmp_path / f"{kind}__{pw}")

    class _Cfg:
        def get(self, k, d=None):
            return {"retrieval.pathway": "retrieval", "retrieval.index_type": "flat_ip",
                    "retrieval.k": 5, "seed": 42}.get(k, d)

    bank = qb.build_query_bank(_Cfg(), exclude_query_ids=["q0", "q1", "q2"],
                               holdout_families=["math"], out_dir=tmp_path, save=True)
    assert set(bank.ids).isdisjoint({"q0", "q1", "q2"})
    assert len(bank) == 17
    assert bank.manifest["excluded_count"] == 3
    assert bank.manifest["holdout_families"] == ["math"]


# ---------------------------------------------------------------- query_pathway swap
def test_nirt_dataset_distinct_query_and_profile_stores():
    from router.data.nirt import NIRTDataset

    obs = pd.DataFrame({
        "query_id": ["a", "b", "a"], "model_id": ["m1", "m1", "m2"],
        "target": [1.0, 0.0, 0.5], "cost": [0.1, 0.1, 0.2],
        "metric_type": ["accuracy"] * 3, "source": ["routerbench"] * 3,
    })
    q_store = EmbeddingStore(["a", "b"], np.ones((2, 11), np.float32),
                             {"dim": 11, "id_field": "query_id", "count": 2,
                              "complete": True}, id_field="query_id")
    p_store = EmbeddingStore(["m1", "m2"], np.ones((2, 5), np.float32),
                             {"dim": 5, "id_field": "model_id", "count": 2,
                              "complete": True}, id_field="model_id")
    ds = NIRTDataset(obs, q_store, p_store)
    assert ds.query_dim == 11 and ds.model_dim == 5
