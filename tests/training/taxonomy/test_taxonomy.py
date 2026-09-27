"""Query ability taxonomy: clustering determinism + relevance-vector properties."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.taxonomy.clustering import ClusterConfig, _nirt_query_ids, cluster_embeddings
from training.taxonomy.relevance import relevance_from_embeddings


def _blobs(n_per=80, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[3, 0, 0], [-3, 0, 0], [0, 3, 0], [0, -3, 0]], dtype=float)
    X = np.vstack([c + 0.25 * rng.standard_normal((n_per, 3)) for c in centers])
    return X.astype(np.float32), centers


def test_nirt_query_ids_builds_instead_of_widening(monkeypatch):
    """XD-05 / XA-11: on a missing nirt_observations.parquet, _nirt_query_ids
    must use the built (correctness-labeled) query set via the shared
    training.data.nirt.observations(build=True) accessor, not silently widen
    to every query in queries.parquet."""
    import training.data.nirt as nirt_data

    obs = pd.DataFrame({"query_id": ["q0", "q1"]})
    calls = []

    def fake_observations(cfg, *, data=None, build=False, **kw):
        calls.append(build)
        return obs

    monkeypatch.setattr(nirt_data, "observations", fake_observations)

    ids = _nirt_query_ids(object())
    assert ids == ["q0", "q1"]
    assert calls == [True]


def test_cluster_embeddings_recovers_blobs_and_is_deterministic():
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    X, _ = _blobs()
    cc = ClusterConfig(seed=42, umap_n_components=2, umap_n_neighbors=15,
                       min_cluster_size=20, min_samples=5)
    r1 = cluster_embeddings(X, cc)
    r2 = cluster_embeddings(X, cc)
    np.testing.assert_array_equal(r1.labels, r2.labels)
    assert 2 <= r1.n_clusters <= 6
    assert r1.centroids.shape[1] == 3
    np.testing.assert_allclose(np.linalg.norm(r1.centroids, axis=1), 1.0, atol=1e-4)


def test_relevance_from_embeddings_is_a_distribution():
    rng = np.random.default_rng(0)
    centroids = rng.standard_normal((5, 8))
    E = rng.standard_normal((20, 8))
    R = relevance_from_embeddings(E, centroids, tau=0.1)
    assert R.shape == (20, 5)
    np.testing.assert_allclose(R.sum(axis=1), 1.0, atol=1e-5)
    assert (R >= 0).all()


def test_relevance_peaks_at_nearest_centroid():
    centroids = np.eye(4)
    E = np.array([[10.0, 0, 0, 0], [0, 0, 9.0, 0]])
    R = relevance_from_embeddings(E, centroids, tau=0.05)
    assert R[0].argmax() == 0
    assert R[1].argmax() == 2


def test_relevance_temperature_controls_peakiness():
    rng = np.random.default_rng(1)
    centroids = rng.standard_normal((6, 10))
    E = rng.standard_normal((30, 10))
    hot = relevance_from_embeddings(E, centroids, tau=1.0).max(axis=1).mean()
    cold = relevance_from_embeddings(E, centroids, tau=0.02).max(axis=1).mean()
    assert cold > hot


# --------------------------------------------------------------------------- #
# The clusters (and so r_q) must be fit on the TRAIN split only               #
# (known_bugs/_module_level/training.taxonomy/MOD-001)                        #
# --------------------------------------------------------------------------- #
def _split_world(tmp_path, monkeypatch):
    """Three train blobs, plus a fourth, distinct blob that only exists in the
    test split and one in the ood split. All embeddings are in the store."""
    from helpers import make_store
    from router.config import Config
    from router.embeddings import default_store_dir
    import training.data.nirt as nirt_data

    rng = np.random.default_rng(0)
    axes = np.eye(5) * 3.0
    ids, rows, split = [], [], []
    for blob, name in enumerate(["train", "train", "train", "test", "ood"]):
        for i in range(60):
            ids.append(f"{name}{blob}-{i}")
            rows.append(axes[blob] + 0.25 * rng.standard_normal(5))
            split.append(name)
    cfg = Config({"seed": 42,
                  "paths": {"taxonomy": str(tmp_path / "tax")},
                  "embedding": {"cache_dir": str(tmp_path / "emb")},
                  "clustering": {"pathway": "retrieval",
                                 "umap": {"n_components": 2, "n_neighbors": 15},
                                 "hdbscan": {"min_cluster_size": 20, "min_samples": 5}}},
                 root=tmp_path)
    make_store(ids, np.array(rows)).save(default_store_dir(cfg, "query", "retrieval"))
    obs = pd.DataFrame({"query_id": ids, "split": split})
    monkeypatch.setattr(nirt_data, "observations", lambda cfg, *, data=None, build=False, **kw: obs)
    return cfg, ids, split, axes


def test_nirt_query_ids_can_be_limited_to_one_split(monkeypatch):
    import training.data.nirt as nirt_data

    obs = pd.DataFrame({"query_id": ["a", "b", "c"], "split": ["train", "test", "train"]})
    monkeypatch.setattr(nirt_data, "observations", lambda cfg, *, data=None, build=False, **kw: obs)
    assert _nirt_query_ids(object(), split="train") == ["a", "c"]
    assert _nirt_query_ids(object()) == ["a", "b", "c"]


def test_build_clusters_fits_on_train_queries_only(tmp_path, monkeypatch):
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    from training.taxonomy.clustering import build_clusters, load_centroids, load_clusters, load_meta

    cfg, ids, split, axes = _split_world(tmp_path, monkeypatch)
    build_clusters(cfg)

    clustered = set(load_clusters(cfg)["query_id"])
    assert clustered == {q for q, s in zip(ids, split) if s == "train"}
    # no centroid points at the blob that exists only in test / ood
    centroids = load_centroids(cfg)
    assert (centroids @ (axes[3] / 3.0)).max() < 0.5
    assert (centroids @ (axes[4] / 3.0)).max() < 0.5
    assert load_meta(cfg)["query_source"] == "nirt_observations:train"


def test_relevance_is_still_built_for_every_query_not_just_the_clustered_ones(tmp_path, monkeypatch):
    """r_q is a pure function of embedding + centroids, so validation / test / ood
    queries still need one even though only train queries shaped the centroids."""
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    from training.taxonomy.clustering import build_clusters
    from training.taxonomy.relevance import build_relevance

    cfg, ids, split, _ = _split_world(tmp_path, monkeypatch)
    build_clusters(cfg)
    rel = build_relevance(cfg)
    assert set(rel.ids) == set(ids)


# --------------------------------------------------------------------------- #
# A held-out-family variant: exclude queries, write to a separate location    #
# (mirrors build_query_bank(exclude_query_ids=..., out_dir=...))              #
# --------------------------------------------------------------------------- #
def test_build_clusters_can_exclude_query_ids(tmp_path, monkeypatch):
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    from training.taxonomy.clustering import build_clusters, load_centroids, load_clusters, load_meta

    cfg, ids, split, axes = _split_world(tmp_path, monkeypatch)
    held_out = [q for q in ids if q.startswith("train0-")]     # one whole train blob
    build_clusters(cfg, exclude_query_ids=held_out)

    assert not set(held_out) & set(load_clusters(cfg)["query_id"])
    assert (load_centroids(cfg) @ (axes[0] / 3.0)).max() < 0.5
    assert load_meta(cfg)["excluded_count"] == len(held_out)


def test_clusters_and_relevance_can_be_written_to_a_separate_directory(tmp_path, monkeypatch):
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    from training.taxonomy.clustering import build_clusters, load_centroids
    from training.taxonomy.relevance import build_relevance, relevance_store_dir

    cfg, ids, split, _ = _split_world(tmp_path, monkeypatch)
    held_out = [q for q in ids if q.startswith("train0-")]
    tax_dir, rel_dir = tmp_path / "tax_ood", tmp_path / "rel_ood"

    build_clusters(cfg, exclude_query_ids=held_out, out_dir=tax_dir)
    rel = build_relevance(cfg, taxonomy_dir=tax_dir, out_dir=rel_dir)

    assert not (tmp_path / "tax").exists()                       # the main taxonomy is untouched
    assert not relevance_store_dir(cfg, "retrieval").exists()    # ...and so is the main r_q store
    assert rel.dim == load_centroids(cfg, tax_dir).shape[0]
    assert set(rel.ids) == set(ids)                              # r_q still covers every query
    assert (rel_dir / "manifest.json").exists()


def test_ood_variant_locations_sit_beside_the_main_ones(tmp_path):
    from router.config import Config
    from training.taxonomy.clustering import ood_taxonomy_dir
    from training.taxonomy.relevance import ood_relevance_dir, relevance_store_dir

    cfg = Config({"paths": {"taxonomy": str(tmp_path / "tax")},
                  "embedding": {"cache_dir": str(tmp_path / "emb")}}, root=tmp_path)
    assert ood_taxonomy_dir(cfg) == tmp_path / "tax__ood"
    assert ood_relevance_dir(cfg, "retrieval") == tmp_path / "emb" / "query_relevance__retrieval__ood"
    assert ood_relevance_dir(cfg, "retrieval") != relevance_store_dir(cfg, "retrieval")
