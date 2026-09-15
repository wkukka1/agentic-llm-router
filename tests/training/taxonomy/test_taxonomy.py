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
