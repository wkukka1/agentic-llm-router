"""UMAP -> HDBSCAN clustering of query embeddings.

Deterministic (UMAP with a fixed ``random_state``; HDBSCAN is deterministic given
its input). Clustering runs in UMAP space; the per-cluster **centroids** are the
unit-normalised mean of each cluster's members in the *original* embedding space,
so ``r_q`` for any query needs only its raw embedding and ``centroids.npy``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from router._normalize import unit_rows as _unit  # re-exported for relevance.py
from router.config import Config, section
from router.determinism import seed_everything

CLUSTERS_FILE = "clusters.parquet"
CENTROIDS_FILE = "centroids.npy"
META_FILE = "clustering.json"


@dataclass
class ClusterConfig:
    seed: int = 42
    pathway: str = "retrieval"
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 10
    umap_metric: str = "cosine"
    min_cluster_size: int = 20
    min_samples: Optional[int] = None
    hdbscan_metric: str = "euclidean"
    cluster_selection_method: str = "eom"

    @classmethod
    def from_config(cls, cfg: Config, *, pathway: Optional[str] = None) -> "ClusterConfig":
        c = section(cfg, "clustering")
        umap, hdb = c.get("umap", {}) or {}, c.get("hdbscan", {}) or {}
        ms = hdb.get("min_samples")
        return cls(
            seed=int(cfg.get("seed", 42)),
            pathway=pathway or cfg.get("clustering.pathway", "retrieval"),
            umap_n_neighbors=int(umap.get("n_neighbors", 15)),
            umap_min_dist=float(umap.get("min_dist", 0.1)),
            umap_n_components=int(umap.get("n_components", 10)),
            umap_metric=str(umap.get("metric", "cosine")),
            min_cluster_size=int(hdb.get("min_cluster_size", 20)),
            min_samples=None if ms in (None, "null") else int(ms),
            hdbscan_metric=str(hdb.get("metric", "euclidean")),
            cluster_selection_method=str(hdb.get("cluster_selection_method", "eom")),
        )

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ClusterResult:
    labels: np.ndarray            # (n,) int; 0..C-1, or -1 for noise
    probabilities: np.ndarray     # (n,) HDBSCAN membership strength
    centroids: np.ndarray         # (C, dim) unit-norm, original embedding space
    config: ClusterConfig
    ids: Optional[list] = None
    extra: dict = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def noise_fraction(self) -> float:
        return float(np.mean(self.labels < 0)) if len(self.labels) else 0.0

    def summary(self) -> dict:
        uniq, counts = np.unique(self.labels[self.labels >= 0], return_counts=True)
        return {
            "n_points": int(len(self.labels)),
            "n_clusters": self.n_clusters,
            "noise_fraction": round(self.noise_fraction, 4),
            "cluster_sizes": {int(k): int(v) for k, v in zip(uniq, counts)},
            "min_cluster_size_seen": int(counts.min()) if len(counts) else 0,
            "max_cluster_size_seen": int(counts.max()) if len(counts) else 0,
        }


def cluster_embeddings(embeddings, config: ClusterConfig, *, ids=None) -> ClusterResult:
    """UMAP -> HDBSCAN on ``embeddings`` (``n x dim``)."""
    import hdbscan
    import umap

    seed_everything(config.seed)
    X = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    n = X.shape[0]
    n_neighbors = min(config.umap_n_neighbors, max(2, n - 1))
    n_components = min(config.umap_n_components, max(2, n - 2))
    emb = umap.UMAP(n_neighbors=n_neighbors, min_dist=config.umap_min_dist,
                    n_components=n_components, metric=config.umap_metric,
                    random_state=config.seed, n_jobs=1).fit_transform(X)

    cl = hdbscan.HDBSCAN(min_cluster_size=max(2, config.min_cluster_size),
                         min_samples=config.min_samples, metric=config.hdbscan_metric,
                         cluster_selection_method=config.cluster_selection_method)
    labels = cl.fit_predict(np.ascontiguousarray(emb, dtype=np.float64)).astype(np.int64)
    probs = np.asarray(getattr(cl, "probabilities_", np.ones(n)), dtype=np.float64)

    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    unit = _unit(X.astype(np.float64))
    centroids = _unit(np.stack([unit[labels == c].mean(0) if (labels == c).any()
                                else np.zeros(unit.shape[1]) for c in range(n_clusters)])
                      if n_clusters else np.zeros((0, unit.shape[1]))).astype(np.float32)

    return ClusterResult(labels=labels, probabilities=probs, centroids=centroids, config=config,
                         ids=list(ids) if ids is not None else None,
                         extra={"umap_n_neighbors": n_neighbors, "umap_n_components": n_components})


# --------------------------------------------------------------------------- #
# build / persist                                                             #
# --------------------------------------------------------------------------- #
def _nirt_query_source(cfg: Config) -> str:
    p = cfg.path("processed") / "nirt_observations.parquet"
    return "nirt_observations.parquet" if p.exists() else "queries.parquet"


def _nirt_query_ids(cfg: Config) -> list:
    """Query ids carrying a NIRT correctness observation; falls back to all queries
    (with a warning -- the taxonomy then depends on build order)."""
    import warnings

    import pandas as pd

    col = _nirt_query_source(cfg)
    if col == "queries.parquet":
        warnings.warn(
            "nirt_observations.parquet not built: clustering EVERY query, including "
            "pairwise-only prompts with no correctness label. Build the NIRT "
            "observation table first (scripts/data/build_nirt_dataset.py)."
        )
    return sorted(pd.read_parquet(cfg.path("processed") / col, columns=["query_id"])["query_id"].unique())


def centroids_fingerprint(centroids: np.ndarray) -> str:
    import hashlib

    c = np.ascontiguousarray(np.asarray(centroids, dtype=np.float32))
    return hashlib.sha1(str(c.shape).encode() + c.tobytes()).hexdigest()[:16]


def build_clusters(cfg: Config, *, pathway=None, query_ids=None, config: Optional[ClusterConfig] = None,
                   save: bool = True) -> ClusterResult:
    from router.embeddings import EmbeddingStore, default_store_dir

    config = config or ClusterConfig.from_config(cfg, pathway=pathway)
    store_dir = default_store_dir(cfg, "query", config.pathway)
    if not EmbeddingStore.exists(store_dir):
        raise FileNotFoundError(
            f"query embeddings for pathway '{config.pathway}' not built ({store_dir})")
    store = EmbeddingStore.load(store_dir)
    source = "explicit" if query_ids else _nirt_query_source(cfg)
    ids = [q for q in (query_ids or _nirt_query_ids(cfg)) if q in store]
    if not ids:
        raise ValueError("no query ids present in the embedding store")

    result = cluster_embeddings(store.gather(ids), config, ids=ids)
    if save:
        write_clusters(result, cfg, store_fingerprint=store.manifest.get("ids_fingerprint"),
                       query_source=source)
    return result


def write_clusters(result: ClusterResult, cfg: Config, *, store_fingerprint=None,
                   query_source: Optional[str] = None) -> Path:
    import pandas as pd

    out = cfg.path("taxonomy")
    out.mkdir(parents=True, exist_ok=True)
    ids = result.ids if result.ids is not None else list(range(len(result.labels)))
    pd.DataFrame({"query_id": ids, "cluster_id": result.labels.astype(int),
                  "cluster_prob": result.probabilities.astype(float),
                  "is_noise": result.labels < 0}).to_parquet(out / CLUSTERS_FILE, index=False)
    np.save(out / CENTROIDS_FILE, result.centroids.astype(np.float32))
    (out / META_FILE).write_text(json.dumps({
        "method": "umap_hdbscan", "pathway": result.config.pathway,
        "config": result.config.to_dict(), "query_embeddings_fingerprint": store_fingerprint,
        "query_source": query_source,
        "centroids_fingerprint": centroids_fingerprint(result.centroids),
        "taxonomy_version": int(cfg.get("taxonomy.version", 1)),
        **result.summary(), **result.extra,
    }, indent=2), encoding="utf-8")
    return out / META_FILE


def _load(cfg: Config, name: str, loader):
    p = cfg.path("taxonomy") / name
    if not p.exists():
        raise FileNotFoundError(f"{p} not built; run scripts/taxonomy/cluster_queries.py")
    return loader(p)


def load_clusters(cfg: Config):
    import pandas as pd

    return _load(cfg, CLUSTERS_FILE, pd.read_parquet)


def load_centroids(cfg: Config) -> np.ndarray:
    return _load(cfg, CENTROIDS_FILE, np.load)


def load_meta(cfg: Config) -> dict:
    p = cfg.path("taxonomy") / META_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
