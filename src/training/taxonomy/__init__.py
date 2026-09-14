"""Query ability taxonomy (deferred Phase 0 stage, built for Phase 1).

Three artefacts, each keyed by ``query_id`` and reproducible from the query
embedding store + ``configs/phase0.yaml``:

1. **clusters** (:mod:`.clustering`) -- UMAP -> HDBSCAN over the NIRT query
   embeddings. ``data/taxonomy/clusters.parquet`` (``query_id, cluster_id,
   cluster_prob, is_noise``), ``data/taxonomy/centroids.npy`` (``C x dim`` unit
   centroids, row ``c`` = cluster ``c``), ``data/taxonomy/clustering.json``.

2. **taxonomy** (:mod:`.taxonomy`) -- a readable label per cluster from the
   dominant benchmark family + top TF-IDF terms (no LLM). ``data/taxonomy/
   taxonomy.json``.

3. **relevance vectors** ``r_q`` (:mod:`.relevance`) -- ``r_q in R^C``, a softmax
   over cosine similarity between the query embedding and each cluster centroid.
   A distribution over ability dimensions; conditions the NIRT item head.
   Stored as an :class:`~router.embeddings.EmbeddingStore`
   (``data/processed/embeddings/query_relevance__<pathway>``).

Nothing here is fit on labels or on the response matrix -- it is a property of
the query text alone, so it is leakage-free across the train/test split and
computable for an unseen query from its embedding + ``centroids.npy``.
"""

from .clustering import (
    ClusterConfig,
    ClusterResult,
    build_clusters,
    cluster_embeddings,
    load_centroids,
    load_clusters,
)
from .relevance import (
    build_relevance,
    load_relevance,
    relevance_from_embeddings,
    relevance_store_dir,
)
from .taxonomy import build_taxonomy, load_taxonomy

__all__ = [
    "ClusterConfig",
    "ClusterResult",
    "cluster_embeddings",
    "build_clusters",
    "load_clusters",
    "load_centroids",
    "build_relevance",
    "load_relevance",
    "relevance_from_embeddings",
    "relevance_store_dir",
    "build_taxonomy",
    "load_taxonomy",
]
