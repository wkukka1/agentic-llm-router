"""UMAP -> HDBSCAN clustering of the NIRT query embeddings.

    python scripts/taxonomy/cluster_queries.py
    python scripts/taxonomy/cluster_queries.py --pathway retrieval --min-cluster-size 30

Writes data/taxonomy/{clusters.parquet, centroids.npy, clustering.json}.
"""

from __future__ import annotations

import argparse
import json
import sys

from router.config import load_config
from router.taxonomy.clustering import ClusterConfig, build_clusters


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--pathway", default=None)
    ap.add_argument("--min-cluster-size", type=int, default=None)
    ap.add_argument("--umap-components", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cc = ClusterConfig.from_config(cfg, pathway=args.pathway)
    if args.min_cluster_size is not None:
        cc.min_cluster_size = args.min_cluster_size
    if args.umap_components is not None:
        cc.umap_n_components = args.umap_components

    result = build_clusters(cfg, config=cc)
    print(json.dumps(result.summary(), indent=2))
    print(f"[cluster] wrote {cfg.path('taxonomy')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
