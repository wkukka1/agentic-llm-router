"""UMAP -> HDBSCAN clustering of the NIRT query embeddings.

    python scripts/taxonomy/cluster_queries.py
    python scripts/taxonomy/cluster_queries.py --pathway retrieval --min-cluster-size 30

Writes data/taxonomy/{clusters.parquet, centroids.npy, clustering.json}. Fits on
TRAIN queries only.

    # leakage-safe taxonomy for an OOD run: also drop the held-out families
    python scripts/taxonomy/cluster_queries.py --exclude-ood-families

writes data/taxonomy__ood/ instead, leaving the main taxonomy untouched (build the
matching r_q with `scripts/taxonomy/build_relevance.py --ood`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from router.config import load_config
from training.taxonomy.clustering import ClusterConfig, build_clusters, ood_taxonomy_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--pathway", default=None)
    ap.add_argument("--min-cluster-size", type=int, default=None)
    ap.add_argument("--umap-components", type=int, default=None)
    ap.add_argument("--exclude-ood-families", action="store_true",
                    help="drop configs/nirt.yaml evaluation.ood_holdout_families from the "
                         "clustering and write to <taxonomy>__ood/")
    ap.add_argument("--nirt-config", default="configs/nirt.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cc = ClusterConfig.from_config(cfg, pathway=args.pathway)
    if args.min_cluster_size is not None:
        cc.min_cluster_size = args.min_cluster_size
    if args.umap_components is not None:
        cc.umap_n_components = args.umap_components

    exclude_ids, out_dir = None, None
    if args.exclude_ood_families:
        from evaluation.nirt.ood import holdout_query_ids
        from training.data.facade import load_training_data

        p = Path(args.nirt_config)
        p = p if p.is_absolute() else Path(cfg.root) / p
        exclude_ids, fams = holdout_query_ids(load_training_data(cfg),
                                              yaml.safe_load(p.read_text(encoding="utf-8")))
        out_dir = ood_taxonomy_dir(cfg)
        print(f"[cluster] excluding {len(exclude_ids):,} queries from families {fams}")

    result = build_clusters(cfg, config=cc, exclude_query_ids=exclude_ids, out_dir=out_dir)
    print(json.dumps(result.summary(), indent=2))
    print(f"[cluster] wrote {out_dir or cfg.path('taxonomy')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
