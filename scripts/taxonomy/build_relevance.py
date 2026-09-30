"""Build query relevance vectors r_q from the cluster centroids.

    python scripts/taxonomy/build_relevance.py
    python scripts/taxonomy/build_relevance.py --tau 0.05

Writes data/processed/embeddings/query_relevance__<pathway>/ (an EmbeddingStore,
(n, C) rows summing to 1). Requires cluster_queries.py to have run first.

    # r_q from the held-out-family taxonomy (cluster_queries.py --exclude-ood-families)
    python scripts/taxonomy/build_relevance.py --ood

reads <taxonomy>__ood/ and writes query_relevance__<pathway>__ood/. r_q is still
built for every query; only the centroids differ.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from router.config import load_config
from training.taxonomy.clustering import ood_taxonomy_dir
from training.taxonomy.relevance import DEFAULT_TAU, build_relevance, ood_relevance_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--pathway", default=None)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU, help="softmax temperature on cosine sims")
    ap.add_argument("--ood", action="store_true",
                    help="use the held-out-family taxonomy and write the __ood store")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from training.taxonomy.relevance import _pathway, relevance_store_dir

    kw = {}
    if args.ood:
        kw = {"taxonomy_dir": ood_taxonomy_dir(cfg),
              "out_dir": ood_relevance_dir(cfg, _pathway(cfg, args.pathway))}
    rel = build_relevance(cfg, pathway=args.pathway, tau=args.tau, **kw)
    R = np.asarray(rel.matrix)
    pw = rel.manifest.get("pathway")
    print(f"[relevance] {len(rel):,} queries x {rel.dim} clusters, tau={args.tau}")
    print(f"[relevance] mean row entropy {(-(R * np.log(R + 1e-12)).sum(1)).mean():.3f} "
          f"(max {np.log(rel.dim):.3f}); mean max-weight {R.max(1).mean():.3f}")
    print(f"[relevance] wrote {kw.get('out_dir') or relevance_store_dir(cfg, pw)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
