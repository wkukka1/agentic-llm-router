"""Build query relevance vectors r_q from the cluster centroids.

    python scripts/taxonomy/build_relevance.py
    python scripts/taxonomy/build_relevance.py --tau 0.05

Writes data/processed/embeddings/query_relevance__<pathway>/ (an EmbeddingStore,
(n, C) rows summing to 1). Requires cluster_queries.py to have run first.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from router.config import load_config
from training.taxonomy.relevance import DEFAULT_TAU, build_relevance


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--pathway", default=None)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU, help="softmax temperature on cosine sims")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from training.taxonomy.relevance import relevance_store_dir

    rel = build_relevance(cfg, pathway=args.pathway, tau=args.tau)
    R = np.asarray(rel.matrix)
    pw = rel.manifest.get("pathway")
    print(f"[relevance] {len(rel):,} queries x {rel.dim} clusters, tau={args.tau}")
    print(f"[relevance] mean row entropy {(-(R * np.log(R + 1e-12)).sum(1)).mean():.3f} "
          f"(max {np.log(rel.dim):.3f}); mean max-weight {R.max(1).mean():.3f}")
    print(f"[relevance] wrote {relevance_store_dir(cfg, pw)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
