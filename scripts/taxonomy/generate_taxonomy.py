"""Label the clusters -> data/taxonomy/taxonomy.json (no LLM).

    python scripts/taxonomy/generate_taxonomy.py

Each cluster gets a dominant benchmark family, top TF-IDF terms, top datasets,
and a short label. Requires cluster_queries.py to have run first.
"""

from __future__ import annotations

import argparse
import json
import sys

from router.config import load_config
from training.taxonomy.taxonomy import build_taxonomy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tax = build_taxonomy(cfg)
    for c in tax["clusters"][:25]:
        print(f"  c{c['cluster_id']:<3d} n={c['size']:<5d} {c['label']}")
    print(f"\n[taxonomy] {tax['n_clusters']} clusters -> {cfg.path('taxonomy') / 'taxonomy.json'}")
    print(json.dumps({"version": tax["version"], "n_clusters": tax["n_clusters"],
                      "noise_fraction": tax["noise_fraction"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
