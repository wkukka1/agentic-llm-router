"""Pre-compute Phase 1 warm-up representations (mean k-NN train-query embedding).

    python scripts/retrieval/build_warmup.py
    python scripts/retrieval/build_warmup.py --model-pathway irt --k 5

Writes data/processed/embeddings/query_warmup__<pathway>/. Requires the FAISS
query bank (scripts/retrieval/build_query_bank.py).
"""

from __future__ import annotations

import argparse
import json
import sys

from router.config import load_config
from training.retrieval.warmup import build_warmup_representations, warmup_store_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-pathway", default=None, help="embedding space the blend happens in (default: data.pathway/irt)")
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = build_warmup_representations(cfg, model_pathway=args.model_pathway, k=args.k)
    print(f"[warmup] {len(store):,} queries x {store.dim}")
    print(json.dumps(store.manifest, indent=2))
    print(f"[warmup] wrote {warmup_store_dir(cfg, store.manifest['pathway'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
