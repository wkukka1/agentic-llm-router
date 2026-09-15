"""Build the structured per-query difficulty feature store (capacity workstream).

    python scripts/embeddings/build_query_features.py                # all queries
    python scripts/embeddings/build_query_features.py --from-nirt    # only NIRT queries (fast)

Output: <embedding.cache_dir>/query_features__<name>/{vectors.npy, ids.parquet,
manifest.json} -- see training.data.query_features for what's in it and why
(the RouterBench 0-/5-shot confound this was built to fix).
"""

from __future__ import annotations

import sys

from training.cli import base_parser, get_config, info
from training.data.query_features import build_query_features, feature_names


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--name", default="default", help="store variant name")
    parser.add_argument("--from-nirt", action="store_true",
                        help="only queries present in nirt_observations.parquet")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = get_config(args)

    query_ids = None
    if args.from_nirt:
        import pandas as pd

        nirt = cfg.path("processed") / "nirt_observations.parquet"
        if not nirt.exists():
            parser.error("--from-nirt needs nirt_observations.parquet "
                         "(run scripts/data/build_nirt_dataset.py first)")
        query_ids = pd.read_parquet(nirt, columns=["query_id"])["query_id"].unique().tolist()
        if args.limit:
            query_ids = query_ids[: args.limit]
        info(f"restricted to NIRT queries: {len(query_ids)}")

    store = build_query_features(cfg, query_ids=query_ids, name=args.name)
    info(f"[query_features:{args.name}] wrote {len(store)} rows x {store.dim} dims "
        f"({feature_names(cfg)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
