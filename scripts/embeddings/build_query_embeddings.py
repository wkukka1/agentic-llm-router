"""Embed queries (data/processed/queries.parquet) with an encoder pathway.

Streaming + resumable: a killed run leaves a valid partial vectors.npy and
re-running picks up from the last flushed batch.

    python scripts/embeddings/build_query_embeddings.py --pathway irt
    python scripts/embeddings/build_query_embeddings.py --pathway retrieval --split train
    python scripts/embeddings/build_query_embeddings.py --pathway irt --limit 500   # smoke
    python scripts/embeddings/build_query_embeddings.py --pathway irt --restart     # ignore progress

Output: <embedding.cache_dir>/query__<pathway>/{vectors.npy, ids.parquet,
manifest.json}. Deterministic for a fixed config + device.
"""

from __future__ import annotations

import json
import sys

from router.cli import base_parser, get_config, info
from router.embeddings import available_pathways, build_store, default_store_dir, load_encoder


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--pathway", default=None,
                        help="encoder pathway (default: embedding.default_pathway)")
    parser.add_argument("--all-pathways", action="store_true")
    parser.add_argument("--split", default=None, help="restrict to train/validation/test")
    parser.add_argument("--from-nirt", action="store_true",
                        help="only queries present in nirt_observations.parquet "
                             "(the set the NIRT dataset actually needs)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--restart", action="store_true", help="ignore any saved progress")
    parser.add_argument("--flush-every", type=int, default=25, help="batches per checkpoint")
    args = parser.parse_args()
    cfg = get_config(args)

    import pandas as pd

    queries = pd.read_parquet(cfg.path("processed") / "queries.parquet")
    if args.split:
        sp = cfg.path("splits") / f"{args.split}.json"
        ids = set(json.loads(sp.read_text(encoding="utf-8"))["query_ids"])
        queries = queries[queries["query_id"].isin(ids)]
        info(f"restricted to split '{args.split}': {len(queries)} queries")
    if args.from_nirt:
        nirt = cfg.path("processed") / "nirt_observations.parquet"
        if not nirt.exists():
            parser.error("--from-nirt needs nirt_observations.parquet "
                         "(run scripts/data/build_nirt_dataset.py first)")
        keep = set(pd.read_parquet(nirt, columns=["query_id"])["query_id"])
        queries = queries[queries["query_id"].isin(keep)]
        info(f"restricted to NIRT queries: {len(queries)}")
    if args.limit:
        queries = queries.head(args.limit)
    queries = queries.sort_values("query_id").reset_index(drop=True)

    if args.all_pathways:
        pathways = available_pathways(cfg)
    elif args.pathway:
        pathways = [args.pathway]
    else:
        pathways = [cfg.get("embedding.default_pathway") or available_pathways(cfg)[0]]

    for pw in pathways:
        encoder = load_encoder(cfg, pathway=pw)
        suffix = f"__{args.split}" if args.split else ("__smoke" if args.limit else "")
        out_dir = default_store_dir(cfg, "query", pw + suffix)
        info(f"[{pw}] {encoder.cfg.backend}:{encoder.cfg.model_name} on "
             f"{encoder.device} -> {len(queries)} queries -> {out_dir.name}")
        store = build_store(
            out_dir, queries["query_id"].tolist(), queries["query"].tolist(),
            encoder, id_field="query_id", resume=not args.restart,
            force=args.restart, flush_every=args.flush_every,
            manifest_extra={"query_set": "nirt" if args.from_nirt
                            else (args.split or "all")},
        )
        info(f"[{pw}] done: {len(store)} embeddings (dim={store.dim}) -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
