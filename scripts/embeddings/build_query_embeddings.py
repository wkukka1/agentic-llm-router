"""Embed queries (data/processed/queries.parquet) with an encoder pathway.

Streaming + resumable: a killed run leaves a valid partial vectors.npy and
re-running picks up from the last flushed batch.

    python scripts/embeddings/build_query_embeddings.py --pathway irt
    python scripts/embeddings/build_query_embeddings.py --pathway retrieval --split train
    python scripts/embeddings/build_query_embeddings.py --pathway irt --limit 500   # smoke
    python scripts/embeddings/build_query_embeddings.py --pathway irt --restart     # ignore progress

Output: <embedding.cache_dir>/query__<pathway>/{vectors.npy, ids.parquet,
manifest.json}. Deterministic for a fixed config + device.

The actual build logic is :func:`training.data.embeddings.build_query_embedding_store`,
reachable from Python via ``TrainingData.query_embeddings(pathway, build=True, ...)``
-- this script is a thin CLI wrapper over it.
"""

from __future__ import annotations

import sys

from training.cli import base_parser, get_config, info
from router.embeddings import available_pathways
from training.data.facade import load_training_data


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
    data = load_training_data(cfg)

    if args.all_pathways:
        pathways = available_pathways(cfg)
    elif args.pathway:
        pathways = [args.pathway]
    else:
        pathways = [cfg.get("embedding.default_pathway") or available_pathways(cfg)[0]]

    for pw in pathways:
        info(f"[{pw}] building query embeddings"
             + (f" (split={args.split})" if args.split else "")
             + (" (from_nirt)" if args.from_nirt else "")
             + (f" (limit={args.limit})" if args.limit else ""))
        store = data.query_embeddings(
            pw, build=True, split=args.split, from_nirt=args.from_nirt,
            limit=args.limit, restart=args.restart, flush_every=args.flush_every,
        )
        info(f"[{pw}] done: {len(store)} embeddings (dim={store.dim})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
