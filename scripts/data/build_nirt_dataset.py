"""Build the NIRT observation table -> data/processed/nirt_observations.parquet.

One lightweight row per (query_id, model_id, metric) correctness example:
query_id, model_id, target, score_raw, metric_type, source, split, cost,
is_multiple_choice, n_choices. No embeddings are stored here -- NIRTDataset
joins them by id at access time.

    python scripts/data/build_nirt_dataset.py
    python scripts/data/build_nirt_dataset.py --score-kind raw --metrics accuracy,mc_accuracy
"""

from __future__ import annotations

import sys

from router.cli import base_parser, get_config, info
from router.data.nirt import NIRTDataset, build_nirt_observations, write_nirt_observations


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--score-kind", default="effective",
                        choices=["effective", "raw", "corrected"])
    parser.add_argument("--metrics", default=None,
                        help="comma list; default = all correctness metrics")
    parser.add_argument("--models", default="warm", choices=["warm", "all", "cold"])
    parser.add_argument("--pathway", default="irt",
                        help="check the dataset builds against this embedding pathway")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()
    cfg = get_config(args)

    metrics = args.metrics.split(",") if args.metrics else None
    df = build_nirt_observations(cfg, score_kind=args.score_kind, metrics=metrics,
                                 models=args.models)
    path = write_nirt_observations(df, cfg)
    info(f"wrote {len(df):,} observations -> {path}")
    info("by split:  " + str(df["split"].value_counts().to_dict()))
    info("by metric: " + str(df["metric_type"].value_counts().to_dict()))
    info(f"queries={df['query_id'].nunique():,}  models={df['model_id'].nunique()}  "
         f"target in [{df['target'].min():.2f}, {df['target'].max():.2f}]")

    if not args.no_check:
        try:
            ds = NIRTDataset.from_config(cfg, split="train", pathway=args.pathway,
                                        observations=df)
            info(f"train NIRTDataset: {len(ds):,} examples "
                 f"(dropped {ds.dropped:,} with no embedding), "
                 f"q_dim={ds.query_dim} m_dim={ds.model_dim}")
            s = ds[0]
            info(f"sample[0]: query_embedding{s['query_embedding'].shape} "
                 f"model_embedding{s['model_embedding'].shape} target={s['target']:.3f}")
        except FileNotFoundError as exc:
            info(f"(embeddings not ready: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
