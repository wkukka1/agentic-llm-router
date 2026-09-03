"""Normalize all sources -> data/processed/{responses,queries,models}.parquet.

Runs loaders, applies chance correction, builds the query / model tables, runs
the data-quality checks (fails on errors, warns on recoverable gaps), and
prints the Phase 0 summary report.
"""

from __future__ import annotations

import sys

from router.cli import base_parser, get_config, info
from router.data.chance_correction import estimate_model_bias
from router.data.quality import check_responses
from router.data.response_matrix import build_tables, write_tables


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--sources", default=None,
                        help="comma list subset of: routerbench,chatbot_arena,lm_eval_harness")
    parser.add_argument("--allow-warnings", action="store_true", default=True)
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as errors")
    args = parser.parse_args()
    cfg = get_config(args)

    sources = args.sources.split(",") if args.sources else cfg.get("source_order")
    info("loading + normalizing sources...")
    tables = build_tables(cfg, sources)

    report = check_responses(tables["responses"])
    print(report.render())

    bias = estimate_model_bias(tables["responses"])
    if not bias.empty:
        print("\n=== Per-model multiple-choice skill-above-chance ===")
        print(bias.to_string(index=False))

    if not report.ok():
        info("ERRORS present -- not writing outputs")
        report.raise_for_errors()
    if args.strict and report.warnings:
        info("strict mode: warnings present -- aborting")
        return 2

    paths = write_tables(tables, cfg)
    for name, p in paths.items():
        info(f"wrote {name}: {p} ({len(tables[name])} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
