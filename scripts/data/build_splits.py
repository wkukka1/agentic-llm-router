"""Build deterministic train/validation/test/cold-start splits.

Reads data/processed/responses.parquet, writes data/splits/*.json, and verifies
no query-id leakage across train/validation/test.
"""

from __future__ import annotations

import sys

from training.cli import base_parser, get_config, info
from training.data.response_matrix import read_responses
from training.data.splits import check_leakage, make_presplit, make_splits, write_splits


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = get_config(args)

    responses = read_responses(cfg)
    if bool(cfg.get("split.presplit", False)):
        info("split.presplit: honouring the source's train/test1/test2 partition")
        splits = make_presplit(responses, cfg)
    else:
        splits = make_splits(responses, cfg)

    problems = check_leakage(splits)
    if problems:
        for p in problems:
            info(f"LEAKAGE: {p}")
        raise SystemExit("split leakage detected")

    paths = write_splits(splits, cfg)
    for k, p in paths.items():
        n = len(splits[k].get("query_ids", splits[k].get("model_ids", [])))
        info(f"wrote {k}: {p} (n={n})")
    ood_n = len(splits["ood"]["query_ids"]) if isinstance(splits.get("ood"), dict) else 0
    info(
        f"train/val/test/ood query counts: "
        f"{len(splits['train']['query_ids'])} / "
        f"{len(splits['validation']['query_ids'])} / "
        f"{len(splits['test']['query_ids'])} / {ood_n}; "
        f"cold-start models: {splits['cold_start_models']['model_ids']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
