"""Derive per-cell token counts for RouterBench (which ships only total cost).

    python scripts/data/build_routerbench_tokens.py

Tokenises each configured shot's native prompt and every model's recorded
response text (roberta-base / GPT-2 BPE) and writes
``data/processed/routerbench_tokens.parquet`` with
``query_id, model_id, input_tokens, output_tokens``. See training.data.tokens.
"""

from __future__ import annotations

import sys

from training.cli import base_parser, get_config, info
from training.data.tokens import build_routerbench_tokens


def main() -> int:
    args = base_parser(__doc__).parse_args()
    info(f"wrote {build_routerbench_tokens(get_config(args))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
