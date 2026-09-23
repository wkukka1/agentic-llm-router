"""Obtain the LLMRouterBench performance-cost benchmark results
(arXiv 2601.07206, Findings of ACL 2026).

The paper's authors publish pre-collected result JSON files as a tarball on the Hugging
Face Hub (huggingface.co/datasets/NPULH/LLMRouterBench) -- see
evaluations/LLMRouterBench/README.md's "Data Download" section for the alternative Google
Drive / Baidu Netdisk mirrors if this one is unavailable. This script fetches and extracts
that tarball into ``sources.llmrouterbench.local_dir``.

    python scripts/data/download_llmrouterbench.py --config configs/llmrouterbench.yaml

This project only uses the performance-cost setting (13 flagship models); the tarball
contains both settings, so this only extracts the ``results/bench/`` subtree relevant here
-- run `du -sh` on the extracted directory to check size before committing disk space, the
full release is multi-GB.
"""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from router.config import section
from training.cli import base_parser, get_config, info

REPO_ID = "NPULH/LLMRouterBench"
ARCHIVE_FILENAME = "bench-release.tar.gz"


def main() -> int:
    parser = base_parser(__doc__)
    args = parser.parse_args()
    cfg = get_config(args)

    src = section(cfg, "sources").get("llmrouterbench", {})
    dest = cfg.resolve(src.get("local_dir", "evaluations/LLMRouterBench/results/bench"))

    info(f"downloading {ARCHIVE_FILENAME} from {REPO_ID} ...")
    archive_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=ARCHIVE_FILENAME)

    info(f"extracting into {dest.parent} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        # a fetched-third-party archive gets the hardened extraction filter
        # (PEP 706) -- refuses members that would write outside dest.parent
        # (tar-slip, CVE-2007-4559), absolute paths, or device/symlink tricks
        tf.extractall(dest.parent, filter="data")

    if not dest.exists():
        info(f"ERROR: expected {dest} after extraction; check the archive's internal layout")
        return 1
    n_files = sum(1 for _ in dest.rglob("*.json"))
    info(f"done: {n_files} result files under {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
