"""Obtain the IRT-Router benchmark payload (arXiv 2506.01048, ACL'25).

The paper's authors publish their processed per-(query, model) response data and
precomputed BERT embeddings in the GitHub repo ``Mercidaiha/IRT-Router`` as Git
LFS objects. This script fetches them from the LFS media endpoint (no ``git
lfs`` needed) into ``sources.irt_router.local_dir``, preserving the repo's
``data/`` and ``utils/`` sub-paths.

    python scripts/data/download_irt_router.py --config configs/irt_router.yaml

~720 MB total (train.csv 450 MB, test1.csv 192 MB, test2.csv 57 MB).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from training.cli import base_parser, get_config, info

# Repo-relative paths to fetch (all Git LFS).
FILES = [
    "data/train.csv",
    "data/test1.csv",
    "data/test2.csv",
    "utils/map/query.csv",
    "utils/map/llm.csv",
    "utils/bert_embeddings/query_embeddings.pkl",
    "utils/bert_embeddings/llm_embeddings.pkl",
    "utils/relevance/relevance_vectors_cluster_train_bert.pkl",
    "utils/relevance/relevance_vectors_cluster_test_bert.pkl",
    "utils/cold/test_avg_embeddings_bert.pkl",
]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted host)
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {dest.name}: {done >> 20} / {total >> 20} MiB "
                          f"({pct:.0f}%)", end="", file=sys.stderr)
        print("", file=sys.stderr)
    tmp.replace(dest)


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = get_config(args)

    src = cfg.get("sources.irt_router")
    if src is None:
        raise SystemExit("config has no sources.irt_router block "
                         "(use --config configs/irt_router.yaml)")
    dest_root = cfg.resolve(src.get("local_dir", "data/raw/irt_router"))
    repo = src.get("repo", "Mercidaiha/IRT-Router")
    commit = src.get("commit", "main")
    base = f"https://media.githubusercontent.com/media/{repo}/{commit}"

    for rel in FILES:
        dest = dest_root / rel
        if dest.exists() and dest.stat().st_size > 0:
            info(f"present: {rel} ({dest.stat().st_size >> 20} MiB)")
            continue
        info(f"fetching {rel}")
        _download(f"{base}/{rel}", dest)

    (dest_root / "PROVENANCE.txt").write_text(
        f"github.com/{repo}\ncommit {commit}\nfetched via LFS media endpoint\n",
        encoding="utf-8",
    )
    info(f"done -> {dest_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
