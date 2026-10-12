"""Write a kNN router file for the gateway (``interceptor.router.artifact``).

    python scripts/routing/export_knn_router.py --config configs/llmrouterbench.yaml \
        --k 25 --dataset llmrouterbench --out router-llmrouterbench.npz

Needs the pool's processed data and retrieval embeddings (RUNBOOK §1); the
file it writes needs neither.
"""

from __future__ import annotations

import sys
from pathlib import Path

from router.config import load_config
from training.cli import raw_parser
from training.data.facade import load_training_data
from training.trainers.knn_router import export_knn_artifact


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--config", required=True, help="the pool's data config")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--pathway", default="retrieval")
    ap.add_argument("--dataset", default="", help="a name recorded in the file, e.g. llmrouterbench")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = load_training_data(load_config(args.config))
    payload, header = export_knn_artifact(data, k=args.k, pathway=args.pathway, dataset=args.dataset)
    out = Path(args.out).expanduser()
    payload.save(out, meta=header)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {header['n_queries']} queries x "
          f"{len(payload.model_ids)} models, k={payload.k}, encoder {payload.encoder['model_name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
