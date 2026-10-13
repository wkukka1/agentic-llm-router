"""Write a router file for the gateway (``interceptor.router.artifact``).

    python scripts/routing/export_router.py --config configs/llmrouterbench.yaml \
        --dataset llmrouterbench --out router-llmrouterbench.npz            # trained MLP (default)
    python scripts/routing/export_router.py ... --kind knn --k 25            # kNN index

``mlp`` holds only the network's weights (no training prompts or scores; what
the gateway ships); ``knn`` holds the training prompts' embeddings and scores.
Needs the pool's processed data and retrieval embeddings (RUNBOOK §1); the
file it writes needs neither.
"""

from __future__ import annotations

import sys
from pathlib import Path

from router.config import load_config
from training.cli import raw_parser
from training.data.facade import load_training_data


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--config", required=True, help="the pool's data config")
    ap.add_argument("--kind", choices=("mlp", "knn"), default="mlp")
    ap.add_argument("--hidden", type=int, default=256, help="mlp: hidden units")
    ap.add_argument("--k", type=int, default=25, help="knn: neighbours")
    ap.add_argument("--pathway", default="retrieval")
    ap.add_argument("--dataset", default="", help="a name recorded in the file, e.g. llmrouterbench")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = load_training_data(load_config(args.config))
    out = Path(args.out).expanduser()
    if args.kind == "mlp":
        from training.trainers.mlp_router import export_mlp_router

        payload, encoder, costs, header = export_mlp_router(data, hidden=args.hidden, pathway=args.pathway,
                                                             dataset=args.dataset)
        payload.save(out, encoder=encoder, model_costs=costs, meta=header)
        what = f"trained MLP, hidden={args.hidden}"
    else:
        from training.trainers.knn_router import export_knn_artifact

        payload, header = export_knn_artifact(data, k=args.k, pathway=args.pathway, dataset=args.dataset)
        payload.save(out, meta=header)
        what = f"kNN, k={args.k}, {header['n_queries']} training prompts"
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB): {what}, {len(payload.model_ids)} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
