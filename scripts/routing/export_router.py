"""Write a router file for the gateway (``interceptor.router.artifact``).

    python scripts/routing/export_router.py --config configs/llmrouterbench.yaml \
        --dataset llmrouterbench --out router-llmrouterbench.npz            # trained MLP (default)
    python scripts/routing/export_router.py ... --kind knn --k 25            # kNN index
    python scripts/routing/export_router.py ... --kind nirt --run lrb-nirt-projected  # a trained NIRT run

``mlp`` holds only the network's weights (no training prompts or scores; what
the gateway ships); ``knn`` holds the training prompts' embeddings and scores;
``nirt`` folds a trained NIRT run into the same ``mlp`` format (exact for a fixed
pool; see :mod:`training.trainers.nirt_router_export`) and checks the saved file
scores like the run through the gateway's own loader.
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
    ap.add_argument("--kind", choices=("mlp", "knn", "nirt"), default="mlp")
    ap.add_argument("--hidden", type=int, default=256, help="mlp: hidden units")
    ap.add_argument("--k", type=int, default=25, help="knn: neighbours")
    ap.add_argument("--run", default="lrb-nirt-projected", help="nirt: the run under nirt_runs/")
    ap.add_argument("--pool", default="",
                    help="nirt: comma list of model ids to fold over instead of the run's trained pool, "
                         "each scored from its own profile (e.g. the Claude tiers)")
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
    elif args.kind == "nirt":
        from training.trainers.nirt_router_export import export_nirt_router, parity_max_diff_file

        pool = [m.strip() for m in args.pool.split(",") if m.strip()] or None
        payload, encoder, costs, header, (e_q, want) = export_nirt_router(data, args.run, dataset=args.dataset,
                                                                          pool_ids=pool)
        payload.save(out, encoder=encoder, model_costs=costs, meta=header)
        saved = parity_max_diff_file(out, payload.model_ids, e_q, want)
        what = (f"NIRT run {args.run} folded to an MLP, hidden={payload.hidden}; "
                f"parity max |dP| {header['parity_max_abs_diff']:.2g} in memory, {saved:.2g} after reload")
    else:
        from training.trainers.knn_router import export_knn_artifact

        payload, header = export_knn_artifact(data, k=args.k, pathway=args.pathway, dataset=args.dataset)
        payload.save(out, meta=header)
        what = f"kNN, k={args.k}, {header['n_queries']} training prompts"
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB): {what}, {len(payload.model_ids)} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
