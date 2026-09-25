"""Cost/quality sweep of the selection rule: argmax vs cheapest-within-margin.

Reproduces the margin tables in docs/prompt_decomposition_experiments.md. The
measurements behind that document were originally run from throwaway scripts,
which meant the numbers could not be re-derived from the repo; this is that
entry point.

The columns that matter are `vs_mixture` and `top_model_share`. A cost/quality
point means nothing on its own -- randomly splitting traffic between the
cheapest and strongest model traces the straight line between them for free, so
a policy sitting on that line has bought nothing with its predictions.
`top_model_share` says whether the router is actually routing or has collapsed
onto one model.

    # RouterBench, the trained NIRT router
    PYTHONPATH=src python scripts/margin_sweep.py
    PYTHONPATH=src python scripts/margin_sweep.py --nirt-run nirt-4d-projected

    # the same rule on an untrained retrieval router, as a control
    PYTHONPATH=src python scripts/margin_sweep.py --router knn --k 25

    # SPROUT (2024-25 pool). NOTE: its prices in configs/model_prices.yaml are
    # marked UNVERIFIED -- treat dollar figures as indicative only.
    PYTHONPATH=src python scripts/margin_sweep.py --pool sprout

    # "how much margin buys a 50% saving on this pool?"
    PYTHONPATH=src python scripts/margin_sweep.py --target-saving 0.5
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from evaluation.prompt_decomposition.policy_eval import (
    frontier_gain,
    margin_for_target_saving,
    margin_sweep,
)
from router.config import load_config
from training.cli import float_table, raw_parser, write_json

DEFAULT_MARGINS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3)


def _routerbench(args):
    """(pred, true, cost, model_ids) for the RouterBench eval split."""
    from evaluation.nirt.routing import eval_matrices
    from router.routing.routers import KNNRouter, NIRTRouter
    from training.data.facade import load_training_data

    data = load_training_data(load_config(args.config))
    true_df, cost_df = eval_matrices(data, split=args.split)
    models = list(true_df.columns)
    router = (
        KNNRouter(data, k=args.k, model_ids=models)
        if args.router == "knn"
        else NIRTRouter.from_run(args.nirt_run, data=data)
    )
    pred = router.aligned_scores(list(true_df.index)).reindex(columns=models).to_numpy(float)
    return pred, true_df.to_numpy(float), cost_df.to_numpy(float), models, router.name


def _sprout(args):
    """(pred, true, cost, model_ids) for SPROUT, kNN over frozen embeddings.

    SPROUT ships no trained router, so the predictor is nearest-neighbour over
    the train split -- which is also the paper's headline baseline.
    """
    from router.embeddings.encoder import EncoderConfig, TextEncoder
    from training.prompt_decomposition.data.pools import load_pool, prices_from_config

    pool = load_pool("sprout", split=args.split, prices=prices_from_config())
    priced = np.isfinite(pool.cost).all(axis=0)
    if not priced.all():
        dropped = [m for m, p in zip(pool.models, priced) if not p]
        print(f"dropping {len(dropped)} unpriced models: {dropped}")
    models = [m for m, p in zip(pool.models, priced) if p]
    correct, cost = pool.correct[:, priced], pool.cost[:, priced]

    encoder = TextEncoder(EncoderConfig(name="margin-sweep", model_name=args.encoder))
    vec = encoder.encode(pool.prompts["prompt"].tolist())
    vec /= np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None)

    is_train = (pool.prompts["split"] == "train").to_numpy()
    train, test = np.where(is_train)[0], np.where(~is_train)[0]
    pred = np.empty((len(test), len(models)))
    for start in range(0, len(test), 512):
        block = vec[test[start:start + 512]] @ vec[train].T
        for i, row in enumerate(block):
            nn = np.argpartition(-row, min(args.k, len(row) - 1))[:args.k]
            pred[start + i] = correct[train[nn]].mean(axis=0)
    return pred, correct[test], cost[test], models, f"knn(k={args.k})"


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--pool", default="routerbench", choices=["routerbench", "sprout"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--router", default="nirt", choices=["nirt", "knn"],
                    help="routerbench only; sprout is always knn")
    ap.add_argument("--nirt-run", default="nirt-2d-projected")
    ap.add_argument("--k", type=int, default=16, help="neighbours for the knn router")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--config", default=None)
    ap.add_argument("--lam", type=float, default=0.0, help="cost weight inside the utility")
    ap.add_argument("--margins", default=",".join(str(m) for m in DEFAULT_MARGINS))
    ap.add_argument("--target-saving", type=float, default=None,
                    help="also report the smallest margin reaching this saving, e.g. 0.5")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    build = _sprout if args.pool == "sprout" else _routerbench
    pred, true, cost, models, router_name = build(args)
    print(f"{args.pool}[{args.split}]: {true.shape[0]} queries x {true.shape[1]} models, "
          f"router={router_name}", flush=True)

    margins = tuple(float(m) for m in args.margins.split(",") if m.strip())
    table = margin_sweep(pred, true, cost, models, margins=margins, lam=args.lam)

    per_model = pd.DataFrame({"quality": true.mean(axis=0),
                              "cost_per_1k": cost.mean(axis=0) * 1000.0}, index=models)
    table = frontier_gain(table, per_model["cost_per_1k"].idxmin(),
                          per_model["quality"].idxmax(), per_model)

    with float_table(220):
        print(f"\n=== {args.pool}: selection rule sweep (lam={args.lam}) ===")
        print(table.to_string(index=False))

    if args.target_saving is not None:
        try:
            hit = margin_for_target_saving(table, args.target_saving)
            print(f"\nto spend {args.target_saving:.0%} less than "
                  f"{hit['baseline_policy']}: margin {hit['margin']:g} "
                  f"(actual saving {hit['saving']:.0%}, "
                  f"quality given up {hit['quality_given_up']:.4f})")
        except ValueError as exc:
            print(f"\n{exc}")

    if args.out:
        write_json(load_config(args.config).resolve(args.out), {
            "pool": args.pool, "split": args.split, "router": router_name,
            "lam": args.lam, "rows": table.to_dict(orient="records"),
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
