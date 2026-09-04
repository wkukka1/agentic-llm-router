"""Synthetic IRT recovery gate for the NIRT model (Bernoulli or a continuous head).

    python scripts/synthetic.py                       # Bernoulli (Phase 1)
    python scripts/synthetic.py --response zoib
    python scripts/synthetic.py --response beta --k 4 --queries 2000 --epochs 60

Generates a well-specified IRT world, trains ``BaselineNIRT`` with the chosen
response head on it, and checks recovery of predictive quality + difficulty /
ability ordering (and, for the continuous heads, the dispersion ordering + finite
NLL). Exit code 0 only if the recovery criteria pass. Writes
artifacts/phase{1,2}/synthetic[_<response>]/recovery.json.
"""

from __future__ import annotations

import json
import sys

from router.cli import raw_parser, write_json
from router.config import load_config
from router.nirt.baseline_train import fit
from router.nirt.synthetic import make_synthetic, make_synthetic_continuous, recovery_report, to_arrays


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--response", default="bernoulli",
                    choices=["bernoulli", "normal", "beta", "zoib"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--queries", type=int, default=1500)
    ap.add_argument("--models", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bernoulli = args.response == "bernoulli"
    epochs = args.epochs if args.epochs is not None else (40 if bernoulli else 60)

    if bernoulli:
        syn = make_synthetic(n_queries=args.queries, n_models=args.models, K=args.k, seed=args.seed)
        print(f"[synthetic] Q={args.queries} M={args.models} K={args.k}  "
              f"bayes_bce={syn.bayes_bce():.4f}")
        train_over = {"target": "binary"}
        response = {"model": "bernoulli"}
    else:
        syn = make_synthetic_continuous(args.response, n_queries=args.queries,
                                        n_models=args.models, K=args.k, seed=args.seed)
        train_over = {"target": "soft", "grad_clip": 5.0}
        response = {"model": args.response, "cfg": {}}

    tr, va = to_arrays(syn, seed=args.seed)
    cfg = {
        "seed": args.seed,
        "model": {"theta_dim": args.k, "model_params": "free", "query_hidden": 64,
                  "use_length_head": False, "use_relevance": False, "use_interaction": False},
        "ablation": {"use_relevance": False, "use_interaction": False, "use_warmup": False},
        "response": response,
        "regularization": {"theta_l2": 1e-4, "theta_center_l2": 1e-3, "difficulty_l2": 1e-4},
        "train": {"lr": 3e-3, "batch_size": 2048, "epochs": epochs,
                  "patience": 8 if bernoulli else 15, "device": "cpu", **train_over},
    }
    res = fit(cfg, arrays=(tr, va), save=False, verbose=True)
    rep = recovery_report(res.model, syn, va)
    print("\n=== recovery report ===")
    print(json.dumps(rep, indent=2, default=float))

    sub = "synthetic" if bernoulli else f"synthetic_{args.response}"
    phase = "phase1" if bernoulli else "phase2"
    write_json(load_config().root / f"artifacts/{phase}/{sub}/recovery.json",
               {"config": cfg, "report": rep})
    print("PASS" if rep["passed"] else "FAIL")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
