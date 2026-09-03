"""Synthetic recovery gate for a Phase 2 continuous response head.

    python scripts/synthetic_continuous.py --response normal
    python scripts/synthetic_continuous.py --response beta --epochs 60
    python scripts/synthetic_continuous.py --response zoib

Exit 0 only if recovery passes. Writes artifacts/phase2/synthetic_<response>/recovery.json.
"""

from __future__ import annotations

import argparse
import json
import sys

from router.config import load_config
from router.nirt.baseline_train import fit
from router.nirt.continuous_synthetic import make_synthetic_continuous, recovery_report, to_arrays


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--response", required=True, choices=["normal", "beta", "zoib"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--queries", type=int, default=1500)
    ap.add_argument("--models", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    syn = make_synthetic_continuous(args.response, n_queries=args.queries, n_models=args.models,
                                    K=args.k, seed=args.seed)
    tr, va = to_arrays(syn, seed=args.seed)
    cfg = {
        "seed": args.seed,
        "model": {"theta_dim": args.k, "model_params": "free", "query_hidden": 64,
                  "use_length_head": False},
        "ablation": {"use_relevance": False, "use_interaction": False, "use_warmup": False},
        "response": {"model": args.response, "cfg": {}},
        "regularization": {"theta_l2": 1e-4, "theta_center_l2": 1e-3, "difficulty_l2": 1e-4},
        "train": {"target": "soft", "lr": 3e-3, "batch_size": 2048, "epochs": args.epochs,
                  "patience": 15, "device": "cpu", "grad_clip": 5.0},
    }
    res = fit(cfg, arrays=(tr, va), save=False, verbose=True)
    rep = recovery_report(res.model, syn, va)
    print("\n=== recovery ===")
    print(json.dumps(rep, indent=2, default=float))

    out = load_config().root / f"artifacts/phase2/synthetic_{args.response}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "recovery.json").write_text(json.dumps({"config": cfg, "report": rep}, indent=2, default=float),
                                       encoding="utf-8")
    print("PASS" if rep["passed"] else "FAIL")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
