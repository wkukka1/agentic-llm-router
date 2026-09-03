"""Synthetic multidimensional-IRT recovery test for the Phase 1 baseline.

    python scripts/synthetic_recovery.py
    python scripts/synthetic_recovery.py --k 4 --queries 2000 --epochs 40

Generates a well-specified IRT world, trains BaselineNIRT on it, and checks
recovery of predictive quality + difficulty ordering + ability ordering + ICC
sanity. Exit code 0 only if the recovery criteria pass. Writes
artifacts/phase1/synthetic/recovery.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from router.config import load_config
from router.nirt.baseline_train import fit
from router.nirt.synthetic import make_synthetic, recovery_report, to_arrays


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--queries", type=int, default=1500)
    ap.add_argument("--models", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    syn = make_synthetic(n_queries=args.queries, n_models=args.models, K=args.k, seed=args.seed)
    tr, va = to_arrays(syn, seed=args.seed)
    print(f"[synthetic] Q={args.queries} M={args.models} K={args.k}  bayes_bce={syn.bayes_bce():.4f}")

    cfg = {
        "seed": args.seed,
        "model": {"theta_dim": args.k, "model_params": "free", "query_hidden": 64,
                  "use_length_head": False, "use_relevance": False, "use_interaction": False},
        "ablation": {"use_relevance": False, "use_warmup": False, "use_interaction": False},
        "regularization": {"theta_l2": 1e-4, "theta_center_l2": 1e-3, "difficulty_l2": 1e-4},
        "train": {"target": "binary", "lr": 3e-3, "batch_size": 2048, "epochs": args.epochs,
                  "patience": 8, "device": "cpu"},
    }
    res = fit(cfg, arrays=(tr, va), save=False, verbose=True)
    rep = recovery_report(res.model, syn, va)
    print("\n=== recovery report ===")
    print(json.dumps(rep, indent=2, default=float))

    out = load_config().root / "artifacts/phase1/synthetic"
    out.mkdir(parents=True, exist_ok=True)
    (out / "recovery.json").write_text(json.dumps({"config": cfg, "report": rep}, indent=2, default=float),
                                       encoding="utf-8")
    print(f"\n[synthetic] wrote {out / 'recovery.json'}")
    print("PASS" if rep["passed"] else "FAIL")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
