"""Evaluate a trained Phase 1 baseline checkpoint.

    python scripts/evaluate_baseline.py --checkpoint artifacts/phase1/baseline
    python scripts/evaluate_baseline.py --checkpoint artifacts/phase1/baseline --split validation

Produces accuracy / log-loss / Brier / calibration / cold-start / ICC / theta
spectrum and writes them under the checkpoint dir (metrics.json, calibration.json,
theta_spectrum.json, plots/).
"""

from __future__ import annotations

import argparse
import json
import sys

from router.config import load_config
from router.nirt.baseline_eval import evaluate_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="artifacts/phase1/baseline")
    ap.add_argument("--config", default=None, help="phase0 config")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-icc", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config(args.config)
    res = evaluate_checkpoint(args.checkpoint, phase0_cfg=cfg, split=args.split, n_icc=args.n_icc)

    pred = res["prediction"]
    print(f"\n=== {args.split} ===")
    for k in ("n", "accuracy", "bce", "log_loss", "brier", "auc", "ece", "spearman_r",
              "delta_bce_vs_model_mean"):
        if k in pred:
            print(f"  {k:24s} {pred[k]:.4f}" if isinstance(pred[k], float) else f"  {k:24s} {pred[k]}")
    spec = res["theta_spectrum"]
    print(f"\n  theta singular values: {[round(s, 3) for s in spec['singular_values']]}")
    print(f"  theta effective rank : {spec['effective_rank']:.3f} / {spec['shape'][1]}")
    print(f"  collapsed            : {spec['collapsed']}")
    if res.get("cold_start") and "pooled" in res["cold_start"]:
        cs = res["cold_start"]["pooled"]
        print(f"\n  cold-start NIRT   acc {cs['baseline_nirt']['accuracy']:.4f}  "
              f"bce {cs['baseline_nirt']['bce']:.4f}")
        print(f"  cold-start global acc {cs['global_mean']['accuracy']:.4f}  "
              f"bce {cs['global_mean']['bce']:.4f}")
    elif res.get("cold_start"):
        print(f"\n  cold-start: {res['cold_start'].get('skipped')}")
    if res["parameter_summary"]["pathology_flags"]:
        print("\n  PATHOLOGY FLAGS:", res["parameter_summary"]["pathology_flags"])
    print(f"\n  wrote calibration.json, theta_spectrum.json, plots/ under {args.checkpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
