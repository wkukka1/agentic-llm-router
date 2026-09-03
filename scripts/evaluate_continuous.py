"""Evaluate a Phase 2 checkpoint (continuous head, or the Bernoulli control).

    python scripts/evaluate_continuous.py --checkpoint artifacts/phase2/zoib
    python scripts/evaluate_continuous.py --checkpoint artifacts/phase2/normal --split validation
"""

from __future__ import annotations

import argparse
import json
import sys

from router.config import load_config
from router.nirt.checkpoint import load_run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--level", type=float, default=0.90)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, _, _ = load_run(args.checkpoint)
    if model.response_model == "bernoulli":
        from router.nirt.baseline_eval import evaluate_checkpoint

        r = evaluate_checkpoint(args.checkpoint, phase0_cfg=cfg, split=args.split)
        print(json.dumps(r["prediction"], indent=2, default=float))
        return 0

    from router.nirt.continuous_eval import evaluate_continuous

    r = evaluate_continuous(args.checkpoint, phase0_cfg=cfg, split=args.split, level=args.level)
    m = r["metrics"]
    print(f"\n=== {r['response_model']} — {args.split} ===")
    for k in ("nll_mean", "nll_median", "mae", "rmse", "mean_calibration_ece",
              "coverage_50", "coverage_90", "bce_diag_bce", "bce_diag_accuracy", "mean_uncertainty"):
        if k in m:
            print(f"  {k:22s} {m[k]:.4f}")
    if r["boundary_calibration"]:
        b = r["boundary_calibration"]
        print(f"  P(y=0) pred {b['pred_P(y=0)_mean']:.3f} vs actual {b['actual_freq(y=0)']:.3f}")
        print(f"  P(y=1) pred {b['pred_P(y=1)_mean']:.3f} vs actual {b['actual_freq(y=1)']:.3f}")
    if r["parameter_summary"]["response_flags"]:
        print("  FLAGS:", r["parameter_summary"]["response_flags"])
    cs = r.get("cold_start")
    if cs and "pooled" in cs:
        print(f"  cold-start NLL {cs['pooled']['baseline_nirt'].get('bce', float('nan')):.4f}")
    print(f"\n  theta eff. rank {r['theta_spectrum']['effective_rank']:.2f} / {r['theta_spectrum']['shape'][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
