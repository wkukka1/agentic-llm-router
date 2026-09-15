"""Evaluate a trained Phase 1/2 checkpoint (Bernoulli or a continuous head).

    python scripts/evaluate.py --checkpoint artifacts/phase1/baseline
    python scripts/evaluate.py --checkpoint artifacts/phase2/zoib --split validation
    python scripts/evaluate.py --checkpoint artifacts/phase2/normal --level 0.8

Dispatches on the checkpoint's response model: Bernoulli -> prediction /
calibration / cold-start / ICC / theta spectrum; continuous -> the same plus
proper-likelihood / interval-coverage / boundary-calibration metrics. Writes the
JSON + plots under the checkpoint dir.
"""

from __future__ import annotations

import sys

from training.cli import raw_parser
from router.config import load_config
from training.nirt.baseline.checkpoint import load_baseline_run


def _print_bernoulli(res: dict, split: str, checkpoint: str) -> None:
    pred = res["prediction"]
    print(f"\n=== {split} ===")
    for k in ("n", "accuracy", "bce", "log_loss", "brier", "auc", "ece", "spearman_r",
              "delta_bce_vs_model_mean"):
        if k in pred:
            v = pred[k]
            print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
    spec = res["theta_spectrum"]
    print(f"\n  theta singular values: {[round(s, 3) for s in spec['singular_values']]}")
    print(f"  theta entropy rank   : {spec['entropy_rank']:.3f} / {spec['shape'][1]}")
    print(f"  collapsed            : {spec['collapsed']}")
    cs = res.get("cold_start")
    if cs and "pooled" in cs:
        p = cs["pooled"]
        print(f"\n  cold-start NIRT   acc {p['baseline_nirt']['accuracy']:.4f}  "
              f"bce {p['baseline_nirt']['bce']:.4f}")
        print(f"  cold-start global acc {p['global_mean']['accuracy']:.4f}  "
              f"bce {p['global_mean']['bce']:.4f}")
    elif cs:
        print(f"\n  cold-start: {cs.get('skipped')}")
    if res["parameter_summary"]["pathology_flags"]:
        print("\n  PATHOLOGY FLAGS:", res["parameter_summary"]["pathology_flags"])
    print(f"\n  wrote calibration.json, theta_spectrum.json, plots/ under {checkpoint}")


def _print_continuous(res: dict, split: str) -> None:
    m = res["metrics"]
    print(f"\n=== {res['response_model']} — {split} ===")
    for k in ("nll_mean", "nll_median", "mae", "rmse", "mean_calibration_ece",
              "coverage_50", "coverage_90", "bce_diag_bce", "bce_diag_accuracy",
              "mean_uncertainty"):
        if k in m:
            print(f"  {k:22s} {m[k]:.4f}")
    if res["boundary_calibration"]:
        b = res["boundary_calibration"]
        print(f"  P(y=0) pred {b['pred_P(y=0)_mean']:.3f} vs actual {b['actual_freq(y=0)']:.3f}")
        print(f"  P(y=1) pred {b['pred_P(y=1)_mean']:.3f} vs actual {b['actual_freq(y=1)']:.3f}")
    if res["parameter_summary"]["response_flags"]:
        print("  FLAGS:", res["parameter_summary"]["response_flags"])
    cs = res.get("cold_start")
    if cs and "pooled" in cs:
        print(f"  cold-start NLL {cs['pooled']['baseline_nirt'].get('bce', float('nan')):.4f}")
    print(f"\n  theta entropy rank {res['theta_spectrum']['entropy_rank']:.2f} / "
          f"{res['theta_spectrum']['shape'][1]}")


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="phase0 config")
    ap.add_argument("--split", default="test")
    ap.add_argument("--level", type=float, default=0.90, help="continuous: interval coverage level")
    ap.add_argument("--n-icc", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, _, _ = load_baseline_run(args.checkpoint)

    if model.response_model == "bernoulli":
        from training.nirt.baseline.eval import evaluate_checkpoint

        res = evaluate_checkpoint(args.checkpoint, phase0_cfg=cfg, split=args.split, n_icc=args.n_icc)
        _print_bernoulli(res, args.split, args.checkpoint)
    else:
        from training.nirt.baseline.continuous_eval import evaluate_continuous

        res = evaluate_continuous(args.checkpoint, phase0_cfg=cfg, split=args.split, level=args.level)
        _print_continuous(res, args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
