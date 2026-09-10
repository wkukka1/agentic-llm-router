"""Compare Bernoulli vs Normal vs Beta vs ZOIB on the same representation + split.

    python scripts/compare_response_models.py
    python scripts/compare_response_models.py --split validation

Writes artifacts/phase2/comparison.json. Uses whatever checkpoints exist under
artifacts/phase1/baseline and artifacts/phase2/{normal,beta,zoib}.
"""

from __future__ import annotations

import sys

from router.cli import raw_parser
from router.config import load_config
from router.nirt.baseline.continuous_eval import compare_response_models


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dirs = {"bernoulli": cfg.root / "artifacts/phase1/baseline",
            "normal": cfg.root / "artifacts/phase2/normal",
            "beta": cfg.root / "artifacts/phase2/beta",
            "zoib": cfg.root / "artifacts/phase2/zoib"}
    out = compare_response_models(dirs, cfg=cfg, split=args.split)

    hdr = f"{'model':<11}{'NLL':>9}{'MAE':>9}{'RMSE':>9}{'mean-ECE':>10}{'cov50':>8}{'cov90':>8}{'acc':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in out["models"].items():
        print(f"{name:<11}{r['nll']:>9.4f}{r['mae']:>9.4f}{r['rmse']:>9.4f}"
              f"{r['mean_calibration_ece']:>10.4f}"
              f"{(r.get('coverage_50') or float('nan')):>8.3f}{(r.get('coverage_90') or float('nan')):>8.3f}"
              f"{r['accuracy']:>8.3f}")
    print(f"\nwrote {cfg.root / 'artifacts/phase2/comparison.json'}")
    print("(NLL is each model's own proper score — compare within a family; "
          "read MAE/RMSE/calibration/coverage across rows.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
