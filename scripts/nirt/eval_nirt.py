"""Evaluate a saved NIRT run on a split (prediction quality).

    python scripts/nirt/eval_nirt.py --run nirt-1d-projected --split test
    python scripts/nirt/eval_nirt.py --run nirt-1d-projected --split validation

Writes <runs_dir>/<run>/eval_<split>.json and prints the table.
"""

from __future__ import annotations

import json
import sys

import yaml

from training.cli import raw_parser, resolve, write_json
from router.config import load_config
from router.nirt.checkpoint import load_run
from evaluation.nirt.evaluate import evaluate_split


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--run", required=True, help="run name under runs_dir")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--config", default="configs/nirt.yaml")
    ap.add_argument("--phase0-config", default=None)
    args = ap.parse_args()

    nirt_cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    runs_dir = resolve(nirt_cfg.get("runs_dir", "data/processed/nirt_runs"))

    model, run_cfg, model_index = load_run(args.run, runs_dir=runs_dir)
    p0 = load_config(args.phase0_config) if args.phase0_config else load_config()
    pathway = run_cfg.get("data", {}).get("pathway", "irt")

    metrics = evaluate_split(model, model_index, args.split, phase0_cfg=p0, pathway=pathway)

    printable = {k: v for k, v in metrics.items() if k != "baselines"}
    printable["baselines"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "per_model"}
        for k, v in metrics["baselines"].items()
    }
    print(json.dumps(printable, indent=2, default=float))

    write_json(runs_dir / args.run / f"eval_{args.split}.json", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
