"""Evaluate a saved NIRT run on a split (prediction quality).

    python scripts/nirt/eval_nirt.py --run nirt-1d-projected --split test
    python scripts/nirt/eval_nirt.py --run nirt-1d-projected --split validation

Writes <runs_dir>/<run>/eval_<split>.json and prints the table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from router.config import load_config
from router.nirt.evaluate import evaluate_split
from router.nirt.train import load_run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name under runs_dir")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--config", default="configs/nirt.yaml")
    ap.add_argument("--phase0-config", default=None)
    args = ap.parse_args()

    root = Path(load_config().root)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    nirt_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    runs_dir = nirt_cfg.get("runs_dir", "data/processed/nirt_runs")
    runs_dir = Path(runs_dir) if Path(runs_dir).is_absolute() else root / runs_dir

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

    out = runs_dir / args.run / f"eval_{args.split}.json"
    out.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
