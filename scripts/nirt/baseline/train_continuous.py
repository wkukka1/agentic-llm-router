"""Train a Phase 2 continuous response model (or the Bernoulli control) on the
same representation + splits as Phase 1.

    python scripts/train_continuous.py --response normal
    python scripts/train_continuous.py --response beta   --epochs 40
    python scripts/train_continuous.py --response zoib

Writes artifacts/phase2/<response>/{model.pt, config.yaml, metrics.json,
training_history.json, provenance.json}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from router.config import load_config
from router.nirt.baseline.train import fit
from router.nirt.baseline.response_head import RESPONSE_MODELS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--response", required=True, choices=list(RESPONSE_MODELS))
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--phase0-config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--model-params", choices=["free", "projected"], default=None)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    root = load_config().root
    base = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    rm = base.pop("response_models", {})
    if args.response not in rm:
        rm[args.response] = {"model": args.response}
    cfg = {k: v for k, v in base.items() if k != "evaluation"}
    cfg["response"] = {**cfg.get("response", {}), **rm[args.response]}
    cfg["out_dir"] = args.checkpoint or f"artifacts/phase2/{args.response}"
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.lr is not None:
        cfg.setdefault("train", {})["lr"] = args.lr
    if args.model_params is not None:
        cfg.setdefault("model", {})["model_params"] = args.model_params
    if args.response == "bernoulli":
        cfg.setdefault("train", {})["target"] = "binary"

    p0 = load_config(args.phase0_config) if args.phase0_config else load_config()
    res = fit(cfg, phase0_cfg=p0)
    print(f"\n=== {args.response} — validation (best epoch {res.best_epoch}) ===")
    print(json.dumps({k: v for k, v in res.val_metrics.items() if isinstance(v, (int, float))},
                     indent=2, default=float))
    if res.path:
        print(f"checkpoint: {res.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
