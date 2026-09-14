"""Train the Phase 1 plain Bernoulli / BCE NIRT baseline.

    python scripts/train_baseline.py --config configs/phase1.yaml
    python scripts/train_baseline.py --theta-dim 4 --model-params projected
    python scripts/train_baseline.py --resume artifacts/phase1/baseline

Writes artifacts/phase1/baseline/{model.pt, config.yaml, metrics.json,
training_history.json, provenance.json}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from router.config import load_config
from training.nirt.baseline.train import fit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--phase0-config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--learning-rate", "--lr", dest="lr", type=float, default=None)
    ap.add_argument("--theta-dim", type=int, default=None)
    ap.add_argument("--model-params", choices=["free", "projected"], default=None)
    ap.add_argument("--no-relevance", action="store_true")
    ap.add_argument("--no-interaction", action="store_true")
    ap.add_argument("--warmup", action="store_true", help="enable the FAISS warm-up blend")
    ap.add_argument("--checkpoint", default=None, help="output dir (overrides out_dir)")
    ap.add_argument("--resume", default=None, help="checkpoint dir to warm-start weights from")
    args = ap.parse_args()

    root = load_config().root
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("model", {}); cfg.setdefault("train", {}); cfg.setdefault("ablation", {})

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.theta_dim is not None:
        cfg["model"]["theta_dim"] = args.theta_dim
    if args.model_params is not None:
        cfg["model"]["model_params"] = args.model_params
    if args.no_relevance:
        cfg["ablation"]["use_relevance"] = False
    if args.no_interaction:
        cfg["ablation"]["use_interaction"] = False
    if args.warmup:
        cfg["ablation"]["use_warmup"] = True
        cfg["model"].setdefault("warmup", {})["enabled"] = True
    if args.checkpoint:
        cfg["out_dir"] = args.checkpoint

    p0 = load_config(args.phase0_config) if args.phase0_config else load_config()
    res = fit(cfg, phase0_cfg=p0, resume_from=args.resume)

    print("\n=== validation (best epoch %d) ===" % res.best_epoch)
    print(json.dumps({k: v for k, v in res.val_metrics.items() if isinstance(v, (int, float))},
                     indent=2, default=float))
    print("\n=== train imbalance ===")
    imb = res.train_imbalance
    print(json.dumps({k: imb[k] for k in ("n", "positive_rate", "negative_rate", "n_models",
                                          "n_queries", "obs_per_query")}, indent=2, default=float))
    if res.path:
        print(f"\ncheckpoint: {res.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
