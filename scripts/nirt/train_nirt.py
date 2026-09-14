"""Train the NIRT baseline predictor (Phase 2, v1).

    python scripts/nirt/train_nirt.py --config configs/nirt.yaml
    python scripts/nirt/train_nirt.py --dim 2 --model-params free --name nirt-2d-free
    python scripts/nirt/train_nirt.py --model-params projected --epochs 15 --name nirt-1d-proj

Reads configs/nirt.yaml, applies any CLI overrides, trains through the Phase 1
facade, and writes model.pt + run.json under <runs_dir>/<name>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from router.config import load_config
from training.nirt.train import fit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/nirt.yaml", help="NIRT config yaml")
    ap.add_argument("--phase0-config", default=None, help="path to phase0.yaml (data pipeline)")
    ap.add_argument("--name", default=None, help="run name (default: nirt-<K>d-<mode>)")
    ap.add_argument("--dim", type=int, default=None, help="latent ability dimensions K")
    ap.add_argument("--orientation", choices=["query_latent", "model_latent"], default=None,
                    help="query_latent (ours) | model_latent (IRT-Router)")
    ap.add_argument("--model-params", choices=["projected", "free"], default=None)
    ap.add_argument("--query-hidden", default=None, help="int, or 'none' for a linear query head")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--loss", choices=["soft_bce", "hard_bce", "mse"], default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--query-features", default=None,
                    help="structured feature variant to concat to e_q (see "
                         "scripts/embeddings/build_query_features.py); "
                         "'none' to force off")
    ap.add_argument("--ood", action="store_true",
                    help="hold out evaluation.ood_holdout_families from training")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(load_config().root) / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})

    if args.dim is not None:
        cfg["model"]["dim"] = args.dim
    if args.orientation is not None:
        cfg["model"]["orientation"] = args.orientation
    if args.model_params is not None:
        cfg["model"]["model_params"] = args.model_params
    if args.query_hidden is not None:
        cfg["model"]["query_hidden"] = None if args.query_hidden.lower() == "none" else int(args.query_hidden)
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.loss is not None:
        cfg["train"]["loss"] = args.loss
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.weight_decay is not None:
        cfg["train"]["weight_decay"] = args.weight_decay
    if args.query_features is not None:
        cfg.setdefault("data", {})["query_features"] = (
            None if args.query_features.lower() == "none" else args.query_features
        )

    p0 = load_config(args.phase0_config) if args.phase0_config else load_config()

    datasets = None
    if args.ood:
        from training.data.facade import load_training_data
        from evaluation.nirt.ood import ood_datasets, ood_families

        fams = ood_families(cfg)
        d = load_training_data(p0)
        tr, va, _ = ood_datasets(d, fams, pathway=cfg.get("data", {}).get("pathway", "irt"))
        datasets = (tr, va)
        cfg.setdefault("data", {})["ood_holdout_families"] = list(fams)
        print(f"[ood] held out {list(fams)}: train {len(tr):,} / val {len(va):,} obs")

    res = fit(cfg, phase0_cfg=p0, datasets=datasets, name=args.name)

    print("\n=== validation (best epoch %d) ===" % res.best_epoch)
    print(json.dumps(res.val_metrics, indent=2, default=float))
    print("\n=== marginal baselines ===")
    print(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "per_model"}
         for k, v in res.baselines.items()},
        indent=2, default=float,
    ))
    if res.path:
        print(f"\nrun dir: {res.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
