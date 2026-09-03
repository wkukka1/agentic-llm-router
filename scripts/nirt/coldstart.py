"""Cold-start experiment: place a held-out LLM from its profile embedding alone.

    python scripts/nirt/coldstart.py --run nirt-2d-projected --split test
    python scripts/nirt/coldstart.py --run irt-2d-projected --split test --also-free irt-2d-free

The cold-start LLMs (data/splits/cold_start_models.json) were held out of all
training. A `projected` run predicts their per-query correctness from their
profile text embedding only; we compare that to no-NIRT references
(global-mean / warm-mean / nearest-warm-model-by-profile) and check whether the
model places each LLM at the right skill rank.

Writes <runs_dir>/<run>/coldstart_<split>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from router.config import load_config
from router.data.phase1 import load_phase1
from router.nirt.evaluate import cold_start_eval
from router.nirt.train import load_run

pd.set_option("display.width", 160)


def _runs_dir(cfg_path: Path, root: Path) -> Path:
    c = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    d = Path(c.get("runs_dir", "data/processed/nirt_runs"))
    return d if d.is_absolute() else root / d


def _table(res: dict) -> pd.DataFrame:
    rows = []
    for m, r in res["per_model"].items():
        pm = r["prediction"]
        rows.append({
            "cold model": m, "n": r["n"],
            "bce": pm["bce"], "auc": pm["auc"], "acc@0.5": pm["acc@0.5"],
            "true": pm["pos_rate"], "pred": pm["pred_mean"],
            "nearest warm": r["nearest_warm_model"],
            "rank err": res["placement"][m]["mean_abs_rank_error"],
            "rank pred/true": f"{res['placement'][m]['mean_rank_pred']:.1f}/"
                              f"{res['placement'][m]['mean_rank_true']:.1f}",
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--config", default="configs/nirt.yaml")
    ap.add_argument("--phase0-config", default=None)
    ap.add_argument("--also-free", default=None, help="a 'free' run name to contrast (should fail)")
    args = ap.parse_args()

    root = Path(load_config().root)
    cfg_path = Path(args.config)
    cfg_path = cfg_path if cfg_path.is_absolute() else root / cfg_path
    runs_dir = _runs_dir(cfg_path, root)

    p0 = load_config(args.phase0_config) if args.phase0_config else load_config()
    d = load_phase1(p0)

    model, run_cfg, model_index = load_run(args.run, runs_dir=runs_dir)
    pathway = run_cfg.get("data", {}).get("pathway", "irt")
    res = cold_start_eval(model, model_index, split=args.split, pathway=pathway, data=d)

    print(f"\n=== cold-start ({res['orientation']}, run={args.run}, split={args.split}) ===")
    print(f"cold LLMs: {res['cold_models']}")
    print("\n-- per-model prediction + placement --")
    print(_table(res).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    pooled = res["pooled_prediction"]
    print("\n-- pooled: NIRT vs no-NIRT references (BCE / MSE) --")
    print(pd.DataFrame([
        {"method": k, "bce": v["bce"], "mse": v["mse"], "auc": v.get("auc", float("nan"))}
        for k, v in pooled.items()
    ]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n-- by benchmark family (does it know the specialisation?) --")
    for m, r in res["per_model"].items():
        fam = r["by_family"]
        if fam:
            print(f"  {m}: " + ", ".join(
                f"{k} true {v['true']:.2f}/pred {v['pred']:.2f}" for k, v in fam.items()))

    print("\n-- augmented-pool ranking (warm + cold) --")
    print("  " + json.dumps({k: round(v, 4) for k, v in res["augmented_ranking"].items()}))

    if args.also_free:
        print(f"\n-- contrast: --also-free {args.also_free} --")
        try:
            fm, fcfg, fidx = load_run(args.also_free, runs_dir=runs_dir)
            cold_start_eval(fm, fidx, split=args.split,
                            pathway=fcfg.get("data", {}).get("pathway", "irt"), data=d)
            print("  (unexpected: free run did not raise)")
        except ValueError as e:
            print(f"  free run correctly refused: {e}")

    out = runs_dir / args.run / f"coldstart_{args.split}.json"
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
