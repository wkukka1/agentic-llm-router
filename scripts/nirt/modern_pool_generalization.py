"""MVP: does ANY trained NIRT variant's zero-shot profile placement
generalize to LLMs released after it was trained -- genuinely unseen
models (zero rows anywhere in this project's data), not split-forced-cold
ones?

Downloads real battles between modern frontier models restricted to
battles where BOTH sides are one of our target "brand-new" models
(``training.data.modern_pool_battles``), builds their profiles straight
from ``configs/model_profiles.yaml``, and scores EVERY ``projected`` NIRT
run on disk against the real outcomes.

    python scripts/nirt/modern_pool_generalization.py
    python scripts/nirt/modern_pool_generalization.py --runs nirt-1d-projected,nirt-4d-projected

Writes <runs_dir>/modern_pool_generalization.json.
"""

from __future__ import annotations

import sys


from training.cli import float_table, load_yaml, raw_parser, resolve, write_json
from training.data.modern_pool_battles import download_modern_pool_battles, modern_pool_profile_texts
from evaluation.nirt.modern_pool import sweep_projected_runs


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--config", default="configs/nirt.yaml")
    ap.add_argument("--runs", default=None,
                    help="comma-separated run names (default: every run.json under runs_dir)")
    ap.add_argument("--force-download", action="store_true",
                    help="re-fetch the battle data instead of using the cached parquet")
    ap.add_argument("--knn-k", type=int, default=None,
                    help="impute each modern-pool model's profile toward the k nearest models "
                         "in the SCORING RUN'S OWN trained model_index before scoring (default: "
                         "off, score the raw profile embedding)")
    ap.add_argument("--knn-weighting", default="similarity", choices=["similarity", "uniform"])
    args = ap.parse_args()

    ncfg = load_yaml(args.config)
    runs_dir = resolve(ncfg.get("runs_dir", "data/processed/nirt_runs"))

    if args.runs:
        run_names = args.runs.split(",")
    else:
        run_names = sorted(p.parent.name for p in runs_dir.glob("*/run.json"))

    print(f"scoring {len(run_names)} run(s) against the modern-pool battle set...")
    battles = download_modern_pool_battles(force=args.force_download)
    print(f"  {len(battles)} both-sides-new battles across "
          f"{sorted(set(battles['model_a']) | set(battles['model_b']))}")
    profile_texts = modern_pool_profile_texts()

    results = sweep_projected_runs(run_names, battles, profile_texts, runs_dir=runs_dir,
                                   knn_k=args.knn_k, knn_weighting=args.knn_weighting)
    ok = results[results.get("error").isna()] if "error" in results.columns else results
    ok = ok.sort_values("bce") if "bce" in ok.columns and not ok.empty else ok

    knn_tag = f" (knn_k={args.knn_k}, {args.knn_weighting})" if args.knn_k else " (raw profiles)"
    print(f"\n=== modern-pool generalization{knn_tag}: {len(ok)} scored run(s) "
          f"(ranked best -> worst BCE) ===")
    with float_table(160):
        cols = [c for c in ("run", "pathway", "dim", "n", "bce", "auc", "acc@0.5", "coin_flip_bce")
               if c in ok.columns]
        print(ok[cols].to_string(index=False) if not ok.empty else "  (no scoreable runs)")

    if "error" in results.columns:
        failed = results[results["error"].notna()]
        if not failed.empty:
            print(f"\n=== {len(failed)} run(s) skipped (error) ===")
            print(failed[["run", "error"]].to_string(index=False))

    suffix = f"_knn{args.knn_k}{args.knn_weighting[0]}" if args.knn_k else ""
    write_json(runs_dir / f"modern_pool_generalization{suffix}.json", results.to_dict(orient="records"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
