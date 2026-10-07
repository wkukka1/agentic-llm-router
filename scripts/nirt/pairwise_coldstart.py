"""E0-E6: pairwise-battle generalization eval for NIRT's profile-only cold-start.

    python scripts/nirt/pairwise_coldstart.py --run nirt-1d-projected
    python scripts/nirt/pairwise_coldstart.py --run nirt-1d-projected --experiment E2 --experiment E4

Headline result: can publicly available capability information, excluding
Arena performance itself, initialise NIRT well enough to predict human Arena
preferences for models completely absent from training? E2 is that claim,
E4 is the Arena-Elo-excluded version that isolates "excluding Arena
performance itself", E3 explains why E2 lands where it does. E1 is a
both-sides-warm reference condition, E5 a permutation sanity check, E6 a
named historical case study (never pooled with E2/E4).

Writes <runs_dir>/<run>/eval_pairwise_coldstart.json.
"""

from __future__ import annotations

import sys

import pandas as pd
import yaml

from training.cli import float_table, raw_parser, resolve, write_json
from router.config import load_config
from router.nirt.checkpoint import load_run
from training.data.facade import load_training_data
from training.models.profiles import VARIANTS, build_variant_profile_embeddings, read_model_profiles
from evaluation.nirt.evaluate import cold_start_eval
from evaluation.nirt.pairwise_coldstart import assert_arena_blind, pairwise_cold_start_eval

ALL_EXPERIMENTS = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]


def _pairwise_table(res: dict) -> pd.DataFrame:
    rows = []
    for m, r in res["per_model"].items():
        pm = r["prediction"]
        if pm.get("n", 0) == 0:
            rows.append({"model": m, "n": 0})
            continue
        row = {"model": m, "n": pm["n"], "bce": pm["bce"], "auc": pm.get("auc"),
               "acc@0.5": pm.get("acc@0.5")}
        for bname, bm in r["baselines"].items():
            row[f"{bname}_bce"] = bm.get("bce")
        rows.append(row)
    return pd.DataFrame(rows)


def _print_pairwise(title: str, res: dict) -> None:
    print(f"\n=== {title} ===")
    if not res["per_model"]:
        print("  (no target models)")
        return
    with float_table(160):
        print(_pairwise_table(res).to_string(index=False))


def _permutation_override(cold_models: list[str]) -> dict[str, str]:
    """A derangement of ``cold_models`` -- each gets a different cold
    model's profile, never its own (E5)."""
    if len(cold_models) < 2:
        return {}
    shifted = cold_models[1:] + cold_models[:1]
    return dict(zip(cold_models, shifted))


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/nirt.yaml")
    ap.add_argument("--phase0-config", default=None)
    ap.add_argument("--experiment", action="append", choices=ALL_EXPERIMENTS,
                    default=None, help="repeatable; default: all of E0-E6")
    ap.add_argument("--models", default=None,
                    help="comma-separated cold-model override for E2-E6 (default: "
                         "data.cold_start_model_ids())")
    args = ap.parse_args()
    experiments = args.experiment or ALL_EXPERIMENTS

    ncfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    runs_dir = resolve(ncfg.get("runs_dir", "data/processed/nirt_runs"))

    p0 = load_config(args.phase0_config) if args.phase0_config else load_config()
    data = load_training_data(p0)

    model, run_cfg, model_index = load_run(args.run, runs_dir=runs_dir)
    pathway = run_cfg.get("data", {}).get("pathway", "irt")
    cold_models = args.models.split(",") if args.models else data.cold_start_model_ids()

    results: dict = {}

    if "E0" in experiments:
        results["E0"] = cold_start_eval(model, model_index, split="test", pathway=pathway, data=data)
        print("\n=== E0: pointwise cold-start baseline (existing, unchanged) ===")
        print(f"  cold models: {results['E0']['cold_models']}")
        print(f"  pooled NIRT bce: {results['E0']['pooled_prediction']['nirt']['bce']:.4f}")

    if "E1" in experiments:
        results["E1"] = pairwise_cold_start_eval(
            model, run_cfg, data, target_models=data.warm_model_ids(),
            sources=("chatbot_arena",), require_warm_opponent=False,
        )
        _print_pairwise("E1: unseen-pair Arena generalization (both warm -- reference)",
                        results["E1"])

    if "E2" in experiments:
        results["E2"] = pairwise_cold_start_eval(
            model, run_cfg, data, target_models=cold_models, sources=("chatbot_arena",),
        )
        _print_pairwise("E2: cold-model Arena generalization -- PRIMARY RESULT", results["E2"])

    if "E3" in experiments:
        results["E3"] = {}
        for variant in VARIANTS:
            store = build_variant_profile_embeddings(p0, pathway, variant)
            r = pairwise_cold_start_eval(
                model, run_cfg, data, target_models=cold_models, sources=("chatbot_arena",),
                profile_store_override=store,
            )
            results["E3"][variant] = r
            _print_pairwise(f"E3: profile ablation -- variant {variant}", r)

    if "E4" in experiments:
        profiles_df = read_model_profiles(p0)
        for m in cold_models:
            row = profiles_df.loc[profiles_df["model_id"] == m]
            if not row.empty:
                assert_arena_blind(row.iloc[0]["structured"].get("benchmarks", {}), "P4")
        store_p4 = build_variant_profile_embeddings(p0, pathway, "P4")
        results["E4"] = pairwise_cold_start_eval(
            model, run_cfg, data, target_models=cold_models, sources=("chatbot_arena",),
            profile_store_override=store_p4,
        )
        _print_pairwise("E4: Arena-blind cold-model evaluation (E2 vs E4 is the headline)",
                        results["E4"])

    if "E5" in experiments:
        results["E5"] = pairwise_cold_start_eval(
            model, run_cfg, data, target_models=cold_models, sources=("chatbot_arena",),
            profile_override=_permutation_override(cold_models),
        )
        _print_pairwise("E5: profile permutation sanity check (expect collapse toward baselines)",
                        results["E5"])

    if "E6" in experiments and "mixtral-8x7b-instruct" in cold_models:
        results["E6"] = pairwise_cold_start_eval(
            model, run_cfg, data, target_models=["mixtral-8x7b-instruct"], sources=("gpt4_judge",),
        )
        print("\n*** E6 is ONE historical matchup (mixtral-8x7b-instruct vs "
              "gpt-4-1106-preview, 109K battles) -- a named case study, NOT a "
              "generalization claim across models. Never pooled with E2/E4. ***")
        _print_pairwise("E6: GPT4-judge Mixtral case study", results["E6"])

    write_json(runs_dir / args.run / "eval_pairwise_coldstart.json", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
