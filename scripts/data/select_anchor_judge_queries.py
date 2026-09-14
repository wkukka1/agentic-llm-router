"""Sample the stratified query set for the anchor-topology judge (docs/anchor_judge.md).

Splits RouterBench's 11-model correctness pool into a "discriminative"
stratum (correctness varies across the pool -- the queries with actual
routing signal) and a "natural" stratum (an unweighted sample of everything
else), and writes both as one frame with a ``stratum`` column. 0-/5-shot
twins are collapsed to the 0-shot item only (``training.cli.zeroshot_only``) so
the sample doesn't double-count the same underlying question.

    python scripts/data/select_anchor_judge_queries.py
"""

from __future__ import annotations

import sys

from training.cli import base_parser, get_config, info, write_json, zeroshot_only

# Kept in sync with scripts/data/select_anchor_model.py's ROUTERBENCH_11 --
# scripts/ isn't a package, so this is duplicated rather than cross-imported.
ROUTERBENCH_11 = [
    "gpt-4-1106-preview", "gpt-3.5-turbo-1106", "claude-v1", "claude-v2",
    "claude-instant-v1", "llama-2-70b-chat", "code-llama-34b-instruct",
    "mixtral-8x7b-instruct", "mistral-7b-instruct", "wizardlm-13b-v1.2", "yi-34b-chat",
]


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = get_config(args)
    acfg = cfg.get("anchor_judge")

    from training.data.facade import load_training_data
    from evaluation.data.stratify import sample_stratified_queries

    d = load_training_data(cfg)
    R = d.correctness_matrix(
        metric="accuracy", models="warm", combine_metrics=["accuracy", "mc_accuracy"],
    )
    R = R.reindex(columns=ROUTERBENCH_11).dropna(axis=0, how="any")
    R = zeroshot_only(R)

    n_discriminative = int(acfg.get("n_discriminative", 1750)) if acfg else 1750
    n_natural = int(acfg.get("n_natural", 750)) if acfg else 750
    seed = int(cfg.get("seed", 42))

    sampled = sample_stratified_queries(
        R, n_discriminative=n_discriminative, n_natural=n_natural, seed=seed,
    )
    out_dir = cfg.resolve(acfg.get("output_dir", "data/processed/anchor_judge") if acfg
                           else "data/processed/anchor_judge")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sampled_queries.parquet"
    sampled.to_parquet(out_path)

    counts = sampled["stratum"].value_counts().to_dict()
    info(f"sampled {len(sampled)} queries -> {out_path}  ({counts})")
    write_json(out_dir / "sampled_queries_summary.json", {
        "n_total": len(sampled), "counts_by_stratum": counts,
        "pool": ROUTERBENCH_11, "n_pool_complete_queries": len(R),
    }, root=cfg.root, announce=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
