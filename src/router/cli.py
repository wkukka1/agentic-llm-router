"""Command line entry point: ``python -m router.cli <command>``.

    build-data     download RouterArena, dedupe, split, write parquet
    describe-data  split statistics
    train          run one or more experiments, write the leaderboard
    analyze        per-class precision/recall, confusion, error slices
    diagnose       feature correlation: VIF, PCA structure, redundancy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from router.analysis import report
from router.config import load_experiments
from router.dataset import (
    PROCESSED_DIR,
    build_capability_dataset,
    build_domain_dataset,
    load_splits,
)
from router.experiment import ARTIFACTS_DIR, run_all

#: Prompt-rendering variants the builder can produce. How the RouterArena
#: fields are reassembled is a real experimental axis: option blocks and
#: context headers are format artifacts a classifier will latch onto instead
#: of the topic.
DATA_VARIANTS = {
    "full_prompt": {"include_context": True, "include_options": True},
    "question_only": {"include_context": False, "include_options": False},
    "no_options": {"include_context": True, "include_options": False},
}


def _configure_logging(verbosity: int) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbosity > 1 else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "filelock", "httpx", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_build_data(args: argparse.Namespace) -> int:
    variants = list(DATA_VARIANTS) if args.variant == "all" else [args.variant]
    for variant in variants:
        if variant not in DATA_VARIANTS:
            raise SystemExit(f"unknown variant {variant!r}; expected one of {list(DATA_VARIANTS)} or 'all'")
        splits = build_domain_dataset(
            **DATA_VARIANTS[variant], variant=variant,
            out_dir=Path(args.out_dir), seed=args.seed,
        )
        print(f"{variant}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    return 0


def cmd_build_capability(args: argparse.Namespace) -> int:
    splits = build_capability_dataset(
        lmarena_shards=args.lmarena_shards,
        include_routerarena=not args.no_routerarena,
        out_dir=Path(args.out_dir), variant=args.variant, seed=args.seed,
    )
    for name, frame in splits.items():
        print(f"\n{name}: {len(frame)} rows")
        print(pd.crosstab(frame["capability"], frame["source"]).to_string())
    return 0


def cmd_describe_data(args: argparse.Namespace) -> int:
    label = "capability" if args.variant.startswith("capability") else "domain"
    for name, frame in load_splits(args.variant, Path(args.out_dir)).items():
        print(f"\n=== {name} ({len(frame)} rows) ===")
        print(frame[label].value_counts().to_string())
        print("-- difficulty --")
        print(frame["difficulty"].value_counts(dropna=False).to_string())
        print(f"-- prompt chars: median={frame['n_chars'].median():.0f} "
              f"p95={frame['n_chars'].quantile(0.95):.0f}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    configs = load_experiments(args.config)
    if args.only:
        wanted = set(args.only)
        configs = [c for c in configs if c.name in wanted]
        if missing := wanted - {c.name for c in configs}:
            raise SystemExit(f"no config found for: {sorted(missing)}")
    if not configs:
        raise SystemExit("no experiment configs matched")

    print(f"running {len(configs)} experiment(s): {', '.join(c.name for c in configs)}\n")
    frame = run_all(configs, out_dir=Path(args.out_dir), save_model=args.save_model,
                    continue_on_error=not args.fail_fast)
    print("\n" + frame.to_string(index=False))
    print(f"\nwrote {Path(args.out_dir) / 'leaderboard.md'}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    names = args.name
    if not names:
        leaderboard = out_dir / "leaderboard.csv"
        if not leaderboard.exists():
            raise SystemExit(f"no runs found under {out_dir}; run `train` first")
        import pandas as pd

        names = [pd.read_csv(leaderboard).iloc[0]["experiment"]]

    for name in names:
        text = report(name, out_dir)
        destination = out_dir / name / "analysis.md"
        destination.write_text(text, encoding="utf-8")
        print(text)
        print(f"\nwrote {destination}")
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Feature-correlation diagnostics: which dimensions carry redundant signal."""
    from router.embeddings import EmbeddingEncoder
    from router.reduction import correlation_report

    splits = load_splits(args.variant, Path(args.data_dir))
    frame = splits["train"]
    if args.limit:
        frame = frame.head(args.limit)

    encoder = EmbeddingEncoder(args.encoder, pooling=args.pooling, max_length=args.max_length)
    print(f"encoding {len(frame)} prompts with {args.encoder}...")
    features = encoder.encode_cached(frame["prompt"].tolist(), tag=f"{args.variant}/diagnose")

    text = correlation_report(features, vif_threshold=args.vif_threshold)
    print("\n" + text)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "feature_diagnostics.md").write_text(text, encoding="utf-8")
    print(f"wrote {out / 'feature_diagnostics.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="router", description="Domain classifier training harness")
    parser.add_argument("-v", "--verbose", action="count", default=1)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-data", help="download, dedupe, split, write parquet")
    build.add_argument("--variant", default="full_prompt", help=f"one of {list(DATA_VARIANTS)} or 'all'")
    build.add_argument("--out-dir", default=str(PROCESSED_DIR))
    build.add_argument("--seed", type=int, default=20260824)
    build.set_defaults(func=cmd_build_data)

    build_cap = sub.add_parser(
        "build-capability",
        help="build capability splits from real LMArena prompts + RouterArena benchmarks",
    )
    build_cap.add_argument("--lmarena-shards", type=int, default=3,
                           help="each shard is ~10k English prompts")
    build_cap.add_argument("--no-routerarena", action="store_true",
                           help="train on real prompts only")
    build_cap.add_argument("--variant", default="capability")
    build_cap.add_argument("--out-dir", default=str(PROCESSED_DIR))
    build_cap.add_argument("--seed", type=int, default=20260826)
    build_cap.set_defaults(func=cmd_build_capability)

    describe = sub.add_parser("describe-data", help="print split statistics")
    describe.add_argument("--variant", default="full_prompt")
    describe.add_argument("--out-dir", default=str(PROCESSED_DIR))
    describe.set_defaults(func=cmd_describe_data)

    train = sub.add_parser("train", help="run one or more experiments")
    train.add_argument("config", nargs="+", help="experiment YAML files or directories")
    train.add_argument("--only", nargs="*", help="restrict to these experiment names")
    train.add_argument("--out-dir", default=str(ARTIFACTS_DIR))
    train.add_argument("--save-model", action="store_true")
    train.add_argument("--fail-fast", action="store_true")
    train.set_defaults(func=cmd_train)

    analyze = sub.add_parser("analyze", help="per-class precision/recall, confusion, error slices")
    analyze.add_argument("name", nargs="*", help="experiment name(s); defaults to the leaderboard leader")
    analyze.add_argument("--out-dir", default=str(ARTIFACTS_DIR))
    analyze.set_defaults(func=cmd_analyze)

    diagnose = sub.add_parser("diagnose", help="feature correlation: VIF, PCA structure, redundancy")
    diagnose.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    diagnose.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    diagnose.add_argument("--max-length", type=int, default=256)
    diagnose.add_argument("--variant", default="full_prompt")
    diagnose.add_argument("--data-dir", default=str(PROCESSED_DIR))
    diagnose.add_argument("--out-dir", default=str(ARTIFACTS_DIR))
    diagnose.add_argument("--vif-threshold", type=float, default=10.0)
    diagnose.add_argument("--limit", type=int, default=2000)
    diagnose.set_defaults(func=cmd_diagnose)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
