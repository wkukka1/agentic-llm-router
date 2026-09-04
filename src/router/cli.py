"""Command line entry point: ``python -m router.cli <command>``.

    build-data     download RouterArena, dedupe, split, write parquet
    describe-data  split statistics
    train          run one or more experiments, write the leaderboard
    analyze        per-class precision/recall, confusion, error slices
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
    build_real_only_dataset,
    build_task_dataset,
    load_splits,
)
from router.experiment import ARTIFACTS_DIR, run_all
from router.external_eval import EXTERNAL_DIR, render, score

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


def cmd_build_real(args: argparse.Namespace) -> int:
    """Hand-labelled real prompts only -- what the shipped model trains on."""
    splits = build_real_only_dataset(merge_domains=args.merge_domains,
                                     out_dir=Path(args.out_dir), variant=args.variant,
                                     seed=args.seed)
    for name, frame in splits.items():
        print(f"\n{name}: {len(frame)} rows")
        print(frame["domain"].value_counts().to_string())
    return 0


def cmd_build_task(args: argparse.Namespace) -> int:
    """Build the task-head splits."""
    splits = build_task_dataset(
        include_synthetic=not args.no_synthetic,
        include_mined=args.include_mined,
        out_dir=Path(args.out_dir),
        variant=args.variant,
        seed=args.seed,
    )
    for name, frame in splits.items():
        print(f"{name}: {len(frame)} rows")
    return 0

def cmd_describe_data(args: argparse.Namespace) -> int:
    label = "domain"
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
                    continue_on_error=not args.fail_fast,
                    measure_latency=not args.no_latency)
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


def cmd_external(args: argparse.Namespace) -> int:
    """Score a trained run against externally-labelled prompt sets."""

    from router.inference import DomainHead

    head = DomainHead(args.run_dir, merge_domains=not args.no_merge, shortlist_size=2)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.data_dir).glob("*.parquet")):
        frame = pd.read_parquet(path)
        result = score(head, frame)
        text = render(result, path.stem)
        (out_dir / f"external_{path.stem}.md").write_text(text, encoding="utf-8")
        print(f"{path.stem:16} n={len(frame):4}  top1={result['top1']:.4f}  top2={result['top2']:.4f}")
    print(f"\nwrote reports to {out_dir}")
    return 0


def cmd_overfit(args: argparse.Namespace) -> int:
    """Audit both heads for overfitting and leakage."""
    from router.overfit import audit_heads

    results = audit_heads(args.encoder)
    for r in results:
        print(r.summary(), end="\n\n")
    failed = [r.name for r in results if not r.clean]
    if failed:
        print(f"NEEDS REVIEW: {', '.join(failed)}")
        return 1
    print("all heads clean")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="router", description="Domain classifier training harness")
    parser.add_argument("-v", "--verbose", action="count", default=1)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build-task", help="hand-labelled task splits (synthetic rows into train only)")
    build.add_argument("--no-synthetic", action="store_true",
                       help="train on the 1,000 real task labels alone")
    build.add_argument("--include-mined", action="store_true",
                       help="also add the 240 targeted-search rows (off by default: "
                            "macro-F1 +0.025, top-1 -0.013, neither significant)")
    build.add_argument("--variant", default="task")
    build.add_argument("--out-dir", default=str(PROCESSED_DIR))
    build.add_argument("--seed", type=int, default=20260902)
    build.set_defaults(func=cmd_build_task)


    build_real = sub.add_parser(
        "build-real",
        help="hand-labelled real prompts only -- the split the shipped model uses",
    )
    build_real.add_argument("--merge-domains", action="store_true",
                            help="collapse business_finance+law_politics and "
                                 "humanities+arts_entertainment (10 -> 8 classes)")
    build_real.add_argument("--variant", default="real_only")
    build_real.add_argument("--out-dir", default=str(PROCESSED_DIR))
    build_real.add_argument("--seed", type=int, default=20260827)
    build_real.set_defaults(func=cmd_build_real)

    describe = sub.add_parser("describe-data", help="print split statistics")
    describe.add_argument("--variant", default="domain_v3")
    describe.add_argument("--out-dir", default=str(PROCESSED_DIR))
    describe.set_defaults(func=cmd_describe_data)

    train = sub.add_parser("train", help="run one or more experiments")
    train.add_argument("config", nargs="+", help="experiment YAML files or directories")
    train.add_argument("--only", nargs="*", help="restrict to these experiment names")
    train.add_argument("--out-dir", default=str(ARTIFACTS_DIR))
    train.add_argument("--save-model", action="store_true")
    train.add_argument("--fail-fast", action="store_true")
    train.add_argument("--no-latency", action="store_true",
                       help="skip single-prompt latency timing; it runs 200 prompts one "
                            "at a time through every member and dominates runtime for "
                            "multi-encoder ensembles")
    train.set_defaults(func=cmd_train)

    external = sub.add_parser(
        "external",
        help="score against externally-labelled prompt sets (labels not written by this project)",
    )
    external.add_argument("run_dir", help="a trained run directory")
    external.add_argument("--data-dir", default=str(EXTERNAL_DIR))
    external.add_argument("--out-dir", default="artifacts/external")
    external.add_argument("--no-merge", action="store_true",
                          help="score in the 10-class space instead of the 8-class merged one")
    external.set_defaults(func=cmd_external)

    analyze = sub.add_parser("analyze", help="per-class precision/recall, confusion, error slices")
    analyze.add_argument("name", nargs="*", help="experiment name(s); defaults to the leaderboard leader")
    analyze.add_argument("--out-dir", default=str(ARTIFACTS_DIR))
    analyze.set_defaults(func=cmd_analyze)

    overfit = sub.add_parser(
        "overfit", help="train-test gap, learning curve, label permutation, "
                        "regularisation sensitivity and near-duplicate leakage")
    overfit.add_argument("--encoder", default="intfloat/e5-large-v2")
    overfit.set_defaults(func=cmd_overfit)


    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
