"""Command line entry point: ``python -m router.cli <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from router.agent import AgentConfig, AgenticRouter, PolicyThresholds, Session
from router.agent.backends import default_backend
from router.agent.tasktype import infer_task_type
from router.config import load_experiments
from router.data.build import PROCESSED_DIR, build_domain_dataset, load_splits
from router.training.analysis import report
from router.training.experiment import ARTIFACTS_DIR
from router.training.runner import run_all

#: Prompt-rendering variants the dataset builder knows how to produce.
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
            **DATA_VARIANTS[variant],
            variant=variant,
            out_dir=Path(args.out_dir),
            seed=args.seed,
        )
        sizes = ", ".join(f"{k}={len(v)}" for k, v in splits.items())
        print(f"{variant}: {sizes}")
    return 0


def cmd_describe_data(args: argparse.Namespace) -> int:
    splits = load_splits(args.variant, Path(args.out_dir))
    for name, frame in splits.items():
        print(f"\n=== {name} ({len(frame)} rows) ===")
        print(frame["domain"].value_counts().to_string())
        print("-- difficulty --")
        print(frame["difficulty"].value_counts(dropna=False).to_string())
        print("-- task_type --")
        print(frame["task_type"].value_counts().to_string())
        print(f"-- prompt chars: median={frame['n_chars'].median():.0f} p95={frame['n_chars'].quantile(0.95):.0f}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    configs = load_experiments(args.config)
    if args.only:
        wanted = set(args.only)
        configs = [c for c in configs if c.name in wanted]
        missing = wanted - {c.name for c in configs}
        if missing:
            raise SystemExit(f"no config found for: {sorted(missing)}")
    if not configs:
        raise SystemExit("no experiment configs matched")

    print(f"running {len(configs)} experiment(s): {', '.join(c.name for c in configs)}\n")
    frame = run_all(
        configs,
        out_dir=Path(args.out_dir),
        save_model=args.save_model,
        continue_on_error=not args.fail_fast,
    )
    print("\n" + frame.to_string(index=False))
    print(f"\nwrote {Path(args.out_dir) / 'leaderboard.md'}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    names = args.name
    if not names:
        # Default to the leaderboard leader, which is what you want 90% of the time.
        leaderboard = out_dir / "leaderboard.csv"
        if not leaderboard.exists():
            raise SystemExit(f"no runs found under {out_dir}; run `train` first")
        import pandas as pd

        names = [pd.read_csv(leaderboard).iloc[0]["experiment"]]

    for name in names:
        text = report(name, out_dir)
        destination = out_dir / name / "analysis.md"
        destination.write_text(text)
        print(text)
        print(f"\nwrote {destination}")
    return 0


def _domain_fn_from(run_dir: str | None):
    """Load a trained domain head, or fall back to a clearly-marked uniform prior."""
    if run_dir:
        from router.inference import DomainHead, as_domain_fn

        return as_domain_fn(DomainHead(run_dir)), f"trained head from {run_dir}"

    from router.data.taxonomy import DOMAIN_LABELS

    uniform = {label: 1 / len(DOMAIN_LABELS) for label in DOMAIN_LABELS}

    def stub(_prompt: str):
        # Deliberately maximally-unconfident: with no head, every prompt is
        # ambiguous, and the policy's ambiguity rule is what should decide.
        return DOMAIN_LABELS[0], 1 / len(DOMAIN_LABELS), uniform

    return stub, "NO TRAINED HEAD (uniform prior) - pass --head to use one"


def cmd_route(args: argparse.Namespace) -> int:
    domain_fn, provenance = _domain_fn_from(args.head)
    router = AgenticRouter(
        domain_fn=domain_fn,
        task_type_fn=infer_task_type,
        backend=default_backend(args.backend_model),
        thresholds=PolicyThresholds(cost_weight=args.cost_weight),
        config=AgentConfig(mode=args.mode, max_cost_per_turn=args.max_cost),
    )
    print(f"domain signal: {provenance}")
    print("task-type signal: keyword heuristic (placeholder)")
    print(f"backend: {type(router.backend).__name__}  mode: {args.mode}\n")

    session = Session()
    for prompt in args.prompt:
        result = router.route(prompt, session)
        print("=" * 78)
        print(f"PROMPT: {prompt}")
        print(result.decision.explain())
        if result.needs_user_input:
            for question in result.clarifying_questions:
                print(f"  ASK USER: {question}")
        for sub in result.sub_queries:
            print(f"    [{sub.index}] -> {sub.assigned_model} "
                  f"skills={sub.skills} deps={sub.depends_on} :: {sub.text[:60]}")
        if result.answer:
            print(f"\nANSWER:\n{result.answer}")
    print("=" * 78)
    print(f"{len(session.turns)} turn(s), estimated total ${session.total_cost:.4f}")
    return 0


def cmd_route_eval(args: argparse.Namespace) -> int:
    from router.agent.evaluate import evaluate_routing

    splits = load_splits(args.variant, Path(args.data_dir))
    frame = splits[args.split]
    if args.limit:
        frame = frame.head(args.limit)

    domain_fn, provenance = _domain_fn_from(args.head)
    thresholds = PolicyThresholds(cost_weight=args.cost_weight)
    router = AgenticRouter(
        domain_fn=domain_fn,
        task_type_fn=infer_task_type,
        thresholds=thresholds,
        config=AgentConfig(max_cost_per_turn=args.max_cost),
    )

    if args.calibrate_thresholds:
        # Fit on the *train* split so the test split still measures a policy
        # that has not seen it.
        from router.agent.policy import fit_thresholds_to_quantiles

        calib = splits["train"]["prompt"].head(args.calibration_size).tolist()
        difficulties = [router.signals_for(p).difficulty for p in calib]
        thresholds = fit_thresholds_to_quantiles(
            difficulties,
            strong_quantile=args.strong_quantile,
            orchestrate_quantile=args.orchestrate_quantile,
            base=thresholds,
        )
        router.policy.thresholds = thresholds
        print(f"calibrated thresholds on {len(calib)} train prompts: "
              f"strong>={thresholds.strong_difficulty:.3f} "
              f"orchestrate>={thresholds.orchestrate_difficulty:.3f}")

    print(f"domain signal: {provenance}")
    print(f"routing {len(frame)} prompts from {args.variant}/{args.split}...\n")

    report = evaluate_routing(router, frame["prompt"].tolist())
    print(report.render())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.per_prompt.to_parquet(out_dir / "routing_decisions.parquet", index=False)
    (out_dir / "routing_report.md").write_text(report.render())
    print(f"wrote {out_dir / 'routing_report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="router", description="Agentic LLM router training harness")
    parser.add_argument("-v", "--verbose", action="count", default=1)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-data", help="download sources and write train/val/test splits")
    build.add_argument("--variant", default="full_prompt",
                       help=f"one of {list(DATA_VARIANTS)} or 'all'")
    build.add_argument("--out-dir", default=str(PROCESSED_DIR))
    build.add_argument("--seed", type=int, default=20260824)
    build.set_defaults(func=cmd_build_data)

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

    route = sub.add_parser("route", help="route prompts through the agentic pipeline")
    route.add_argument("prompt", nargs="+", help="one or more prompts, routed as a session")
    route.add_argument("--head", help="path to a trained run directory, e.g. artifacts/domain/07_finetune_minilm")
    route.add_argument("--mode", choices=["plan", "execute"], default="plan",
                       help="'plan' routes without generating answers; 'execute' calls the models")
    route.add_argument("--backend-model", help="LiteLLM model for gate/decompose/answer; omit for heuristics")
    route.add_argument("--cost-weight", type=float, default=0.5, help="0 = ignore cost, 1+ = cost dominates")
    route.add_argument("--max-cost", type=float, default=None, help="downgrade any turn costing more than this")
    route.set_defaults(func=cmd_route)

    route_eval = sub.add_parser("route-eval", help="score the routing policy over a split")
    route_eval.add_argument("--head", help="trained run directory for the domain signal")
    route_eval.add_argument("--variant", default="full_prompt")
    route_eval.add_argument("--split", default="test", choices=["train", "val", "test"])
    route_eval.add_argument("--data-dir", default=str(PROCESSED_DIR))
    route_eval.add_argument("--out-dir", default="artifacts/routing")
    route_eval.add_argument("--limit", type=int, default=None)
    route_eval.add_argument("--cost-weight", type=float, default=0.5)
    route_eval.add_argument("--max-cost", type=float, default=None)
    route_eval.add_argument("--calibrate-thresholds", action="store_true",
                            help="fit difficulty thresholds to quantiles of the train split")
    route_eval.add_argument("--calibration-size", type=int, default=1000)
    route_eval.add_argument("--strong-quantile", type=float, default=0.70,
                            help="share of traffic kept on the weak path")
    route_eval.add_argument("--orchestrate-quantile", type=float, default=0.92)
    route_eval.set_defaults(func=cmd_route_eval)

    analyze = sub.add_parser("analyze", help="error analysis for a finished run")
    analyze.add_argument("name", nargs="*", help="experiment name(s); defaults to the leaderboard leader")
    analyze.add_argument("--out-dir", default=str(ARTIFACTS_DIR))
    analyze.set_defaults(func=cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
