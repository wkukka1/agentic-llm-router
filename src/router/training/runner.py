"""Sweep a set of experiments and produce the comparison table."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from router.config import ExperimentConfig
from router.training.experiment import ARTIFACTS_DIR, ExperimentResult, run_experiment

log = logging.getLogger(__name__)

#: Columns of the leaderboard, in the order a reader should scan them.
SUMMARY_COLUMNS = [
    "experiment", "model", "variant",
    "test_accuracy", "test_macro_f1", "test_balanced_acc", "test_top2_acc",
    "test_ece", "test_ece_cal", "temperature", "test_acc@cov70", "val_macro_f1",
    "latency_ms_p50", "train_seconds", "model_mb",
]


def run_all(
    configs: list[ExperimentConfig],
    *,
    out_dir: Path = ARTIFACTS_DIR,
    save_model: bool = False,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """Run every config, writing the leaderboard after each one.

    Writing incrementally means a crash in experiment 7 does not cost you the
    six that already succeeded.
    """
    results: list[ExperimentResult] = []
    failures: dict[str, str] = {}

    for config in configs:
        try:
            results.append(run_experiment(config, out_dir=out_dir, save_model=save_model))
        except Exception as exc:  # noqa: BLE001 - a bad config must not kill the sweep
            log.exception("experiment %s failed", config.name)
            failures[config.name] = f"{type(exc).__name__}: {exc}"
            if not continue_on_error:
                raise
        if results:
            _write_summary(results, failures, out_dir)

    return _summary_frame(results)


def _summary_frame(results: list[ExperimentResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    frame = pd.DataFrame([r.summary_row() for r in results])
    return frame[SUMMARY_COLUMNS].sort_values("test_macro_f1", ascending=False).reset_index(drop=True)


def _write_summary(results: list[ExperimentResult], failures: dict[str, str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _summary_frame(results)
    frame.to_csv(out_dir / "leaderboard.csv", index=False)
    (out_dir / "leaderboard.md").write_text(render_markdown(frame, failures))
    if failures:
        (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))


def render_markdown(frame: pd.DataFrame, failures: dict[str, str] | None = None) -> str:
    """Human-readable leaderboard, sorted by macro-F1."""
    lines = ["# Domain classifier leaderboard", ""]
    if frame.empty:
        lines.append("_no successful runs_")
    else:
        display = frame.copy()
        for col in display.columns:
            if display[col].dtype.kind == "f":
                display[col] = display[col].map(lambda v: f"{v:.4f}")
        lines.append(display.to_markdown(index=False))
    if failures:
        lines += ["", "## Failures", ""]
        lines += [f"- `{name}`: {msg}" for name, msg in failures.items()]
    return "\n".join(lines) + "\n"
