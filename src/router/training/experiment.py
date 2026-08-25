"""Run one experiment end to end and write a self-describing result directory."""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from router.config import ExperimentConfig
from router.data.build import load_splits
from router.models import build as build_model
from router.training.calibration import apply_temperature, fit_temperature
from router.training.metrics import evaluate, measure_latency

log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts/domain")


@dataclass(slots=True)
class ExperimentResult:
    name: str
    config: dict[str, Any]
    val: dict[str, Any]
    test: dict[str, Any]
    runtime: dict[str, Any]

    def summary_row(self) -> dict[str, Any]:
        """The one-line-per-experiment view used by the comparison table."""
        return {
            "experiment": self.name,
            "model": self.config["model"]["name"],
            "variant": self.config["data"]["variant"],
            "val_macro_f1": self.val["macro_f1"],
            "test_accuracy": self.test["accuracy"],
            "test_macro_f1": self.test["macro_f1"],
            "test_balanced_acc": self.test["balanced_accuracy"],
            "test_top2_acc": self.test["top2_accuracy"],
            "test_ece": self.test["ece"],
            "test_ece_cal": self.test["ece_calibrated"],
            "temperature": self.test["temperature"],
            "test_acc@cov70": self.test["acc@coverage70"],
            "latency_ms_p50": self.runtime["latency_ms_p50"],
            "train_seconds": self.runtime["train_seconds"],
            "model_mb": self.runtime["model_mb"],
        }


def _subsample(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame
    return frame.sample(n=limit, random_state=seed).reset_index(drop=True)


def run_experiment(
    config: ExperimentConfig,
    *,
    out_dir: Path = ARTIFACTS_DIR,
    save_model: bool = False,
    measure_latency_on_test: bool = True,
) -> ExperimentResult:
    """Train, evaluate on val and test, and persist everything about the run."""
    splits = load_splits(config.data.variant)
    train = _subsample(splits["train"], config.data.max_train_rows, config.seed)
    val, test = splits["val"], splits["test"]

    text_col, label_col = config.data.text_column, config.data.label_column
    params = dict(config.model.params)
    if config.model.name.startswith("embed_"):
        # Frozen-encoder models reuse one embedding cache across experiments,
        # namespaced by dataset variant so a rebuild cannot serve stale vectors.
        params.setdefault("cache_tag", config.data.variant)

    log.info("=== %s (%s) | train=%d val=%d test=%d",
             config.name, config.model.name, len(train), len(val), len(test))

    model = build_model(config.model.name, **params)

    start = time.perf_counter()
    model.fit(
        train[text_col].tolist(), train[label_col].tolist(),
        val[text_col].tolist(), val[label_col].tolist(),
    )
    train_seconds = time.perf_counter() - start

    val_proba = model.predict_proba(val[text_col].tolist())
    test_proba = model.predict_proba(test[text_col].tolist())
    val_metrics = evaluate(val[label_col].tolist(), val_proba, model.labels)
    test_metrics = evaluate(test[label_col].tolist(), test_proba, model.labels)

    # Calibrate on validation only, then score the untouched test split. This is
    # free accuracy-wise (temperature scaling is monotonic) and is what makes the
    # router's confidence thresholds mean what they say.
    label_index = {label: i for i, label in enumerate(model.labels)}
    temperature = fit_temperature(
        val_proba, np.array([label_index[label] for label in val[label_col]])
    )
    calibrated_test = evaluate(
        test[label_col].tolist(), apply_temperature(test_proba, temperature), model.labels
    )
    test_metrics["temperature"] = temperature
    test_metrics["ece_calibrated"] = calibrated_test["ece"]
    test_metrics["log_loss_calibrated"] = calibrated_test["log_loss"]
    for coverage in (50, 70, 90):
        test_metrics[f"acc@coverage{coverage}_calibrated"] = calibrated_test[f"acc@coverage{coverage}"]

    runtime: dict[str, Any] = {
        "train_seconds": round(train_seconds, 2),
        "model_mb": round(model.size_bytes() / 1e6, 2),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if measure_latency_on_test:
        runtime.update(measure_latency(model.predict_proba, test[text_col].tolist()))
    else:
        runtime.update({"latency_ms_p50": float("nan"), "latency_ms_p95": float("nan"),
                        "latency_ms_mean": float("nan")})

    result = ExperimentResult(
        name=config.name, config=config.to_dict(),
        val=val_metrics, test=test_metrics, runtime=runtime,
    )

    run_dir = out_dir / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(_as_yaml(config.to_dict()))
    (run_dir / "metrics.json").write_text(json.dumps(
        {"val": val_metrics, "test": test_metrics, "runtime": runtime}, indent=2))
    _write_predictions(run_dir, test, test_proba, model.labels, text_col, label_col)
    if save_model:
        model.save(run_dir / "model")

    log.info("%s: test acc=%.4f macro_f1=%.4f ece=%.3f p50=%.2fms",
             config.name, test_metrics["accuracy"], test_metrics["macro_f1"],
             test_metrics["ece"], runtime["latency_ms_p50"])
    return result


def _write_predictions(run_dir: Path, test: pd.DataFrame, proba: np.ndarray,
                       labels: list[str], text_col: str, label_col: str) -> None:
    """Per-row predictions, so error analysis never requires a retrain."""
    frame = pd.DataFrame({
        "uid": test["uid"],
        "prompt": test[text_col].str.slice(0, 400),
        "subset": test.get("subset"),
        "difficulty": test.get("difficulty"),
        "y_true": test[label_col],
        "y_pred": [labels[i] for i in proba.argmax(axis=1)],
        "confidence": proba.max(axis=1),
    })
    for i, label in enumerate(labels):
        frame[f"p_{label}"] = proba[:, i]
    frame.to_parquet(run_dir / "test_predictions.parquet", index=False)


def _as_yaml(payload: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(payload, sort_keys=False)
