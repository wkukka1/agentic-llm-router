"""Error analysis over the per-row predictions each experiment writes.

The leaderboard answers "which model". This answers "where does it break", which
is what determines whether the router can trust the head or has to hedge:
confusion structure, behaviour across difficulty, and the risk/coverage curve
that sets the escalation threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.experiment import ARTIFACTS_DIR

#: Metrics that :func:`top_up_metrics` must not write back over a stored run.
#: ``temperature`` and everything derived from it were fitted on the
#: *validation* split by the experiment runner; recomputing here would refit on
#: test and quietly relabel a test-fitted number as the original. The accuracy
#: family is excluded for a different reason -- see :func:`top_up_metrics`.
_NOT_RECOMPUTABLE = frozenset({
    "temperature", "ece_calibrated", "log_loss_calibrated",
    "acc@coverage50_calibrated", "acc@coverage70_calibrated", "acc@coverage90_calibrated",
})

#: Metrics that depend on the model's decision rule rather than on the
#: probability matrix alone. ``evaluate`` derives them by argmax, which is only
#: the same thing when the model actually predicts by argmax -- a head with a
#: threshold, a shortlist rule or a prior correction disagrees. The stored
#: values came from the real ``y_pred``, so they win.
_DECISION_DEPENDENT = frozenset({
    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "per_class",
})


def load_run(name: str, out_dir: Path = ARTIFACTS_DIR) -> tuple[pd.DataFrame, dict]:
    """Load one run's test predictions and metrics.

    Metrics missing from an older run are topped up from the stored per-row
    probabilities rather than requiring a retrain, but only the ones that are
    genuinely pure functions of that matrix -- see :func:`top_up_metrics`.
    """
    run_dir = out_dir / name
    predictions = pd.read_parquet(run_dir / "test_predictions.parquet")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    test = metrics.setdefault("test", {})
    if "per_class" not in test and "labels" in test:
        top_up_metrics(test, predictions)
    return predictions, metrics


def top_up_metrics(test: dict, predictions: pd.DataFrame) -> dict:
    """Fill in *missing* metrics from stored probabilities; overwrite nothing.

    Three things this deliberately does not do, each of which the obvious
    ``test.update(evaluate(...))`` would:

    * **Overwrite decision-dependent metrics.** ``evaluate`` takes the argmax of
      the probability matrix. The stored ``accuracy`` came from the model's own
      ``y_pred``, which is only the same thing for a plain-argmax head. Letting
      the recomputed value win makes ``accuracy`` disagree with the confusion
      matrix in the same report.
    * **Refit calibration on test.** ``temperature`` and its derived fields were
      fitted on validation. Recomputing them here fits them on the test split
      and stores the result under the same name, so a leaked number becomes
      indistinguishable from an honest one.
    * **Assume the run has a ``test`` block at all.**

    Returns the keys it added, so a caller can see what was inferred rather
    than measured.
    """
    from training.metrics import evaluate

    labels = test.get("labels")
    columns = [f"p_{label}" for label in labels or []]
    if not labels or any(c not in predictions.columns for c in columns):
        return {}

    recomputed = evaluate(predictions["y_true"].tolist(), predictions[columns].to_numpy(), labels)
    added = {
        k: v for k, v in recomputed.items()
        if k not in test and k not in _NOT_RECOMPUTABLE and k not in _DECISION_DEPENDENT
    }
    test.update(added)
    return added


def per_class_table(test_metrics: dict) -> pd.DataFrame:
    """Precision, recall, F1 and support per class, worst F1 first.

    ``leak_direction`` names the dominant failure mode so the table can be read
    without cross-referencing the confusion matrix.
    """
    per_class = test_metrics.get("per_class")
    if not per_class:
        return pd.Series(test_metrics.get("per_class_f1", {})).to_frame("f1")

    frame = pd.DataFrame(per_class).T
    frame["support"] = frame["support"].astype(int)

    def direction(row):
        gap = row["precision"] - row["recall"]
        if abs(gap) < 0.03:
            return "balanced"
        return "leaks out" if gap > 0 else "absorbs others"

    frame["leak_direction"] = frame.apply(direction, axis=1)
    return frame.sort_values("f1").round(3)


def confusion_table(predictions: pd.DataFrame, labels: list[str] | None = None) -> pd.DataFrame:
    """Row-normalised confusion matrix (rows = true, values = share of row).

    Both axes span the *union* of true and predicted labels, or ``labels`` if
    given. Using the true labels for both -- the obvious shortcut, since the
    matrix is usually square anyway -- silently deletes any class the model
    predicts but never actually occurs, and because rows are normalised after
    the reindex the remaining mass renormalises to 1. The table still looks
    well-formed with the evidence removed.

    That failure lands precisely on the classes worth watching. `extract` has
    three rows in the 1,000-prompt task eval, so a run where the model finally
    starts predicting it -- the thing you would want to see -- is also a run
    where those predictions disappear from the matrix.
    """
    counts = pd.crosstab(predictions["y_true"], predictions["y_pred"])
    axis = list(labels) if labels else sorted(set(counts.index) | set(counts.columns))
    counts = counts.reindex(index=axis, columns=axis, fill_value=0)
    return counts.div(counts.sum(axis=1).clip(lower=1), axis=0)


def top_confusions(predictions: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """The k most frequent (true, predicted) mistakes, with mean confidence.

    High-confidence mistakes are the dangerous ones: the router cannot detect
    them by thresholding, so they need either a label-space fix or a fallback.
    """
    errors = predictions[predictions["y_true"] != predictions["y_pred"]]
    if errors.empty:
        return pd.DataFrame(columns=["y_true", "y_pred", "count", "mean_confidence"])
    grouped = (
        errors.groupby(["y_true", "y_pred"])
        .agg(count=("uid", "size"), mean_confidence=("confidence", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return grouped.head(k).reset_index(drop=True)


def breakdown(predictions: pd.DataFrame, by: str) -> pd.DataFrame:
    """Accuracy and mean confidence sliced by any column of the test split."""
    if by not in predictions.columns:
        return pd.DataFrame()
    frame = predictions.copy()
    frame["correct"] = (frame["y_true"] == frame["y_pred"]).astype(float)
    return (
        frame.groupby(by, dropna=False)
        .agg(n=("uid", "size"), accuracy=("correct", "mean"), mean_confidence=("confidence", "mean"))
        .sort_values("n", ascending=False)
        .reset_index()
    )


def risk_coverage_curve(predictions: pd.DataFrame, steps: int = 11) -> pd.DataFrame:
    """Accuracy vs coverage as the confidence threshold sweeps.

    Read this to pick the router's escalation threshold: the coverage at which
    accuracy is high enough to route directly, with the remainder deferred to a
    stronger model or a clarifying follow-up.
    """
    frame = predictions.sort_values("confidence", ascending=False).reset_index(drop=True)
    correct = (frame["y_true"] == frame["y_pred"]).to_numpy(dtype=float)
    rows = []
    for coverage in np.linspace(0.1, 1.0, steps):
        keep = max(1, int(round(len(frame) * coverage)))
        rows.append({
            "coverage": round(float(coverage), 3),
            "n_routed": keep,
            "accuracy": float(correct[:keep].mean()),
            "min_confidence": float(frame.loc[keep - 1, "confidence"]),
        })
    return pd.DataFrame(rows)


def hardest_examples(predictions: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    """Confidently-wrong rows, worst first -- the ones worth reading by hand."""
    errors = predictions[predictions["y_true"] != predictions["y_pred"]]
    return (
        errors.sort_values("confidence", ascending=False)
        .head(k)[["y_true", "y_pred", "confidence", "subset", "difficulty", "prompt"]]
        .reset_index(drop=True)
    )


def report(name: str, out_dir: Path = ARTIFACTS_DIR) -> str:
    """Render the full markdown analysis for one run."""
    predictions, metrics = load_run(name, out_dir)
    test = metrics["test"]

    lines = [
        f"# Error analysis: `{name}`",
        "",
        f"- test accuracy: **{test['accuracy']:.4f}**  |  macro-F1: **{test['macro_f1']:.4f}**"
        f"  |  top-2: {test['top2_accuracy']:.4f}",
        f"- macro precision {test.get('macro_precision', float('nan')):.4f}"
        f"  |  macro recall {test.get('macro_recall', float('nan')):.4f}",
        f"- ECE raw {test['ece']:.4f} -> calibrated {test.get('ece_calibrated', float('nan')):.4f}"
        f" (T={test.get('temperature', float('nan')):.3f})",
        f"- single-prompt latency p50 {metrics['runtime']['latency_ms_p50']:.2f} ms",
        "",
        "## Per-class precision / recall / F1",
        "",
        "Low **recall** = prompts of this class leak out to other classes.",
        "Low **precision** = other classes leak *in*. F1 alone hides which.",
        "",
        per_class_table(test).to_markdown(),
        "",
        "## Confusion (row-normalised, rows = true label)",
        "",
        # The run's own label list, so the matrix spans the full label space
        # even for classes that neither occur nor get predicted in this split.
        confusion_table(predictions, test.get("labels")).round(3).to_markdown(),
        "",
        "## Most frequent confusions",
        "",
        top_confusions(predictions).round(3).to_markdown(index=False),
        "",
        "## Risk / coverage",
        "",
        risk_coverage_curve(predictions).round(4).to_markdown(index=False),
    ]

    for column, heading in (("difficulty", "difficulty"), ("subset", "source benchmark")):
        slice_frame = breakdown(predictions, column)
        if slice_frame.empty:
            continue
        lines += ["", f"## Accuracy by {heading}", "", slice_frame.head(20).round(4).to_markdown(index=False)]

    lines += ["", "## Confidently wrong", "", hardest_examples(predictions).round(3).to_markdown(index=False), ""]
    return "\n".join(lines)
