"""Head-to-head: does predicting the routing decision beat inferring it?

Two ways to decide "should this prompt go to a strong model":

* **direct**  -- a head trained on preference battles predicts it outright.
* **indirect** -- the domain head predicts a topic, the heuristic difficulty
  estimator scores the prompt, and the policy infers the decision from both.

Both are scored against the same ground truth on the same rows: the preference
test split, where ``routing_label`` records whether the strong model actually
won. That is the only comparison that settles priority 0, because it holds the
objective fixed and varies only how the decision is reached.

A third row is always reported: the majority-class baseline. Any path that fails
to beat it is not routing, it is guessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from router.data.sources.preference import STRONG_NEEDED

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PathResult:
    name: str
    accuracy: float
    macro_f1: float
    roc_auc: float | None
    escalation_rate: float
    notes: str = ""


def _rows(result: PathResult) -> dict:
    return {
        "path": result.name,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "roc_auc": result.roc_auc,
        "escalation_rate": result.escalation_rate,
        "notes": result.notes,
    }


def score_scores(name: str, y_true: np.ndarray, scores: np.ndarray, threshold: float, notes: str = "") -> PathResult:
    """Score a continuous 'needs strong' signal at a given threshold."""
    predicted = (scores >= threshold).astype(int)
    auc = None
    if 0 < y_true.sum() < len(y_true):
        auc = float(roc_auc_score(y_true, scores))
    return PathResult(
        name=name,
        accuracy=float(accuracy_score(y_true, predicted)),
        macro_f1=float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        roc_auc=auc,
        escalation_rate=float(predicted.mean()),
        notes=notes,
    )


def majority_baseline(y_true: np.ndarray) -> PathResult:
    majority = int(round(y_true.mean()))
    predicted = np.full_like(y_true, majority)
    return PathResult(
        name="majority baseline",
        accuracy=float(accuracy_score(y_true, predicted)),
        macro_f1=float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        roc_auc=None,
        escalation_rate=float(predicted.mean()),
        notes="always predicts the more common class",
    )


def compare(
    test: pd.DataFrame,
    *,
    direct_scores: np.ndarray | None,
    indirect_scores: np.ndarray | None,
    direct_threshold: float = 0.5,
    indirect_threshold: float | None = None,
) -> pd.DataFrame:
    """Build the comparison table.

    ``indirect_threshold`` defaults to the value that makes the indirect path
    escalate at the same rate as the ground-truth positive rate, so the two
    paths are compared at a matched operating point rather than at whatever cut
    each happens to prefer.
    """
    y_true = (test["routing_label"] == STRONG_NEEDED).to_numpy().astype(int)
    results: list[PathResult] = [majority_baseline(y_true)]

    if direct_scores is not None:
        results.append(score_scores(
            "direct (preference head)", y_true, direct_scores, direct_threshold,
            "trained on the routing decision itself",
        ))

    if indirect_scores is not None:
        if indirect_threshold is None:
            indirect_threshold = float(np.quantile(indirect_scores, 1 - y_true.mean()))
        results.append(score_scores(
            "indirect (domain + difficulty)", y_true, indirect_scores, indirect_threshold,
            f"policy signal, threshold matched to base rate ({indirect_threshold:.3f})",
        ))

    return pd.DataFrame([_rows(r) for r in results])


def render(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in ("accuracy", "macro_f1", "roc_auc", "escalation_rate"):
        display[col] = display[col].map(lambda v: "—" if pd.isna(v) else f"{v:.4f}")
    return (
        "# Priority 0: direct vs indirect routing\n\n"
        "Both paths scored on the same preference test split against the same\n"
        "ground truth (`routing_label`: did the strong model actually win?).\n\n"
        + display.to_markdown(index=False) + "\n"
    )
