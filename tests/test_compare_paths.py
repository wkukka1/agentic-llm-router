"""The priority-0 comparison: direct vs indirect routing, same ground truth."""

import numpy as np
import pandas as pd
import pytest

from router.data.sources.preference import STRONG_NEEDED, WEAK_SUFFICIENT
from router.training.compare_paths import compare, majority_baseline, render, score_scores


def frame(labels):
    return pd.DataFrame({"routing_label": labels, "prompt": ["p"] * len(labels)})


def test_majority_baseline_predicts_one_class():
    y = np.array([1, 1, 1, 0])
    result = majority_baseline(y)
    assert result.accuracy == pytest.approx(0.75)
    assert result.escalation_rate == 1.0
    assert result.roc_auc is None


def test_perfect_scores_give_auc_one():
    y = np.array([0, 0, 1, 1])
    result = score_scores("perfect", y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert result.roc_auc == pytest.approx(1.0)
    assert result.accuracy == pytest.approx(1.0)


def test_inverted_scores_give_auc_zero():
    """A path that is anti-correlated must show as below 0.5, not be hidden."""
    y = np.array([0, 0, 1, 1])
    result = score_scores("inverted", y, np.array([0.9, 0.8, 0.2, 0.1]), 0.5)
    assert result.roc_auc == pytest.approx(0.0)


def test_auc_is_none_when_one_class_is_absent():
    y = np.array([1, 1, 1])
    assert score_scores("x", y, np.array([0.1, 0.5, 0.9]), 0.5).roc_auc is None


def test_escalation_rate_tracks_the_threshold():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    assert score_scores("lo", y, scores, 0.0).escalation_rate == 1.0
    assert score_scores("hi", y, scores, 0.99).escalation_rate == 0.0


def test_compare_always_includes_the_baseline():
    test = frame([STRONG_NEEDED, WEAK_SUFFICIENT] * 10)
    result = compare(test, direct_scores=np.linspace(0, 1, 20), indirect_scores=None)
    assert "majority baseline" in set(result["path"])


def test_compare_matches_the_indirect_threshold_to_the_base_rate():
    """Both paths must be compared at a matched operating point."""
    labels = [STRONG_NEEDED] * 6 + [WEAK_SUFFICIENT] * 14  # 30% positive
    test = frame(labels)
    indirect = np.linspace(0, 1, 20)
    result = compare(test, direct_scores=None, indirect_scores=indirect)
    row = result[result["path"].str.startswith("indirect")].iloc[0]
    assert row["escalation_rate"] == pytest.approx(0.30, abs=0.06)


def test_compare_requires_at_least_one_path():
    test = frame([STRONG_NEEDED, WEAK_SUFFICIENT])
    result = compare(test, direct_scores=None, indirect_scores=None)
    assert len(result) == 1, "only the baseline when no path is supplied"


def test_render_shows_missing_auc_as_a_dash_not_nan():
    test = frame([STRONG_NEEDED, WEAK_SUFFICIENT] * 5)
    text = render(compare(test, direct_scores=np.linspace(0, 1, 10), indirect_scores=None))
    assert "nan" not in text.lower()
    assert "—" in text
