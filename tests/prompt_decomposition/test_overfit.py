"""The overfitting audit has to be trustworthy before its verdicts are.

Each test builds data with a known answer -- separable, pure noise, or
duplicated -- and checks the audit reaches it.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.prompt_decomposition.overfit import AuditResult, audit

# Every permutation is a full k-fold refit, so the audit's cost is linear in
# this. Ten is enough for a shuffled baseline; only the p-value's *resolution*
# needs the production default of 100, and that is covered by constructing
# AuditResult directly in test_review_fixes.py.
PERMUTATIONS = 10


def _separable(n=300, d=256, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([i % classes for i in range(n)])
    # Separation of 2 keeps within-cluster cosine near 0.87, comfortably under
    # the 0.95 near-duplicate threshold, so only genuine copies are counted.
    centres = rng.normal(size=(classes, d)) * 2
    return centres[y] + rng.normal(size=(n, d)), y.astype(str)


def _noise(n=300, d=256, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.integers(0, classes, n).astype(str)


@pytest.fixture(scope="module")
def separable_audit():
    """One audit shared by every test that reads the same separable result --
    each audit is a multi-second k-fold sweep and the tests only inspect it."""
    X, y = _separable()
    return audit(X, y, "separable", permutations=PERMUTATIONS)


@pytest.fixture(scope="module")
def noise_audit():
    X, y = _noise()
    return audit(X, y, "noise", permutations=PERMUTATIONS)


class TestPermutationCheck:
    def test_separable_data_beats_its_permutation_null(self, separable_audit):
        r = separable_audit
        assert r.test_acc > 0.9
        assert r.passes_permutation
        assert r.permutation_sd > 10

    def test_pure_noise_does_not_pass(self, noise_audit):
        """The point of the check: a model with no signal must be caught."""
        assert not noise_audit.passes_permutation
        assert not noise_audit.clean

    def test_the_yardstick_is_the_shuffled_score_not_the_majority_rate(self):
        """A balanced-weight model scores *below* the majority rate on shuffled
        labels. That is expected, not suspicious, and must not fail the check."""
        rng = np.random.default_rng(0)
        n, d = 400, 256
        y = np.array(["a"] * 320 + ["b"] * 80)
        centres = {"a": rng.normal(size=d) * 2, "b": rng.normal(size=d) * 2}
        X = np.vstack([centres[v] for v in y]) + rng.normal(size=(n, d))
        r = audit(X, y, "imbalanced", balanced=True, permutations=PERMUTATIONS)
        assert r.shuffled_mean < r.majority_rate      # the situation being guarded
        assert r.passes_permutation                   # and it is not flagged


class TestNearDuplicateCheck:
    def test_duplicated_rows_are_counted_and_rescored(self):
        X, y = _separable(n=200)
        X = np.vstack([X, X[:40]])
        y = np.concatenate([y, y[:40]])
        r = audit(X, y, "duplicated", permutations=PERMUTATIONS)
        assert r.near_dupe_pairs >= 40
        assert r.near_dupe_checked
        assert not np.isnan(r.test_without_near_dupes)

    def test_a_rescore_that_cannot_run_is_reported_not_assumed_clean(self):
        """In a tight low-dimensional space nearly every pair clears 0.95 and
        there is nothing left to refit. That must say so rather than pass."""
        X, y = _separable(n=200, d=4)
        r = audit(X, y, "tight", permutations=PERMUTATIONS)
        assert not r.near_dupe_checked
        assert "NOT CHECKED" in r.summary()

    def test_clean_data_reports_no_near_duplicate_effect(self, separable_audit):
        assert separable_audit.passes_near_dupe


class TestReporting:
    def test_learning_curve_and_regularisation_are_populated(self, separable_audit):
        r = separable_audit
        assert len(r.learning_curve) >= 3
        assert [c[0] for c in r.learning_curve] == sorted(c[0] for c in r.learning_curve)
        assert len(r.regularisation) == 4

    def test_summary_names_the_verdict(self, separable_audit, noise_audit):
        assert "CLEAN" in separable_audit.summary()
        assert "NEEDS REVIEW" in noise_audit.summary()

    def test_permutation_sd_is_finite_when_permutations_agree_exactly(self):
        """Zero variance across permutations must not divide by zero."""
        r = AuditResult(name="x", n=10, n_classes=2, majority_rate=0.5,
                        train_acc=1.0, test_acc=0.9, fold_sd=0.0,
                        shuffled_mean=0.5, shuffled_sd=0.0)
        assert np.isfinite(r.permutation_sd)
        assert r.passes_permutation
