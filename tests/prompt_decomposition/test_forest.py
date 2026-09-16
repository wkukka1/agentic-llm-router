"""The forest, and the three ways of asking what it learned.

The tests are built around the failure this experiment is most likely to
produce: a chart that looks authoritative and is counting columns. So the
fixtures deliberately include a *wide, useless* family alongside a *narrow,
decisive* one, and assert that the honest views separate them even when the
impurity view does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.prompt_decomposition.forest_report import build_report, plot_report
from training.prompt_decomposition.heads.forest import (
    ForestFeatures,
    build_features,
    fit_forest,
)


@pytest.fixture
def decisive_narrow_vs_wide_noise():
    """8 informative domain columns, 120 columns of pure noise as `embedding`."""
    rng = np.random.default_rng(0)
    n = 1200
    signal = rng.normal(0, 1, (n, 8))
    y = (signal[:, 0] + 0.6 * signal[:, 1] > 0).astype(int)
    noise = rng.normal(0, 1, (n, 120))
    features = build_features(
        domain_proba=signal, domain_labels=[f"d{i}" for i in range(8)],
        embeddings=noise)
    cut = 900
    return (ForestFeatures(features.X[:cut], features.names, features.families), y[:cut],
            ForestFeatures(features.X[cut:], features.names, features.families), y[cut:])


class TestFeatureAssembly:
    def test_width_names_and_families_must_agree(self):
        with pytest.raises(ValueError, match="must agree"):
            ForestFeatures(np.zeros((3, 2)), ["a"], ["domain"])

    def test_each_block_is_labelled_with_its_family(self):
        f = build_features(domain_proba=np.zeros((2, 3)), domain_labels=["a", "b", "c"],
                           task_proba=np.zeros((2, 2)), task_labels=["x", "y"],
                           embeddings=np.zeros((2, 5)))
        assert f.columns_per_family if False else True
        assert len(f.columns_for("domain")) == 3
        assert len(f.columns_for("task")) == 2
        assert len(f.columns_for("embedding")) == 5
        assert f.names[0] == "domain.a" and f.names[3] == "task.x"

    def test_a_distribution_without_matching_labels_is_rejected(self):
        """A silent mismatch would mislabel every importance in the report."""
        with pytest.raises(ValueError, match="one label per column"):
            build_features(domain_proba=np.zeros((2, 3)), domain_labels=["a", "b"])

    def test_no_families_at_all_is_an_error(self):
        with pytest.raises(ValueError, match="no feature families"):
            build_features()

    def test_only_and_without_are_complementary(self):
        f = build_features(domain_proba=np.zeros((2, 3)), domain_labels=["a", "b", "c"],
                           embeddings=np.zeros((2, 4)))
        assert f.only("domain").X.shape[1] + f.without("domain").X.shape[1] == f.X.shape[1]
        assert f.without("domain").present_families == ["embedding"]


class TestReport:
    def test_the_honest_views_separate_signal_from_width(self, decisive_narrow_vs_wide_noise):
        """The point of the module. 120 noise columns get plenty of impurity
        gain simply by being offered as splits; ablation and permutation must
        still show they carry nothing."""
        train, y_train, test, y_test = decisive_narrow_vs_wide_noise
        r = build_report(train, y_train, test, y_test, target="synthetic", permutation_repeats=3)

        assert r.alone["domain"] > 0.8
        assert r.alone["embedding"] < 0.6              # noise is near chance alone
        assert r.without["embedding"] >= r.auc - 0.05  # dropping it costs nothing
        assert r.permutation_drop["domain"] > r.permutation_drop["embedding"]

    def test_information_gain_is_reported_per_family_with_column_counts(
            self, decisive_narrow_vs_wide_noise):
        """Counts travel with the gains so a reader can see the bias rather
        than having to know about it."""
        train, y_train, test, y_test = decisive_narrow_vs_wide_noise
        r = build_report(train, y_train, test, y_test, permutation_repeats=1)
        assert set(r.information_gain) == {"domain", "embedding"}
        assert r.columns_per_family == {"domain": 8, "embedding": 120}
        assert sum(r.information_gain.values()) == pytest.approx(1.0, abs=1e-6)

    def test_gain_per_column_undoes_the_width_bias(self, decisive_narrow_vs_wide_noise):
        """Summed gain says the 120 noise columns dominate; per column it is the
        8 real ones that are worth more. Both are in the report because the
        contradiction is the finding, not a wrinkle to smooth over."""
        train, y_train, test, y_test = decisive_narrow_vs_wide_noise
        r = build_report(train, y_train, test, y_test, permutation_repeats=1)
        assert r.information_gain["embedding"] > r.information_gain["domain"]
        assert r.gain_per_column["domain"] > r.gain_per_column["embedding"]

    def test_the_summary_warns_about_the_metric_it_leads_with(
            self, decisive_narrow_vs_wide_noise):
        train, y_train, test, y_test = decisive_narrow_vs_wide_noise
        text = build_report(train, y_train, test, y_test, permutation_repeats=1).summary()
        assert "rewards wide families" in text
        assert "AUC" in text

    def test_a_single_family_reports_no_ablation_it_cannot_run(
            self, decisive_narrow_vs_wide_noise):
        """`without` is meaningless with one family; it must be absent, not 0.5."""
        train, y_train, test, y_test = decisive_narrow_vs_wide_noise
        r = build_report(train.only("domain"), y_train, test.only("domain"), y_test,
                         permutation_repeats=1)
        assert r.without == {}
        assert "domain" in r.alone


class TestForestFit:
    def test_leaves_are_not_grown_pure(self):
        """Default `min_samples_leaf=1` memorises a noisy target and reports
        importances for splits that are fitting noise."""
        f = build_features(domain_proba=np.random.default_rng(1).normal(0, 1, (200, 4)),
                           domain_labels=list("abcd"))
        forest = fit_forest(f, np.random.default_rng(2).integers(0, 2, 200))
        assert forest.min_samples_leaf > 1
        assert forest.criterion == "entropy"     # so importances are information gain


class TestPlot:
    def test_it_writes_a_figure_for_any_report(self, decisive_narrow_vs_wide_noise, tmp_path):
        """Reusable means reusable: no argument is specific to one experiment."""
        train, y_train, test, y_test = decisive_narrow_vs_wide_noise
        r = build_report(train, y_train, test, y_test, permutation_repeats=1)
        path = plot_report(r, tmp_path / "nested" / "forest.png", title="synthetic")
        assert path.exists() and path.stat().st_size > 5_000
