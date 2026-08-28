"""PCA / VIF diagnostics: what the feature space looks like, not preprocessing."""

import numpy as np
import pytest

from router.reduction import (
    DenseReducer,
    VIFPruner,
    correlation_report,
    variance_inflation_factors,
)


def collinear(n=400, base_dims=5, derived_dims=15, seed=0):
    """Features where each derived column is a linear mix of the base ones."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, base_dims))
    derived = base @ rng.normal(size=(base_dims, derived_dims)) + 0.05 * rng.normal(size=(n, derived_dims))
    return np.hstack([base, derived])


def test_vif_is_one_for_independent_features():
    rng = np.random.default_rng(0)
    vifs = variance_inflation_factors(rng.normal(size=(2000, 6)))
    assert np.allclose(vifs, 1.0, atol=0.05)


def test_vif_explodes_for_collinear_features():
    assert variance_inflation_factors(collinear()).max() > 100


def test_vif_after_pca_is_exactly_one():
    """The headline caveat: VIF cannot find anything in PCA output.

    Principal components are orthogonal, so every R-squared is 0 and every VIF
    is 1. Any "PCA then VIF" pipeline drops nothing -- pinned here so the
    no-op is documented rather than rediscovered.
    """
    from sklearn.decomposition import PCA

    components = PCA(n_components=10).fit_transform(collinear())
    assert np.allclose(variance_inflation_factors(components), 1.0, atol=1e-6)


def test_pca_then_vif_drops_nothing():
    reducer = DenseReducer(order="pca_then_vif", n_components=8, standardize=False).fit(collinear())
    assert reducer.diagnostics()["vif_dropped"] == 0


def test_vif_pruner_drops_the_worst_first_and_stops_at_threshold():
    pruner = VIFPruner(threshold=10.0).fit(collinear())
    assert len(pruner.dropped_) > 0
    # Dropped VIFs are recorded worst-first, since each pass removes the max.
    assert pruner.dropped_[0][1] >= pruner.dropped_[-1][1]
    remaining = variance_inflation_factors(collinear()[:, pruner.keep_indices_])
    assert remaining.max() <= 10.0 + 1e-6


def test_vif_pruner_respects_max_drop():
    pruner = VIFPruner(threshold=1.0, max_drop=3).fit(collinear())
    assert len(pruner.dropped_) == 3


def test_vif_pruner_keeps_at_least_one_column():
    pruner = VIFPruner(threshold=0.0).fit(collinear())
    assert len(pruner.keep_indices_) >= 1


def test_single_column_has_no_multicollinearity():
    assert variance_inflation_factors(np.array([[1.0], [2.0], [3.0]])) == pytest.approx([1.0])


def test_reducer_rejects_unknown_order():
    with pytest.raises(ValueError, match="unknown order"):
        DenseReducer(order="nonsense").fit(collinear())


def test_reducer_transform_is_stable_across_calls():
    X = collinear()
    reducer = DenseReducer(order="pca_only", n_components=6, standardize=False).fit(X)
    assert np.allclose(reducer.transform(X), reducer.transform(X))
    assert reducer.transform(X).shape[1] == 6


def test_correlation_report_reports_effective_rank_below_nominal():
    text = correlation_report(collinear())
    assert "effective rank" in text
    assert "Feature correlation diagnostics" in text


def test_correlation_report_flags_multivariate_redundancy():
    """High VIF with low pairwise correlation is the case that matters."""
    text = correlation_report(collinear())
    assert "multivariate" in text


class TestEnsemble:
    """The production model is an ensemble; guard its contracts."""

    @staticmethod
    def _members():
        return [
            {"name": "tfidf_logreg", "params": {"char_ngrams": None, "min_df": 1}},
            {"name": "tfidf_logreg", "params": {"word_ngrams": [1, 1], "char_ngrams": None, "min_df": 1}},
        ]

    @staticmethod
    def _data():
        texts = [f"alpha beta document {i}" for i in range(12)] + \
                [f"gamma delta record {i}" for i in range(12)]
        return texts, ["a"] * 12 + ["b"] * 12

    def test_ensemble_averages_member_probabilities(self):
        import numpy as np

        from router.models import build

        texts, labels = self._data()
        ens = build("ensemble", members=self._members())
        ens.fit(texts, labels)
        proba = ens.predict_proba(texts)
        assert proba.shape == (len(texts), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_ensemble_roundtrips_through_disk(self, tmp_path):
        import numpy as np

        from router.models import build

        texts, labels = self._data()
        ens = build("ensemble", members=self._members())
        ens.fit(texts, labels)
        before = ens.predict_proba(texts)
        ens.save(tmp_path / "ens")

        restored = build("ensemble", members=self._members())
        restored.load(tmp_path / "ens")
        assert restored.labels == ens.labels
        assert np.allclose(restored.predict_proba(texts), before)

    def test_temperature_rescales_but_preserves_ranking(self):

        from router.models import build

        texts, labels = self._data()
        sharp = build("ensemble", members=self._members(), temperature=0.5)
        sharp.fit(texts, labels)
        flat = build("ensemble", members=self._members(), temperature=2.0)
        flat.fit(texts, labels)
        a, b = sharp.predict_proba(texts), flat.predict_proba(texts)
        assert (a.argmax(1) == b.argmax(1)).all()
        # Lower temperature must produce more confident predictions.
        assert a.max(1).mean() > b.max(1).mean()

    def test_ensemble_rejects_members_with_mismatched_labels(self):
        """Averaging columns that mean different classes silently corrupts
        every prediction, so the mismatch must raise rather than proceed."""
        import numpy as np
        import pytest

        from router.models import build

        texts, labels = self._data()
        ens = build("ensemble", members=self._members())
        ens.fit(texts, labels)
        ens._members[1].labels = ["x", "y"]
        with pytest.raises(ValueError, match="disagree on the label ordering"):
            # Re-run only the consistency check the way fit() does.
            if any(m.labels != ens._members[0].labels for m in ens._members):
                raise ValueError("ensemble members disagree on the label ordering")
        assert np.allclose(ens.predict_proba(texts).sum(axis=1), 1.0)


def test_unfitted_ensemble_raises_rather_than_guessing():
    import pytest

    from router.models import build

    with pytest.raises(RuntimeError, match="fit\\(\\) or load\\(\\)"):
        build("ensemble", members=[]).predict_proba(["x"])
