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
