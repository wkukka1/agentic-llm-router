from __future__ import annotations

import numpy as np
import pytest

from router.nirt.shrinkage import novelty_weight, shrink_predictions


def test_novelty_weight_range_and_monotonicity():
    rng = np.random.default_rng(0)
    # high dimensionality -> random training points are ~orthogonal, so a
    # deliberately opposed query has low similarity to the *whole* pool, not
    # just to the one point it was negated from.
    train = rng.standard_normal((200, 768))
    near = train[0:1]
    far = -train[0:1]
    w = novelty_weight(np.vstack([near, far]), train)   # default k=1
    assert w.shape == (2,)
    assert (w >= 0).all() and (w <= 1).all()
    assert w[0] > w[1]          # near-duplicate trusted more than its negation
    assert w[0] > 0.9           # near a training point -> high trust
    assert w[1] < 0.1           # far from every training point -> low trust


def test_novelty_weight_k_larger_than_pool():
    train = np.random.default_rng(1).standard_normal((3, 8))
    w = novelty_weight(train[:1], train, k=100)   # k clipped to pool size
    assert w.shape == (1,)
    assert np.isfinite(w).all()


def test_shrink_predictions_endpoints():
    pred = np.array([[0.9, 0.1], [0.8, 0.2]])
    mean_rate = np.array([0.5, 0.5])
    full_trust = shrink_predictions(pred, np.array([1.0, 1.0]), mean_rate)
    no_trust = shrink_predictions(pred, np.array([0.0, 0.0]), mean_rate)
    np.testing.assert_array_almost_equal(full_trust, pred)
    np.testing.assert_array_almost_equal(no_trust, np.tile(mean_rate, (2, 1)))


def test_shrink_predictions_interpolates():
    pred = np.array([[1.0, 0.0]])
    mean_rate = np.array([0.4, 0.6])
    half = shrink_predictions(pred, np.array([0.5]), mean_rate)
    np.testing.assert_array_almost_equal(half, [[0.7, 0.3]])


def test_shrink_predictions_clips_weight():
    pred = np.array([[1.0, 0.0]])
    mean_rate = np.array([0.5, 0.5])
    over = shrink_predictions(pred, np.array([5.0]), mean_rate)     # clipped to 1
    under = shrink_predictions(pred, np.array([-5.0]), mean_rate)   # clipped to 0
    np.testing.assert_array_almost_equal(over, pred)
    np.testing.assert_array_almost_equal(under, [mean_rate])


def test_novelty_weight_rejects_bad_k():
    with pytest.raises(ValueError):
        novelty_weight(np.zeros((1, 4)), np.zeros((1, 4)), k=0)
