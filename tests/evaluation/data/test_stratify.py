from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.data.stratify import sample_stratified_queries


def _matrix(n_queries=20, n_models=4, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"q{i}" for i in range(n_queries)]
    data = rng.integers(0, 2, size=(n_queries, n_models)).astype(float)
    return pd.DataFrame(data, index=ids, columns=[f"m{i}" for i in range(n_models)])


def test_rejects_nan():
    R = _matrix()
    R.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sample_stratified_queries(R, n_discriminative=5, n_natural=5, seed=0)


def test_strata_are_disjoint_and_labelled():
    R = _matrix(n_queries=30, n_models=5, seed=1)
    out = sample_stratified_queries(R, n_discriminative=10, n_natural=10, seed=0)
    assert set(out["stratum"]) <= {"discriminative", "natural"}
    assert out.index.is_unique
    assert (out["stratum"] == "natural").sum() == 10


def test_discriminative_excludes_unanimous_rows():
    ids = [f"q{i}" for i in range(6)]
    R = pd.DataFrame(
        [[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]],
        index=ids, columns=["a", "b", "c"], dtype=float,
    )
    out = sample_stratified_queries(R, n_discriminative=10, n_natural=0, seed=0)
    assert set(out.index) == {"q2", "q3"}
    assert (out["row_mean_accuracy"] > 0).all() and (out["row_mean_accuracy"] < 1).all()


def test_seeded_reproducibility():
    R = _matrix(n_queries=40, n_models=4, seed=2)
    a = sample_stratified_queries(R, n_discriminative=8, n_natural=8, seed=7)
    b = sample_stratified_queries(R, n_discriminative=8, n_natural=8, seed=7)
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_natural_draw_excluded_from_discriminative_pool():
    # every row is discriminative -- natural draws should still be excluded
    # from the discriminative sample so the two strata never overlap.
    ids = [f"q{i}" for i in range(10)]
    R = pd.DataFrame(
        [[1, 0]] * 5 + [[0, 1]] * 5, index=ids, columns=["a", "b"], dtype=float,
    )
    out = sample_stratified_queries(R, n_discriminative=10, n_natural=4, seed=3)
    disc = set(out.index[out["stratum"] == "discriminative"])
    nat = set(out.index[out["stratum"] == "natural"])
    assert disc.isdisjoint(nat)
    assert len(disc) == 6  # 10 rows - 4 taken by the natural draw
