"""Phase 1 diagnostics: theta spectrum / effective rank / ICC sanity."""

from __future__ import annotations

import numpy as np

from router.nirt.baseline.diagnostics import (
    effective_rank,
    icc_curve,
    parameter_summary,
    theta_spectrum,
)


def test_effective_rank_bounds():
    assert effective_rank([1, 0, 0, 0]) == 1.0
    assert effective_rank([1, 1, 1, 1]) == 4.0
    assert 1.0 < effective_rank([3, 1, 0.5, 0.1]) < 4.0


def test_theta_spectrum_detects_collapse():
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(5)
    theta = np.outer(rng.standard_normal(40), direction)   # rank-1
    spec = theta_spectrum(theta)
    assert spec["collapsed"] is True
    assert spec["effective_rank"] < 1.2


def test_theta_spectrum_full_rank():
    rng = np.random.default_rng(1)
    theta = rng.standard_normal((40, 5))
    spec = theta_spectrum(theta)
    assert spec["collapsed"] is False
    assert spec["effective_rank"] > 3.0
    assert len(spec["singular_values"]) == 5


def test_icc_monotone_increasing_for_positive_discrimination():
    a = np.array([1.5, 0.2, 0.0])
    c = icc_curve(a, b_q=0.3, dim_index=0)
    p = np.array(c["p_correct"])
    assert np.all(np.diff(p) >= -1e-9)
    assert c["monotone_increasing"] is True
    assert c["p_range"][0] < 0.2 and c["p_range"][1] > 0.8


def test_icc_flat_when_discrimination_zero_on_swept_dim():
    a = np.array([0.0, 1.0])
    c = icc_curve(a, b_q=0.0, dim_index=0)
    p = np.array(c["p_correct"])
    assert p.max() - p.min() < 1e-6


def test_icc_difficulty_shifts_curve():
    a = np.array([1.0])
    easy = icc_curve(a, b_q=-2.0, dim_index=0)
    hard = icc_curve(a, b_q=2.0, dim_index=0)
    # at theta = 0, easier item has higher P(correct)
    mid = len(easy["p_correct"]) // 2
    assert easy["p_correct"][mid] > hard["p_correct"][mid]


def test_parameter_summary_flags_nonfinite():
    s = parameter_summary(theta=np.array([[np.nan, 1.0]]), a_q=np.array([[1.0, 2.0]]),
                          b_q=np.array([0.0]))
    assert any("non-finite" in f for f in s["pathology_flags"])
