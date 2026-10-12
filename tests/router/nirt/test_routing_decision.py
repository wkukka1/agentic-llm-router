"""Pure routing-policy helpers (router.nirt.routing_decision)."""

from __future__ import annotations

import pytest

from router.nirt.routing_decision import alpha_to_lam, linear_cost


def test_linear_cost_clips_to_pool_range():
    pool = [0.0001, 0.001, 0.004]
    assert linear_cost(0.0001, pool) == pytest.approx(0.0)
    assert linear_cost(0.004, pool) == pytest.approx(1.0)
    assert linear_cost(10.0, pool) == pytest.approx(1.0)  # above range clips, doesn't extrapolate


def test_linear_cost_degenerate_pool_is_zero():
    """Every model costs the same -> no cost signal to normalize against."""
    assert linear_cost(0.5, [0.002, 0.002, 0.002]) == pytest.approx(0.0)


def test_alpha_to_lam_endpoints():
    assert alpha_to_lam(1.0) == pytest.approx(0.0)   # quality-only
    assert alpha_to_lam(0.5) == pytest.approx(1.0)   # equal weight
    assert alpha_to_lam(0.1) == pytest.approx(9.0)   # cost-averse -> large lam


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.1])
def test_alpha_to_lam_rejects_out_of_range(bad):
    with pytest.raises(ValueError, match="alpha"):
        alpha_to_lam(bad)
