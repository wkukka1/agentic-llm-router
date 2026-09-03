from __future__ import annotations

import warnings

import pandas as pd
import pytest

from router.data import schemas
from router.data.chance_correction import (
    apply_chance_correction,
    estimate_model_bias,
    normalized_correction,
)


class _CC:
    def __init__(self, method="normalized", clip=True, warn=True):
        self._d = {"chance_correction": {"method": method, "clip": clip,
                                         "warn_on_missing_choices": warn}}

    def get(self, key, default=None):
        return self._d.get(key, default)


def test_normalized_formula_known_values():
    # 4 choices, chance = 0.25
    assert normalized_correction(0.25, 4) == pytest.approx(0.0)
    assert normalized_correction(1.0, 4) == pytest.approx(1.0)
    assert normalized_correction(0.625, 4) == pytest.approx(0.5)
    # 2 choices, chance = 0.5
    assert normalized_correction(0.5, 2) == pytest.approx(0.0)
    assert normalized_correction(0.75, 2) == pytest.approx(0.5)


def test_below_chance_clips_to_zero():
    assert normalized_correction(0.0, 4, clip=True) == 0.0
    assert normalized_correction(0.0, 4, clip=False) < 0


def test_original_score_is_preserved(toy_responses):
    before = toy_responses["score"].copy()
    out = apply_chance_correction(toy_responses, _CC(warn=False))
    pd.testing.assert_series_equal(out["score"], before)
    assert "corrected_score" in out.columns


def test_no_correction_for_non_mc(toy_responses):
    out = apply_chance_correction(toy_responses, _CC(warn=False))
    non_mc = out[~out["is_multiple_choice"].astype(bool)]
    assert non_mc["chance_corrected"].eq(False).all()
    assert non_mc["corrected_score"].isna().all()


def test_correction_applied_for_mc_with_n(toy_responses):
    out = apply_chance_correction(toy_responses, _CC(warn=False))
    mc = out[(out["metric_type"] == schemas.MetricType.MC_ACCURACY)
             & out["n_choices"].notna()]
    assert mc["chance_corrected"].all()
    assert mc["corrected_score"].between(0, 1).all()
    row = mc[mc["score"] == 0.25]
    assert row["corrected_score"].iloc[0] == pytest.approx(0.0)


def test_missing_choice_count_warns_and_skips(toy_responses):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = apply_chance_correction(toy_responses, _CC(warn=True))
    assert any("no n_choices" in str(x.message) for x in w)
    mystery = out[out["dataset"] == "mystery"]
    assert mystery["chance_corrected"].eq(False).all()
    assert mystery["corrected_score"].isna().all()


def test_method_none_is_noop(toy_responses):
    out = apply_chance_correction(toy_responses, _CC(method="none"))
    assert out["chance_corrected"].eq(False).all()


def test_estimate_model_bias_columns(toy_responses):
    bias = estimate_model_bias(toy_responses)
    assert set(bias.columns) == {
        "model_id", "mc_n", "mc_mean_score", "mc_mean_chance", "mc_excess"
    }
