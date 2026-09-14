"""configs/model_prices.yaml loading + cost-fill from token counts (pool-expansion E3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from router.config import load_config
from training.data.pricing import cost_for, fill_costs, load_prices, price_snapshot


def test_prices_load_and_canonicalize():
    prices = load_prices(load_config())
    assert prices, "configs/model_prices.yaml produced no prices"
    # keyed by canonical id
    p = prices.get("llama-3.1-8b-instruct")
    assert p and p["input_per_1m"] > 0 and p["output_per_1m"] > 0


def test_price_snapshot_is_dated():
    snap = price_snapshot(load_config())
    assert snap["snapshot_date"]
    assert snap["unit"] == "per_1m_tokens"


def test_cost_for_known_tokens():
    prices = {"m": {"input_per_1m": 1.0, "output_per_1m": 2.0}}
    # 1000 in, 500 out -> 1000*1/1e6 + 500*2/1e6 = 0.001 + 0.001 = 0.002
    assert cost_for("m", 1000, 500, prices) == pytest.approx(0.002)
    # unknown model -> NaN
    assert np.isnan(cost_for("nope", 1000, 500, prices))
    # missing token counts treated as 0
    assert cost_for("m", None, None, prices) == 0.0


def test_fill_costs_only_touches_missing_non_routerbench():
    df = pd.DataFrame({
        "model_id": ["llama-3.1-8b-instruct", "llama-3.1-8b-instruct", "gpt-4-1106-preview"],
        "source": ["lm_eval_harness", "lm_eval_harness", "routerbench"],
        "input_tokens": [1_000_000, 500_000, 1000],
        "output_tokens": [0, 1_000_000, 1000],
        "cost": [np.nan, np.nan, 0.123],
    })
    out = fill_costs(df, load_config())
    p = load_prices(load_config())["llama-3.1-8b-instruct"]
    assert out.loc[0, "cost"] == pytest.approx(p["input_per_1m"])          # 1M input tokens
    assert out.loc[1, "cost"] == pytest.approx(0.5 * p["input_per_1m"] + p["output_per_1m"])
    assert out.loc[2, "cost"] == 0.123                                     # routerbench untouched


def test_fill_costs_leaves_unpriced_model_nan():
    df = pd.DataFrame({
        "model_id": ["some-unpriced-model"],
        "source": ["lm_eval_harness"],
        "input_tokens": [1000], "output_tokens": [1000],
        "cost": [np.nan],
    })
    out = fill_costs(df, load_config())
    assert np.isnan(out.loc[0, "cost"])
