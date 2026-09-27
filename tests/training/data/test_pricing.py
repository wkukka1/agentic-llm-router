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


# --------------------------------------------------------------------------- #
# impute_missing_usage: rows whose usage accounting is missing (0 tokens, $0)  #
# --------------------------------------------------------------------------- #
from training.data.pricing import impute_missing_usage  # noqa: E402

_IN_RATE, _OUT_RATE = 1.0, 3.0  # USD per 1M tokens, for the synthetic model "m"


def _usage_frame(n_priced: int = 60) -> pd.DataFrame:
    """Model ``m`` priced at 1/3 per 1M on ``n_priced`` rows (varied token counts,
    so both rates are identifiable), one 0/0/0 row, and model ``n`` answering the
    same 0/0/0 query with 200 input tokens."""
    rng = np.random.default_rng(0)
    it = rng.integers(100, 500, n_priced).astype(float)
    ot = rng.integers(10, 200, n_priced).astype(float)
    rows = [
        {"query_id": f"q{i}", "model_id": "m", "dataset": "d", "score": 1.0,
         "input_tokens": it[i], "output_tokens": ot[i],
         "cost": (it[i] * _IN_RATE + ot[i] * _OUT_RATE) / 1e6, "metadata": {}}
        for i in range(n_priced)
    ]
    rows.append({"query_id": "qz", "model_id": "m", "dataset": "d", "score": 1.0,
                 "input_tokens": 0.0, "output_tokens": 0.0, "cost": 0.0, "metadata": {}})
    rows.append({"query_id": "qz", "model_id": "n", "dataset": "d", "score": 1.0,
                 "input_tokens": 200.0, "output_tokens": 50.0, "cost": 0.5, "metadata": {}})
    return pd.DataFrame(rows)


def _zero_row(df: pd.DataFrame, model: str = "m") -> pd.Series:
    return df[(df.query_id == "qz") & (df.model_id == model)].iloc[0]


def test_zero_usage_row_is_priced_at_the_rate_recovered_from_its_own_model():
    out = impute_missing_usage(_usage_frame())
    row = _zero_row(out)
    assert row["input_tokens"] > 0 and row["output_tokens"] > 0
    expected = (row["input_tokens"] * _IN_RATE + row["output_tokens"] * _OUT_RATE) / 1e6
    assert row["cost"] == pytest.approx(expected, rel=1e-6)


def test_imputed_input_tokens_come_from_the_same_query_on_other_models():
    out = impute_missing_usage(_usage_frame())
    assert _zero_row(out)["input_tokens"] == 200.0


def test_rows_with_real_usage_are_left_untouched():
    df = _usage_frame()
    out = impute_missing_usage(df)
    keep = df.query_id != "qz"
    pd.testing.assert_frame_equal(out[keep.to_numpy()].drop(columns="metadata"),
                                  df[keep].drop(columns="metadata"))
    assert out[keep.to_numpy()]["metadata"].map(lambda m: "usage_imputed" not in m).all()


def test_imputed_rows_are_flagged_in_metadata():
    out = impute_missing_usage(_usage_frame())
    assert _zero_row(out)["metadata"]["usage_imputed"] is True


def test_model_with_no_priced_rows_gets_nan_never_a_free_price():
    # only the two ``qz`` rows: model ``m`` has no priced row to recover a rate from
    df = _usage_frame()
    out = impute_missing_usage(df[df.query_id == "qz"])
    row = _zero_row(out)
    assert pd.isna(row["cost"]) and pd.isna(row["input_tokens"]) and pd.isna(row["output_tokens"])


def test_free_model_with_real_token_counts_stays_free():
    df = _usage_frame()
    free = df[df.query_id != "qz"].copy().assign(model_id="free", cost=0.0)
    out = impute_missing_usage(pd.concat([df, free], ignore_index=True))
    assert (out[out.model_id == "free"]["cost"] == 0.0).all()
