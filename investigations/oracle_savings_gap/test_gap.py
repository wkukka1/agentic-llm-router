"""Tests for the pure router-vs-oracle comparison in ``gap.py``.

Run from the repo root:  pytest investigations/oracle_savings_gap
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gap


# --------------------------------------------------------------------------- #
# oracle_pick                                                                 #
# --------------------------------------------------------------------------- #
def test_oracle_pick_is_cheapest_model_within_tolerance_of_best():
    # best score is model 0 (0.90); model 1 is within tol and 10x cheaper;
    # model 2 is cheapest overall but far below the best.
    true = np.array([[0.90, 0.89, 0.50]])
    cost = np.array([[0.010, 0.001, 0.0001]])

    idx = gap.oracle_pick(true, cost, ["a", "b", "c"], tol=0.02)

    assert idx.tolist() == [1]


def test_oracle_pick_breaks_full_ties_by_smallest_model_id_not_column_order():
    true = np.array([[1.0, 1.0]])
    cost = np.array([[0.001, 0.001]])

    idx = gap.oracle_pick(true, cost, ["b", "a"], tol=0.0)

    assert idx.tolist() == [1]  # "a" sits in column 1


def test_oracle_pick_rejects_non_finite_cells():
    true = np.array([[1.0, np.nan]])
    cost = np.array([[0.001, 0.001]])

    with pytest.raises(ValueError, match="non-finite"):
        gap.oracle_pick(true, cost, ["a", "b"], tol=0.02)


# --------------------------------------------------------------------------- #
# compare_to_oracle                                                           #
# --------------------------------------------------------------------------- #
MODELS = ["cheap", "mid", "big"]
COST = np.array([[0.001, 0.010, 0.100]])


def _one(true_row, selected_col, **kw):
    """Compare a single query whose router picked ``MODELS[selected_col]``."""
    out = gap.compare_to_oracle(
        np.array([true_row]), COST, np.array([selected_col]), MODELS, **kw
    )
    return out.iloc[0]


def test_router_paying_for_a_pricier_model_at_equal_quality_is_overspend():
    # every model is right; the oracle takes "cheap", the router took "big"
    row = _one([1.0, 1.0, 1.0], selected_col=2)

    assert row["failure"] == "overspend"
    assert row["oracle_model"] == "cheap"
    assert row["router_model"] == "big"
    assert row["savings_gap"] == pytest.approx(0.100 - 0.001)
    assert row["savings_gap_frac"] == pytest.approx((0.100 - 0.001) / 0.100)


def test_router_scoring_below_tolerance_of_best_is_quality_loss_even_if_cheaper():
    # only "big" is right; the router saved money by picking "cheap" and lost
    row = _one([0.0, 0.0, 1.0], selected_col=0)

    assert row["failure"] == "quality_loss"
    assert row["oracle_model"] == "big"
    assert row["savings_gap"] < 0  # the oracle spent more than the router


def test_router_picking_the_oracle_model_is_ok():
    row = _one([1.0, 1.0, 1.0], selected_col=0)

    assert row["failure"] == "ok"
    assert row["savings_gap"] == 0.0


def test_pricier_pick_with_gap_below_min_savings_is_ok():
    # "mid" costs 10x "cheap" but the oracle only needs to beat the router's
    # cost by 25% -- raise the bar past the actual 90% gap and it is not flagged
    row = _one([1.0, 1.0, 1.0], selected_col=1, min_savings=0.95)

    assert row["failure"] == "ok"
    assert row["savings_gap_frac"] == pytest.approx(0.9)  # still reported


def test_pick_within_tolerance_of_best_is_not_quality_loss():
    # router's pick is 0.015 below best: inside tol=0.02, and it is the oracle
    row = _one([0.985, 1.0, 1.0], selected_col=0, tol=0.02)

    assert row["failure"] == "ok"


# --------------------------------------------------------------------------- #
# build_gap_table                                                             #
# --------------------------------------------------------------------------- #

def _frames():
    idx = ["q_overspend", "q_lost", "q_ok"]
    true = pd.DataFrame(
        [[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], index=idx, columns=MODELS
    )
    cost = pd.DataFrame(np.tile(COST, (3, 1)), index=idx, columns=MODELS)
    return true, cost


def test_gap_table_has_one_row_per_query_with_router_prefixed_columns():
    true, cost = _frames()
    sel = {"nirt": np.array([2, 0, 0]), "zoib": np.array([2, 2, 0])}

    table = gap.build_gap_table(true, cost, sel)

    assert list(table.index) == list(true.index)
    assert table.loc["q_overspend", "oracle_model"] == "cheap"
    assert table.loc["q_overspend", "nirt_model"] == "big"
    assert table.loc["q_overspend", "nirt_failure"] == "overspend"
    assert table.loc["q_lost", "nirt_failure"] == "quality_loss"
    assert table.loc["q_lost", "zoib_failure"] == "ok"   # zoib picked "big", the oracle
    assert table.loc["q_ok", "nirt_failure"] == "ok"


def test_failed_by_names_every_router_that_failed_the_query():
    true, cost = _frames()
    sel = {"nirt": np.array([2, 0, 0]), "zoib": np.array([2, 2, 0])}

    table = gap.build_gap_table(true, cost, sel)

    assert table["failed_by"].to_dict() == {
        "q_overspend": "nirt+zoib",
        "q_lost": "nirt",
        "q_ok": "none",
    }


# --------------------------------------------------------------------------- #
# summaries: which models fail, and where                                     #
# --------------------------------------------------------------------------- #
def _table_with_dataset():
    true, cost = _frames()
    sel = {"nirt": np.array([2, 0, 0]), "zoib": np.array([2, 2, 0])}
    table = gap.build_gap_table(true, cost, sel)
    table["dataset"] = ["gsm8k", "gsm8k", "mmlu"]
    return table


def test_by_model_table_counts_oracle_vs_router_pick_per_failure_type():
    out = gap.by_model_table(_table_with_dataset(), ["nirt"])

    row = out[(out.router == "nirt") & (out.oracle_model == "cheap")
              & (out.router_model == "big")].iloc[0]
    assert row["failure"] == "overspend"
    assert row["n"] == 1
    assert row["total_savings_gap"] == pytest.approx(0.100 - 0.001)
    assert out["n"].sum() == 3  # every query lands in exactly one row


def test_failure_rates_by_group_reports_each_router_separately():
    out = gap.failure_rates_by(_table_with_dataset(), ["nirt", "zoib"], by="dataset")

    g = out.set_index(["router", "dataset"])
    assert g.loc[("nirt", "gsm8k"), "n"] == 2
    assert g.loc[("nirt", "gsm8k"), "overspend_rate"] == pytest.approx(0.5)
    assert g.loc[("nirt", "gsm8k"), "quality_loss_rate"] == pytest.approx(0.5)
    assert g.loc[("nirt", "mmlu"), "fail_rate"] == 0.0
    assert g.loc[("zoib", "gsm8k"), "overspend_rate"] == pytest.approx(0.5)
    assert g.loc[("zoib", "gsm8k"), "quality_loss_rate"] == 0.0


# --------------------------------------------------------------------------- #
# reading_sample                                                              #
# --------------------------------------------------------------------------- #
def _table_with_text(n_overspend=6):
    idx = [f"q{i}" for i in range(n_overspend)] + ["q_ok"]
    true = pd.DataFrame(np.ones((len(idx), 3)), index=idx, columns=MODELS)
    cost = pd.DataFrame(np.tile(COST, (len(idx), 1)), index=idx, columns=MODELS)
    table = gap.build_gap_table(true, cost, {"nirt": np.array([2] * n_overspend + [0])})
    table["dataset"] = "gsm8k"
    table["query"] = [f"prompt number {i}" for i in range(n_overspend)] + ["fine prompt"]
    return table


def test_reading_sample_shows_failed_prompts_with_both_picks_and_skips_ok_rows():
    md = gap.reading_sample(_table_with_text(), ["nirt"], per_group=10)

    assert "prompt number 0" in md
    assert "nirt" in md and "overspend" in md and "oracle=cheap" in md
    assert "big" in md  # what the router paid for
    assert "fine prompt" not in md  # q_ok was not a failure


def test_reading_sample_takes_at_most_per_group_prompts_and_is_seed_stable():
    table = _table_with_text(n_overspend=6)

    a = gap.reading_sample(table, ["nirt"], per_group=2, seed=1)
    b = gap.reading_sample(table, ["nirt"], per_group=2, seed=1)

    assert a == b
    assert a.count("prompt number") == 2
    assert "n=6" in a  # the header still reports the full group size


def test_reading_sample_truncates_long_prompts_and_honours_max_prompts():
    table = _table_with_text(n_overspend=6)
    table["query"] = "x" * 500

    md = gap.reading_sample(table, ["nirt"], per_group=10, max_chars=40, max_prompts=3)

    assert "x" * 41 not in md
    assert md.count("x" * 40) == 3


def test_gap_table_counts_how_many_models_were_good_enough_per_query():
    true, cost = _frames()

    table = gap.build_gap_table(true, cost, {"nirt": np.array([0, 0, 0])})

    # q_overspend / q_ok: all three models right; q_lost: only "big"
    assert table["n_good_models"].to_dict() == {"q_overspend": 3, "q_lost": 1, "q_ok": 3}
