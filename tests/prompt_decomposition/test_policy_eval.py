"""`policy_eval`: scoring a selection rule with the project's own metrics.

These numbers are the ones the margin policy is argued from, so the tests are
arithmetic, not smoke. The fixture is built so every answer can be worked out
by hand and written in the assertion.

The load-bearing claim is the one in the module docstring: handing
`routing_evaluation` an indicator matrix makes its argmax equal our selection,
so quality, cost, regret and oracle-hit come out of Will's code against the
same ground truth rather than a fork of the definitions. The first test class
pins exactly that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.prompt_decomposition.policy_eval import (
    frontier_gain,
    margin_for_target_saving,
    margin_selection,
    margin_sweep,
    score_selection,
)
from evaluation.routing.oracle import routing_evaluation

MODELS = ["strong", "mid", "cheap"]

# Four queries. `mid` is a hair below `strong` on q0-q2 and ties on q3; `cheap`
# is well below everywhere except q3, where everything is correct.
TRUE = np.array([
    [1.0, 0.9, 0.2],
    [1.0, 0.9, 0.3],
    [1.0, 0.9, 0.1],
    [1.0, 1.0, 1.0],
])
# cost is per-model and constant across queries: 100x, 10x, 1x
COST = np.tile(np.array([1.00, 0.10, 0.01]), (4, 1))
# a perfect predictor, so selection differences are the rule's, not the model's
PRED = TRUE.copy()


class TestSelectionRule:
    def test_a_zero_margin_is_argmax(self):
        """The knob has to contain the incumbent policy or the comparison in
        `margin_sweep` is not a like-for-like one.

        q3 ties at 1.0 across all three models and still goes to column 0, not
        to the cheapest: plain `argmax` breaks ties by column order. This once
        read `[0, 0, 0, 2]`, from a private copy of the rule that took the
        tie-break saving for free and so understated what the baseline costs.
        """
        assert margin_selection(PRED, COST, margin=0.0).tolist() == [0, 0, 0, 0]

    def test_a_zero_margin_is_the_served_rule_itself(self):
        """Not merely equal to argmax -- the same function the router calls."""
        from router.nirt.routing_decision import routing_decision

        assert margin_selection(PRED, COST, margin=0.0).tolist() == \
            routing_decision(PRED, model_costs=COST.mean(axis=0)).tolist()

    def test_a_margin_takes_the_cheapest_within_it(self):
        picks = margin_selection(PRED, COST, margin=0.1)
        assert picks.tolist() == [1, 1, 1, 2]   # mid is within 0.1 on q0-q2

    def test_a_model_outside_the_margin_is_never_taken(self):
        """`cheap` is 0.8 below the best on q0 and must not be picked at 0.1,
        however little it costs."""
        assert margin_selection(PRED, COST, margin=0.1)[0] == 1

    def test_a_wide_margin_reaches_the_cheapest_model(self):
        assert margin_selection(PRED, COST, margin=1.0).tolist() == [2, 2, 2, 2]

    def test_ties_on_cost_keep_the_higher_quality_model(self):
        flat = np.ones((1, 3)) * 0.5
        pred = np.array([[0.8, 0.79, 0.78]])
        assert margin_selection(pred, flat, margin=0.05)[0] == 0

    def test_lam_folds_cost_into_the_utility_the_margin_is_measured_on(self):
        """With `lam > 0` the margin applies to quality-minus-cost, not to raw
        quality -- the same definition `OptimizationObjective.utility` uses."""
        assert margin_selection(PRED, COST, margin=0.0, lam=1.0)[0] == 1  # 0.9-0.1 > 1.0-1.0


class TestItUsesTheProjectMetrics:
    def test_an_indicator_matrix_makes_the_scorer_agree_with_our_pick(self):
        picks = margin_selection(PRED, COST, margin=0.1)
        m = score_selection(picks, TRUE, COST, MODELS)
        # quality of [mid, mid, mid, cheap] = (0.9 + 0.9 + 0.9 + 1.0) / 4
        assert m["mean_selected_quality"] == pytest.approx(0.925)

    def test_zero_margin_reproduces_the_incumbent_scored_the_normal_way(self):
        """Scoring argmax through the indicator must equal scoring the
        prediction matrix directly, or the detour changes the numbers."""
        direct = routing_evaluation(PRED, TRUE, COST, MODELS, lam=0.0)
        via = score_selection(margin_selection(PRED, COST, margin=0.0), TRUE, COST, MODELS)
        for key in ("mean_selected_quality", "mean_regret", "oracle_hit_rate_any_best"):
            assert via[key] == pytest.approx(direct[key])

    def test_regret_is_the_gap_to_the_best_available_model(self):
        """Selecting `cheap` everywhere: the best is 1.0 on every query, so
        regret is the mean of (1-0.2, 1-0.3, 1-0.1, 1-1.0)."""
        m = score_selection(np.full(4, 2), TRUE, COST, MODELS)
        assert m["mean_regret"] == pytest.approx((0.8 + 0.7 + 0.9 + 0.0) / 4)

    def test_cost_is_reported_for_the_model_actually_selected(self):
        cheap = score_selection(np.full(4, 2), TRUE, COST, MODELS)
        strong = score_selection(np.full(4, 0), TRUE, COST, MODELS)
        assert cheap["mean_selected_cost_per_1k"] < strong["mean_selected_cost_per_1k"]


class TestTheSweepTable:
    @pytest.fixture
    def table(self):
        return margin_sweep(PRED, TRUE, COST, MODELS, margins=(0.0, 0.1, 1.0))

    def test_it_carries_its_own_control_row(self, table):
        """Any claim about the margin rule is a difference against this row."""
        assert table.loc[0, "policy"] == "argmax (incumbent)"

    def test_the_reference_rows_are_present(self, table):
        assert table["policy"].tolist()[-2:] == ["always strong", "oracle (cheapest correct)"]

    def test_a_wider_margin_is_cheaper_and_no_better(self, table):
        """Over the margin rows only -- the two reference rows are appended
        after them and are not part of the sweep."""
        margins = table[table["policy"].str.startswith(("argmax", "margin"))]
        assert margins["cost_per_1k"].is_monotonic_decreasing
        assert margins["quality"].is_monotonic_decreasing

    def test_the_oracle_row_is_the_ceiling(self, table):
        oracle = table.iloc[-1]
        assert oracle["regret"] == pytest.approx(0.0)
        assert oracle["quality"] >= table["quality"].max() - 1e-9

    def test_the_oracle_takes_the_cheapest_among_equals(self, table):
        """q3 has three correct answers; a quality-only oracle would bill for
        `strong` there and overstate the cost a router has to beat. It is not
        cheaper than argmax here only because argmax breaks its own ties on
        cost too -- so the comparison that bites is against the fixed model."""
        oracle, fixed = table.iloc[-1], table.iloc[-2]
        assert oracle["quality"] == pytest.approx(fixed["quality"])
        assert oracle["cost_per_1k"] < fixed["cost_per_1k"]
        assert oracle["top_model_share"] == pytest.approx(0.75)   # q3 went to `cheap`


class TestFrontierGain:
    """The column that says whether the rule bought anything.

    A cost-quality point means nothing alone: randomly splitting traffic
    between two models traces the straight line between them for free.
    """

    @pytest.fixture
    def per_model(self):
        return pd.DataFrame(
            {"quality": [1.0, 0.925, 0.4], "cost_per_1k": [1.00, 0.10, 0.01]}, index=MODELS)

    def test_a_point_on_the_mixture_line_scores_zero(self, per_model):
        mid = pd.DataFrame({"policy": ["halfway"], "quality": [0.7], "cost_per_1k": [0.505]})
        out = frontier_gain(mid, "cheap", "strong", per_model)
        assert out.loc[0, "vs_mixture"] == pytest.approx(0.0)   # (0.4 + 1.0) / 2

    def test_a_point_above_the_line_is_the_gain(self, per_model):
        better = pd.DataFrame({"policy": ["ours"], "quality": [0.8], "cost_per_1k": [0.505]})
        assert frontier_gain(better, "cheap", "strong", per_model).loc[0, "vs_mixture"] \
            == pytest.approx(0.1)

    def test_the_endpoints_score_zero(self, per_model):
        ends = pd.DataFrame({"policy": ["a", "b"], "quality": [0.4, 1.0],
                             "cost_per_1k": [0.01, 1.00]})
        assert frontier_gain(ends, "cheap", "strong", per_model)["vs_mixture"].abs().max() \
            == pytest.approx(0.0)

    def test_a_point_outside_the_cost_span_is_not_scored(self, per_model):
        """Nothing to interpolate against, so the honest answer is NaN rather
        than an extrapolated baseline that flatters whatever sits there."""
        out = frontier_gain(
            pd.DataFrame({"policy": ["dearer"], "quality": [1.0], "cost_per_1k": [5.0]}),
            "cheap", "strong", per_model)
        assert pd.isna(out.loc[0, "vs_mixture"])

    def test_it_does_not_mutate_the_table_it_is_given(self, per_model):
        table = pd.DataFrame({"policy": ["x"], "quality": [0.8], "cost_per_1k": [0.5]})
        frontier_gain(table, "cheap", "strong", per_model)
        assert list(table.columns) == ["policy", "quality", "cost_per_1k"]


def test_an_endpoint_on_the_boundary_is_not_lost_to_float_noise():
    """Observed on SPROUT: the `always <strongest>` row read NaN because its
    cost came from `routing_evaluation` while the endpoint's came from a mean
    over the same numbers, and the two differed in the last bits."""
    per_model = pd.DataFrame(
        {"quality": [0.4, 1.0], "cost_per_1k": [0.01, 1.0]}, index=["cheap", "strong"])
    nudged = pd.DataFrame({"policy": ["always strong"], "quality": [1.0],
                           "cost_per_1k": [1.0 + 1e-13]})
    out = frontier_gain(nudged, "cheap", "strong", per_model)
    assert out.loc[0, "vs_mixture"] == pytest.approx(0.0, abs=1e-9)


class TestMarginForTargetSaving:
    """Turning "spend 50% less" into a margin, against measured data.

    The reason this exists rather than a hard-coded preset: the same margin
    buys different savings on different pools (0.1 cut cost 68% on RouterBench,
    81% on SPROUT), so a preset is correct only on the pool it was read off.
    """

    @pytest.fixture
    def table(self):
        return margin_sweep(PRED, TRUE, COST, MODELS, margins=(0.0, 0.05, 0.2))

    def test_it_returns_the_smallest_margin_that_clears_the_target(self, table):
        """Margin is quality given up, so the cheapest way to hit a target is
        the one to report."""
        loose = margin_for_target_saving(table, 0.10)
        tight = margin_for_target_saving(table, 0.90)
        assert loose["margin"] <= tight["margin"]

    def test_it_reports_what_the_target_costs_in_quality(self, table):
        hit = margin_for_target_saving(table, 0.50)
        assert hit["saving"] >= 0.50
        assert hit["quality_given_up"] >= 0.0
        assert hit["baseline_policy"].startswith("always ")

    def test_a_zero_target_is_met_by_doing_nothing(self, table):
        assert margin_for_target_saving(table, 0.0)["margin"] == 0.0

    def test_an_unreachable_target_raises_and_says_what_was_possible(self, table):
        with pytest.raises(ValueError, match="no margin in this sweep saves 99%"):
            margin_for_target_saving(table, 0.99)

    def test_a_nonsense_target_is_refused(self, table):
        with pytest.raises(ValueError, match="target_saving must be in"):
            margin_for_target_saving(table, 1.5)

    def test_a_table_with_no_fixed_row_is_refused(self):
        bare = pd.DataFrame({"policy": ["margin=0.05"], "quality": [0.9], "cost_per_1k": [1.0]})
        with pytest.raises(ValueError, match="no 'always <model>' row"):
            margin_for_target_saving(bare, 0.5)
