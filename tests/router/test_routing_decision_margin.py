"""`routing_decision(margin=...)`: the served selection rule.

`margin=0` has to be a strict no-op -- this function is what every routing call
in the repo goes through, so anything that changes its default changes every
served decision at once. The first class exists to pin that, and is the reason
the cost tie-break only switches on above zero.

The rule itself is small; the reason it is here is not. Argmax on this
repo's rankers collapses onto the single strongest model and scores about what
no router would score; a margin recovers most of the cost. Figures live in
docs/prompt_decomposition_experiments.md, not in copies here.
"""

from __future__ import annotations

import numpy as np
import pytest

from router.nirt.routing_decision import routing_decision

# strong is best everywhere but dearest; mid is a hair behind at a tenth the
# price; cheap is far behind. q3 is a three-way tie.
PRED = np.array([
    [1.0, 0.9, 0.2],
    [1.0, 0.9, 0.3],
    [1.0, 1.0, 1.0],
])
COSTS = np.array([1.0, 0.1, 0.01])


class TestZeroMarginIsUntouched:
    def test_it_is_plain_argmax(self):
        assert routing_decision(PRED, model_costs=COSTS).tolist() == [0, 0, 0]

    def test_ties_still_go_to_the_lowest_column_not_the_cheapest(self):
        """The documented `np.argmax` semantics. If this flips, every existing
        routing decision on a tie silently changes."""
        assert routing_decision(PRED, model_costs=COSTS)[2] == 0

    def test_it_needs_no_costs_at_all(self):
        assert routing_decision(PRED).tolist() == [0, 0, 0]

    def test_lam_still_works_on_its_own(self):
        # util = [1-1, 0.9-0.1, 0.2-0.01] -> mid wins on q0
        assert routing_decision(PRED, lam=1.0, model_costs=COSTS)[0] == 1


class TestTheMargin:
    def test_it_takes_the_cheapest_within_the_margin(self):
        assert routing_decision(PRED, model_costs=COSTS, margin=0.1).tolist() == [1, 1, 2]

    def test_a_candidate_outside_the_margin_is_never_taken(self):
        """`cheap` is 0.8 behind on q0 and must not be picked at margin 0.1,
        however little it costs."""
        assert routing_decision(PRED, model_costs=COSTS, margin=0.1)[0] == 1

    def test_a_wide_margin_reaches_the_cheapest(self):
        assert routing_decision(PRED, model_costs=COSTS, margin=1.0).tolist() == [2, 2, 2]

    def test_a_cost_tie_falls_back_to_the_argmax_answer(self):
        flat = np.array([1.0, 1.0, 1.0])
        assert routing_decision(PRED, model_costs=flat, margin=0.5).tolist() == [0, 0, 0]

    def test_it_composes_with_lam(self):
        """Different levers: `lam` prices cost into the utility, `margin`
        spends bounded quality to buy a cheaper model."""
        picked = routing_decision(PRED, lam=0.5, model_costs=COSTS, margin=0.05)
        assert picked.tolist() == [1, 1, 2]


class TestItRefusesWhatItCannotDo:
    def test_a_margin_without_prices_is_an_error(self):
        with pytest.raises(ValueError, match="needs model_costs"):
            routing_decision(PRED, margin=0.1)

    def test_a_negative_margin_is_an_error(self):
        with pytest.raises(ValueError, match="margin must be >= 0"):
            routing_decision(PRED, model_costs=COSTS, margin=-0.1)

    def test_an_unpriced_model_is_not_treated_as_free(self):
        """NaN must not win the cheapness sort -- that would route everything
        to whichever model nobody has priced yet. Here `mid` is unpriced and
        inside the margin, so it is the one a NaN-wins bug would select."""
        costs = np.array([1.0, np.nan, 0.5])
        picked = routing_decision(PRED, model_costs=costs, margin=0.2)[0]
        assert picked != 1        # the unpriced candidate
        assert picked == 0        # the dearer but priced one, on quality

    def test_per_cell_costs(self):
        """A [Q, M] cost matrix (predicted per-query cost) picks per row."""
        costs = np.array([[1.0, 0.1, 0.01], [1.0, 2.0, 0.01], [0.5, 0.5, 0.1]])
        assert routing_decision(PRED, model_costs=costs, margin=0.1).tolist() == [1, 0, 2]


class TestMaskingStillHolds:
    def test_an_ineligible_model_is_never_selected(self):
        elig = np.array([[True, False, True]] * 3)
        assert 1 not in routing_decision(PRED, model_costs=COSTS, margin=0.1, eligible=elig)

    def test_a_missing_prediction_is_never_selected(self):
        pred = np.array([[np.nan, 0.9, 0.2]])
        assert routing_decision(pred, model_costs=COSTS, margin=0.5)[0] != 0

    def test_a_row_with_nothing_selectable_still_falls_back_to_column_zero(self):
        """The caller detects this via `fallback`; the margin path must not
        change where an unselectable row lands."""
        pred = np.array([[np.nan, np.nan, np.nan]])
        assert routing_decision(pred, model_costs=COSTS, margin=0.1)[0] == 0


class TestItReachesTheRouter:
    """`routing_decision` is reached through `Router`, so the parameter has to
    survive the trip and the result has to say which rule actually ran."""

    def _router(self):
        from tests.router.test_router_pipeline import FakeTextRouter

        return FakeTextRouter()

    def test_the_result_records_the_margin_that_was_applied(self):
        out = self._router().route_text(["anything"], model_costs=[1.0, 0.1, 0.01],
                                        margin=0.05)
        assert out.margin == 0.05

    def test_the_margin_changes_which_model_is_dispatched(self):
        r, prices = self._router(), [1.0, 0.1, 0.01]
        plain = r.route_text(["anything"], model_costs=prices).selected_model_ids
        cheap = r.route_text(["anything"], model_costs=prices, margin=0.5).selected_model_ids
        assert plain != cheap

    def test_an_unpriced_pool_downgrades_to_argmax_with_a_warning(self):
        """Rather than raising in the middle of a served request: the caller
        gets a decision, and `RoutingResult.margin` says 0.0 so a log or trace
        never claims a rule that did not run."""
        r = self._router()
        assert r.default_model_costs is None, "fixture is meant to be unpriced"
        with pytest.warns(UserWarning, match="no model cost vector"):
            assert r.route_text(["anything"], margin=0.05).margin == 0.0


class TestTheTwoImplementationsAgree:
    """`MarginRoutingPolicy` states the same rule over `ModelScore` objects.

    Two implementations of one rule is exactly the bug that corrupted the
    offline baselines (a private copy in `policy_eval` had drifted to
    cost-breaking ties at margin=0). The policy cannot simply delegate -- it
    ranks objects, not a matrix -- so the agreement is pinned here instead.
    """

    def _decide(self, margin, pred, costs):
        from decompose.signals import PromptSignals
        from router.constraints import OptimizationObjective, RoutingConstraints
        from router.context import RoutingContext, RoutingRequest
        from router.decision import ModelScore
        from router.llm.profile import LLMProfile
        from router.policy import MarginRoutingPolicy

        scores = [
            ModelScore(model=LLMProfile(name=f"m{i}", provider="t", model_id=f"m{i}",
                                        version="1"),
                       score=q, expected_quality=q, expected_cost=c, confidence=0.5)
            for i, (q, c) in enumerate(zip(pred, costs))
        ]
        ctx = RoutingContext(
            request=RoutingRequest(
                request_id="r", prompt="p",
                constraints=RoutingConstraints(
                    objective=OptimizationObjective(quality_weight=1.0, cost_weight=0.0))),
            signals=PromptSignals())
        return MarginRoutingPolicy(margin=margin).decide(ctx, scores).selected_model_id

    @pytest.mark.parametrize("margin", [0.05, 0.1, 0.5])
    @pytest.mark.parametrize("row", [0, 1])
    def test_same_pick_as_the_served_rule(self, margin, row):
        pred = PRED[row]
        expected = routing_decision(pred[None, :], model_costs=COSTS, margin=margin)[0]
        assert self._decide(margin, pred, COSTS) == f"m{expected}"

    def test_the_margin_zero_tie_break_differs_on_purpose(self):
        """The one place they diverge, recorded rather than left to be found.

        On an exact utility tie at margin=0 the matrix rule keeps argmax's
        column order, because it must stay a no-op for every existing caller.
        The policy has no column order to preserve -- it ranks objects -- and
        takes the cheapest. Choosing `MarginRoutingPolicy` is already opting
        into cost-seeking, so that is the right answer for it.
        """
        tied = np.array([1.0, 1.0, 1.0])
        assert routing_decision(tied[None, :], model_costs=COSTS, margin=0.0)[0] == 0
        assert self._decide(0.0, tied, COSTS) == "m2"        # the cheapest


class TestPerRequestOnTheLivePath:
    """`AgenticRouter.with_margin` -- the live path's per-request knob.

    A copy rather than a mutable attribute, so two concurrent callers with
    different budgets cannot change each other's routing.
    """

    def _router(self, margin=0.0):
        from router.agentic.router import AgenticRouter
        from tests.router.test_router_pipeline import FakeTextRouter

        return AgenticRouter(FakeTextRouter(), margin=margin)

    def test_it_returns_a_copy_with_the_new_budget(self):
        base = self._router(0.0)
        wide = base.with_margin(0.2)
        assert wide.margin == 0.2
        assert base.margin == 0.0, "the original must be untouched"
        assert wide is not base

    def test_the_copy_keeps_everything_else(self):
        base = self._router(0.0)
        wide = base.with_margin(0.2)
        assert wide.router is base.router
        assert wide.triage is base.triage
        assert wide.max_depth == base.max_depth

    def test_a_negative_budget_is_refused(self):
        with pytest.raises(ValueError, match="margin must be >= 0"):
            self._router().with_margin(-0.1)

    def test_a_copy_cannot_inherit_another_run_state(self):
        """A budget change must not inherit routes decided under the old budget.

        That used to need `_shallow_copy` to clear `_decisions`/`_calls`. Per-run
        state now lives in a `contextvars.ContextVar` rather than on the router
        (see `router.agentic.router._RunState`), so the property holds by
        construction -- this pins that it stays off the instance.
        """
        wide = self._router(0.0).with_margin(0.2)
        assert not hasattr(wide, "_decisions"), "per-run state is back on the instance"
        assert not hasattr(wide, "_calls")
