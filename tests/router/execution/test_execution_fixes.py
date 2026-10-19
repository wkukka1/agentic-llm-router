"""Execution-layer fixes the plan orchestrator depends on: atomic hold
resizing (ORCHESTRATOR-001) and score_plan's status handling (PLANFORECASTER-001)."""

from __future__ import annotations

import pytest

from router.constraints import OptimizationObjective
from router.execution.budget import BudgetLedger, ExecutionLimits
from router.execution.forecast import CostForecast, PlanForecaster
from router.execution.orchestrator import Orchestrator, OrchestratorConfig
from router.execution.plan import Plan, PlanStatus, PlanStep
from router.execution.results import ExecutionResult
from router.execution.task import AgentTask, TaskStatus


# --------------------------------------------------------------------------- #
# BudgetLedger.resize                                                         #
# --------------------------------------------------------------------------- #
def test_resize_swaps_a_hold_in_place():
    ledger = BudgetLedger("root", 10.0)
    assert ledger.reserve("a", 6.0)
    assert ledger.resize("a", 5.0) and ledger.reserved == pytest.approx(5.0)
    assert ledger.resize("a", 9.0) and ledger.reserved == pytest.approx(9.0)   # grows into free budget


def test_resize_that_does_not_fit_leaves_the_hold_untouched():
    ledger = BudgetLedger("root", 10.0)
    ledger.reserve("a", 6.0)
    ledger.reserve("b", 3.0)
    assert not ledger.resize("a", 8.0)                 # only 1.0 free beyond a's own 6.0
    assert ledger.reserved == pytest.approx(9.0)


def test_resize_without_a_hold_is_a_reserve():
    ledger = BudgetLedger("root", 10.0)
    assert ledger.resize("a", 4.0) and ledger.remaining() == pytest.approx(6.0)
    assert not ledger.resize("b", 7.0)


# --------------------------------------------------------------------------- #
# Orchestrator.run: ORCHESTRATOR-001                                          #
# --------------------------------------------------------------------------- #
class _Fixed(Orchestrator):
    def plan(self, task):
        return Plan("p", task.task_id, steps=[PlanStep("s", "x", None, estimated_cost=5.0)])

    def execute(self, plan):
        return ExecutionResult(actual_cost=5.0)

    def replan(self, plan):
        return plan


def test_child_keeps_its_allocation_while_resizing_to_the_plan():
    """The repro: total 10, child allocated 6, plan needs 5. A sibling that
    tries to grab budget the moment the child's hold is released must not be
    able to take the child's allocation -- run() swaps the hold atomically."""
    ledger = BudgetLedger("root", 10.0)
    orch = _Fixed(OrchestratorConfig("o", "0", llm=None, ledger=ledger,
                                     limits=ExecutionLimits(max_cost=10.0)))
    orch.task = AgentTask("root", None, "root", "parent prompt")
    child = AgentTask("child", "root", "root", "child prompt", depth=1, allocated_budget=6.0)
    assert ledger.reserve("child", 6.0)  # the allocation a parent holds for its child
    child_orch = orch

    sibling = {}
    real_release = ledger.release

    def release_then_sibling_reserves(task_id):
        real_release(task_id)
        if task_id == "child":
            sibling["got"] = ledger.reserve("sibling", 8.0)

    ledger.release = release_then_sibling_reserves
    res = child_orch.run(child)
    assert res.success and child.status is TaskStatus.COMPLETED
    assert ledger.spent == pytest.approx(5.0)


def test_over_budget_plan_drops_the_allocation_hold():
    ledger = BudgetLedger("root", 10.0)

    class _Big(_Fixed):
        def plan(self, task):
            return Plan("p", task.task_id, steps=[PlanStep("s", "x", None, estimated_cost=50.0)])

    orch = _Big(OrchestratorConfig("o", "0", llm=None, ledger=ledger, limits=ExecutionLimits()))
    orch.task = AgentTask("root", None, "root", "p")
    child = AgentTask("child", "root", "root", "c", depth=1, allocated_budget=6.0)
    assert ledger.reserve("child", 6.0)
    res = orch.run(child)
    assert res.error_code == "over_budget" and ledger.reserved == 0.0


# --------------------------------------------------------------------------- #
# PlanForecaster.score_plan: PLANFORECASTER-001                               #
# --------------------------------------------------------------------------- #
class _ConstForecaster(PlanForecaster):
    def forecast_cost(self, plan):
        return CostForecast(total_cost=1.0)

    def forecast_quality(self, plan):
        return 0.8

    def forecast_latency(self, plan):
        return 2.0


@pytest.mark.parametrize("status", [PlanStatus.APPROVED, PlanStatus.EXECUTING,
                                    PlanStatus.COMPLETED, PlanStatus.FAILED])
def test_rescoring_a_later_stage_plan_keeps_its_status(status):
    plan = Plan("p", "t", status=status)
    _ConstForecaster().score_plan(plan, OptimizationObjective())
    assert plan.status is status and plan.estimated_cost == 1.0      # estimates still refreshed


def test_scoring_a_draft_marks_it_evaluated():
    plan = Plan("p", "t")
    _ConstForecaster().score_plan(plan, OptimizationObjective())
    assert plan.status is PlanStatus.EVALUATED
