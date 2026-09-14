"""``PlanForecaster``: estimate a :class:`~router.execution.plan.Plan`'s cost,
quality, and latency before it runs, for admission control.

Named ``PlanForecaster``, not ``PlanEvaluator`` -- repo-structure.md's
explicit naming fix. It forecasts for *admission*; scoring an already-executed
trace against ground truth is a completely different job
(``evaluation.routing.oracle.evaluate_router``), and sharing the word
"evaluator" across that serving/offline boundary would cause confusion in
every conversation about this codebase. Nothing calls this yet -- there is no
admission-control step in the live agentic flow today.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from ..constraints import OptimizationObjective
    from .plan import Plan


@dataclass
class CostForecast:
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    confidence: float = 1.0


@dataclass
class PlanScore:
    quality: float
    cost: float
    latency: float
    utility: float
    within_budget: bool


class PlanForecaster(abc.ABC):
    @abc.abstractmethod
    def forecast_cost(self, plan: "Plan") -> CostForecast:
        raise NotImplementedError

    @abc.abstractmethod
    def forecast_quality(self, plan: "Plan") -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def forecast_latency(self, plan: "Plan") -> float:
        raise NotImplementedError

    def score_plan(self, plan: "Plan", objective: "OptimizationObjective") -> PlanScore:
        """Combine the three forecasts into one :class:`PlanScore` via
        ``objective`` -- the same :class:`~router.constraints.OptimizationObjective`
        the router itself optimizes, so a plan's admission score and a
        model's routing score never drift apart."""
        cost = self.forecast_cost(plan)
        quality = self.forecast_quality(plan)
        latency = self.forecast_latency(plan)
        utility = (
            objective.quality_weight * quality
            - objective.cost_weight * cost.total_cost
            - objective.latency_weight * latency
        )
        # within_budget needs an ExecutionLimits/BudgetLedger to check against,
        # which this signature doesn't receive -- always True is a placeholder
        # until admission control (the caller of score_plan) exists.
        return PlanScore(quality=quality, cost=cost.total_cost, latency=latency,
                         utility=utility, within_budget=True)
