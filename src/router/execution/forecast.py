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
from typing import TYPE_CHECKING, Optional

from ..decision import ModelScore
from .plan import PlanStatus

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from ..constraints import OptimizationObjective
    from .budget import BudgetLedger, ExecutionLimits
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
    #: ``None`` when no ledger / limits were given to check against -- an
    #: unchecked score must not read as a pass
    within_budget: Optional[bool] = None


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

    def score_plan(
        self,
        plan: "Plan",
        objective: "OptimizationObjective",
        *,
        ledger: Optional["BudgetLedger"] = None,
        limits: Optional["ExecutionLimits"] = None,
    ) -> PlanScore:
        """Combine the three forecasts into one :class:`PlanScore` via
        ``objective.utility`` -- the same function the router optimizes, risk
        term included (``CostForecast.confidence`` feeds ``1 - confidence``), so
        a plan's admission score and a model's routing score never drift apart.

        Writes the estimates, utility and ``EVALUATED`` status back onto
        ``plan``. ``within_budget`` is checked against ``ledger.remaining()``
        and ``limits`` when given."""
        cost = self.forecast_cost(plan)
        quality = self.forecast_quality(plan)
        latency = self.forecast_latency(plan)
        utility = objective.utility(ModelScore(
            model=None, score=quality, expected_quality=quality,
            expected_cost=cost.total_cost, expected_latency=latency,
            confidence=cost.confidence,
        ))
        within: Optional[bool] = None
        if ledger is not None or limits is not None:
            within = True
            if ledger is not None:
                within = within and cost.total_cost <= ledger.remaining()
            if limits is not None:
                within = within and cost.total_cost <= limits.max_cost and latency <= limits.max_latency
        plan.estimated_cost = cost.total_cost
        plan.estimated_quality = quality
        plan.estimated_latency = latency
        plan.utility = utility
        plan.status = PlanStatus.EVALUATED
        return PlanScore(quality=quality, cost=cost.total_cost, latency=latency,
                         utility=utility, within_budget=within)
