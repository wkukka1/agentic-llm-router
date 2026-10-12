"""``PlanForecaster``: estimate a :class:`~router.execution.plan.Plan`'s cost,
quality, and latency before it runs, for admission control.

Named ``PlanForecaster``, not ``PlanEvaluator`` -- repo-structure.md's
explicit naming fix. It forecasts for *admission*; scoring an already-executed
trace against ground truth is a completely different job
(``evaluation.routing.oracle.evaluate_router``), and sharing the word
"evaluator" across that serving/offline boundary would cause confusion in
every conversation about this codebase. :class:`RouterPlanForecaster` (below)
is the concrete one; :class:`~router.execution.routed.RoutedOrchestrator` uses
it to price a plan before reserving budget for it.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional

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

        Writes the estimates and utility back onto ``plan``, and moves a
        ``DRAFT`` plan to ``EVALUATED`` (a later status is left alone). ``within_budget`` is checked against ``ledger.remaining()``
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
        # only advance DRAFT -> EVALUATED: re-scoring an approved / executing /
        # finished plan refreshes its estimates but must not move it backwards
        # through the lifecycle (PLANFORECASTER-001)
        if plan.status is PlanStatus.DRAFT:
            plan.status = PlanStatus.EVALUATED
        return PlanScore(quality=quality, cost=cost.total_cost, latency=latency,
                         utility=utility, within_budget=within)


class RouterPlanForecaster(PlanForecaster):
    """The concrete forecaster: prices, quality and latency for a plan whose
    steps were routed by a :class:`~router.routing.base.RouterModel`.

    * **cost** per step, best source first: the step's own ``estimated_cost``;
      else its profile's per-token prices x the step's token estimates
      (``cost_model.predict_cost``); else the router's per-query cost vector
      (:attr:`RouterModel.default_model_costs`, the training-data mean);
      else unknown (``0`` and the forecast's ``confidence`` drops to ``0``).
      Each fallback below per-token prices lowers ``confidence``, which
      ``score_plan`` turns into the objective's risk term.
    * **quality**: the mean of the steps' ``predicted_quality`` (NaN steps
      skipped; NaN if none is known). A plan answers as well as its parts on
      average -- a deliberately simple aggregate for the MVP.
    * **latency**: the critical path through the step DAG -- independent
      steps run in parallel, so the plan takes as long as its slowest chain --
      using ``latency_table[model_id]`` seconds per step, else ``default_latency``.
      Dependencies on steps not in the plan (already done) cost nothing.
    """

    def __init__(self, router=None, *, cost_model=None,
                 latency_table: Optional[Mapping[str, float]] = None,
                 default_latency: float = 5.0):
        from ..llm.cost import CostModel

        self.router = router
        self.cost_model = cost_model or CostModel()
        self.latency_table = dict(latency_table or {})
        self.default_latency = float(default_latency)

    def forecast_cost(self, plan: "Plan") -> CostForecast:
        """Per step: its own ``estimated_cost`` when set, else
        :func:`~router.llm.cost.estimate_call_cost` at the step's token
        estimates -- the same estimate every single routed call is recorded
        with, so a plan and its calls are priced alike."""
        from ..llm.cost import estimate_call_cost, price_tokens

        router_costs = self.router.cost_by_model() if self.router is not None else {}
        input_cost = output_cost = total = 0.0
        confidence = 1.0
        for step in plan.steps:
            if step.estimated_cost > 0:
                total += step.estimated_cost
                continue
            model = step.model
            cost, source = estimate_call_cost(
                model, input_tokens=step.estimated_input_tokens,
                output_tokens=step.estimated_output_tokens, cost_model=self.cost_model,
                fallback=router_costs.get(str(getattr(model, "model_id", ""))),
            )
            if source == "profile":
                in_p, out_p = model.input_cost_per_token, model.output_cost_per_token
                input_cost += price_tokens(step.estimated_input_tokens, 0, in_p, out_p)
                output_cost += price_tokens(0, step.estimated_output_tokens, in_p, out_p)
                total += cost
            elif source == "router_prior":
                total += cost
                confidence = min(confidence, 0.5)
            else:
                confidence = 0.0
        return CostForecast(input_cost=input_cost, output_cost=output_cost,
                            total_cost=total, confidence=confidence)

    def forecast_quality(self, plan: "Plan") -> float:
        qs = [float(s.predicted_quality) for s in plan.steps if math.isfinite(s.predicted_quality)]
        return sum(qs) / len(qs) if qs else float("nan")

    def forecast_latency(self, plan: "Plan") -> float:
        steps = {s.step_id: s for s in plan.steps}
        finish: dict[str, float] = {}

        def done_at(sid: str) -> float:
            if sid not in steps:
                return 0.0
            if sid not in finish:
                finish[sid] = 0.0                       # cycle guard; validate() rejects cycles
                s = steps[sid]
                own = self.latency_table.get(str(getattr(s.model, "model_id", "")), self.default_latency)
                finish[sid] = own + max((done_at(d) for d in s.dependencies), default=0.0)
            return finish[sid]

        return max((done_at(sid) for sid in steps), default=0.0)
