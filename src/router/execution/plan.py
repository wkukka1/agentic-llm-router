"""A ``Plan`` is a sequence of ``PlanStep``\\ s an orchestrator commits to
before executing -- built by `RoutedOrchestrator` (:mod:`router.execution.routed`). The
:class:`router.agentic.orchestrator.RecursiveOrchestrator` decomposes and
dispatches directly, with no intermediate plan object to forecast or approve
first; that's the gap this module (and :mod:`router.execution.forecast`)
fills."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from ..llm.profile import LLMProfile


class PlanStatus(enum.Enum):
    DRAFT = "draft"
    EVALUATED = "evaluated"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PlanStep:
    step_id: str
    prompt: str
    model: "LLMProfile"
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_cost: float = 0.0
    dependencies: list[str] = field(default_factory=list)
    #: the router's predicted quality for ``model`` on ``prompt`` (NaN: unknown)
    #: -- what a forecaster aggregates into the plan's quality
    predicted_quality: float = float("nan")
    #: fallback model ids, best first, for ``Orchestrator.replan`` to move a
    #: failed step to without re-routing it
    alternatives: list[str] = field(default_factory=list)
    #: ``(low, high, limit)`` words the step's answer should take (``None``:
    #: no length guidance) -- see ``router.agentic.decompose.Budget``
    budget: Optional[tuple[int, int, int]] = None


@dataclass
class Plan:
    """``estimated_*`` / ``utility`` / ``status`` are written by
    :meth:`~router.execution.forecast.PlanForecaster.score_plan`. Holds should
    use :meth:`budget_estimate`, which never under-reserves relative to the
    steps."""

    plan_id: str
    task_id: str
    steps: list[PlanStep] = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_quality: float = 0.0
    estimated_latency: float = 0.0
    utility: float = 0.0
    status: PlanStatus = PlanStatus.DRAFT
    revision: int = 0

    @property
    def steps_estimated_cost(self) -> float:
        return sum(s.estimated_cost for s in self.steps)

    def budget_estimate(self) -> float:
        """The larger of the plan-level estimate and the sum of its steps."""
        return max(float(self.estimated_cost), self.steps_estimated_cost)

    def validate(self) -> None:
        """Raise ``ValueError`` on duplicate step ids, dependencies on unknown
        steps, or a dependency cycle."""
        ids = [s.step_id for s in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError(f"plan {self.plan_id!r} has duplicate step ids")
        deps = {s.step_id: list(s.dependencies) for s in self.steps}
        unknown = sorted({d for ds in deps.values() for d in ds} - set(ids))
        if unknown:
            raise ValueError(f"plan {self.plan_id!r} depends on unknown steps {unknown}")
        state: dict[str, int] = {}          # 1 = visiting, 2 = done

        def visit(sid: str) -> None:
            if state.get(sid) == 2:
                return
            if state.get(sid) == 1:
                raise ValueError(f"plan {self.plan_id!r} has a dependency cycle through {sid!r}")
            state[sid] = 1
            for d in deps[sid]:
                visit(d)
            state[sid] = 2

        for sid in ids:
            visit(sid)
