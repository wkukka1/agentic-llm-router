"""The production ``Orchestrator``: plan a task, execute the plan, replan on
failure, spawn depth-bounded child orchestrators.

Distinct from the working
:class:`router.agentic.orchestrator.RecursiveOrchestrator` /
:class:`~router.agentic.orchestrator.LangChainToolOrchestrator`, which
decompose and dispatch directly with no intermediate :class:`~router.execution.plan.Plan`
to forecast, approve, or replan -- recursion there is a bare depth counter,
not :meth:`spawn`/:meth:`can_spawn` checked against a shared
:class:`~router.execution.budget.BudgetLedger`. Nothing constructs this
``Orchestrator`` yet.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .budget import BudgetLedger, ExecutionLimits
from .plan import Plan
from .results import ExecutionResult

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from ..llm.profile import LLMProfile
    from ..tools.base import Tool
    from ..tools.router_tool import RouterTool
    from .forecast import PlanForecaster
    from .task import AgentTask


@dataclass
class OrchestratorConfig:
    name: str
    version: str
    llm: "LLMProfile"
    ledger: BudgetLedger
    limits: ExecutionLimits
    tools: list["Tool"] = field(default_factory=list)


class Orchestrator(abc.ABC):
    def __init__(self, config: OrchestratorConfig, *, router_tool: Optional["RouterTool"] = None,
                 plan_forecaster: Optional["PlanForecaster"] = None):
        self.name = config.name
        self.version = config.version
        self.llm = config.llm
        self.ledger = config.ledger
        self.limits = config.limits
        self.tools = list(config.tools)
        self.router_tool = router_tool
        self.plan_forecaster = plan_forecaster
        self.task: Optional["AgentTask"] = None

    @abc.abstractmethod
    def plan(self, task: "AgentTask") -> Plan:
        raise NotImplementedError

    @abc.abstractmethod
    def execute(self, plan: Plan) -> ExecutionResult:
        raise NotImplementedError

    @abc.abstractmethod
    def replan(self, plan: Plan) -> Plan:
        raise NotImplementedError

    def run(self, task: "AgentTask") -> ExecutionResult:
        """The default plan -> execute template. Subclasses needing bespoke
        control flow (retry-then-replan, partial execution) override this
        directly rather than fighting the template."""
        self.task = task
        return self.execute(self.plan(task))

    def can_spawn(self, child: "AgentTask") -> bool:
        return (
            child.depth < self.limits.max_depth
            and self.ledger.remaining() > 0
        )

    def spawn(self, child: "AgentTask") -> "Orchestrator":
        if not self.can_spawn(child):
            raise ValueError(
                f"cannot spawn child task {child.task_id!r} at depth {child.depth} "
                f"(max_depth={self.limits.max_depth}, remaining budget={self.ledger.remaining()})"
            )
        return type(self)(
            OrchestratorConfig(
                name=self.name, version=self.version, llm=self.llm,
                ledger=self.ledger, limits=self.limits, tools=self.tools,
            ),
            router_tool=self.router_tool, plan_forecaster=self.plan_forecaster,
        )


class OrchestratorFactory:
    def create(self, config: OrchestratorConfig, orchestrator_cls: type[Orchestrator],
               **kwargs) -> Orchestrator:
        """No default ``Orchestrator`` implementation exists to construct
        without one -- pass the concrete subclass explicitly, unlike the
        diagram's zero-arg ``create(config)`` (which implies a single
        registered implementation, the way :mod:`router.routing.registry`
        picks a ``Router`` by ``kind``; no such registry exists for
        orchestrators yet)."""
        return orchestrator_cls(config, **kwargs)
