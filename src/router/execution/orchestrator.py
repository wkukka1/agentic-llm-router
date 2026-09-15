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

:meth:`can_spawn`'s ``max_depth`` convention (deepest node allowed to exist,
not deepest child allowed to be spawned) is deliberately kept aligned with
``AgenticRouter._solve``'s (XA-08) -- so if this ever does get wired up, the
same ``max_depth`` value means the same effective depth in both places.

An ``Orchestrator`` instance runs one task at a time (``self.task``); spawn a
child orchestrator per child task.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .budget import BudgetLedger, ExecutionLimits
from .plan import Plan, PlanStatus
from .results import ExecutionResult
from .task import TaskStatus

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
        limits, ledger = config.limits, config.ledger
        if math.isfinite(limits.max_cost) and ledger.total > limits.max_cost:
            raise ValueError(
                f"ledger total {ledger.total} exceeds limits.max_cost {limits.max_cost}; "
                "size the ledger from the limit"
            )
        self.name = config.name
        self.version = config.version
        self.llm = config.llm
        self.ledger = ledger
        self.limits = limits
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
        """The default template: plan -> reserve -> execute -> commit, with up
        to ``limits.max_replans`` replans after an unsuccessful result.

        The plan's :meth:`~router.execution.plan.Plan.budget_estimate` is held
        on the ledger before executing and always committed or released
        (``try``/``finally``), and task / plan statuses are kept current. A plan
        that doesn't fit the remaining budget terminates the task with
        ``error_code="over_budget"``. Subclasses needing bespoke control flow
        override this directly rather than fighting the template."""
        if self.task is not None and self.task.status in (TaskStatus.PLANNING, TaskStatus.RUNNING):
            raise RuntimeError(
                f"orchestrator {self.name!r} is already running task {self.task.task_id!r}; "
                "use one orchestrator per task (spawn())"
            )
        self.task = task
        key = task.task_id
        task.status = TaskStatus.PLANNING
        try:
            plan = self.plan(task)
            replans = 0
            while True:
                plan.validate()
                self.ledger.release(key)       # drop any spawn-time allocation hold
                if not self.ledger.reserve(key, plan.budget_estimate()):
                    return self._finish(task, TaskStatus.TERMINATED,
                                        ExecutionResult(error_code="over_budget"))
                task.status, plan.status = TaskStatus.RUNNING, PlanStatus.EXECUTING
                committed = False
                try:
                    result = self.execute(plan)
                    self.ledger.commit(key, result.actual_cost)   # a failed attempt still spent
                    committed = True
                finally:
                    if not committed:
                        self.ledger.release(key)
                if result.success:
                    plan.status = PlanStatus.COMPLETED
                    return self._finish(task, TaskStatus.COMPLETED, result)
                plan.status = PlanStatus.FAILED
                if replans >= self.limits.max_replans:
                    return self._finish(task, TaskStatus.FAILED, result)
                replans += 1
                task.status = TaskStatus.PLANNING
                plan = self.replan(plan)
        except BaseException:
            task.status = TaskStatus.FAILED
            raise

    @staticmethod
    def _finish(task: "AgentTask", status: TaskStatus, result: ExecutionResult) -> ExecutionResult:
        task.status, task.result = status, result
        return result

    def can_spawn(self, child: "AgentTask") -> bool:
        """Side-effect-free admission check for ``child`` under ``self.task``:
        depth cap, fan-out cap, same root/ledger, depth == parent depth + 1,
        and enough remaining budget for ``child.allocated_budget`` (a positive
        remainder when no allocation is requested).

        ``max_depth`` is the depth of the deepest node allowed to *exist* --
        the same convention ``router.agentic.router.AgenticRouter._solve``
        uses (nodes exist through ``depth == max_depth``; recursion just stops
        there) -- so a child AT ``max_depth`` is still spawnable, only one
        deeper than that is rejected (XA-08)."""
        parent = self.task
        if child.depth > self.limits.max_depth:
            return False
        if child.root_task_id != self.ledger.root_task_id:
            return False
        if parent is not None:
            if child.depth != parent.depth + 1 or child.root_task_id != parent.root_task_id:
                return False
            if len(parent.children) >= self.limits.max_fanout:
                return False
        remaining = self.ledger.remaining()
        if child.allocated_budget > 0:
            return child.allocated_budget <= remaining
        return remaining > 0

    def spawn(self, child: "AgentTask") -> "Orchestrator":
        """Admit ``child``: reserve its ``allocated_budget``, link it under
        ``self.task``, and build a sibling orchestrator for it. Subclasses with
        extra constructor arguments extend :meth:`_child_kwargs`."""
        if not self.can_spawn(child):
            raise ValueError(
                f"cannot spawn child task {child.task_id!r} at depth {child.depth} "
                f"(max_depth={self.limits.max_depth}, max_fanout={self.limits.max_fanout}, "
                f"remaining budget={self.ledger.remaining()})"
            )
        if child.allocated_budget > 0 and not self.ledger.reserve(child.task_id, child.allocated_budget):
            raise ValueError(f"budget for child task {child.task_id!r} was taken concurrently")
        if self.task is not None:
            child.parent_task_id = self.task.task_id
            self.task.children.append(child)
        return type(self)(
            OrchestratorConfig(
                name=self.name, version=self.version, llm=self.llm,
                ledger=self.ledger, limits=self.limits, tools=self.tools,
            ),
            **self._child_kwargs(),
        )

    def _child_kwargs(self) -> dict:
        """Keyword arguments (beyond the config) a spawned child is built with."""
        return {"router_tool": self.router_tool, "plan_forecaster": self.plan_forecaster}


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
