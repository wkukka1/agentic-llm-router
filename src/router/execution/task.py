"""``AgentTask``: the task tree an orchestrator's recursion builds, with an
explicit ``depth`` field bounded by :class:`~router.execution.budget.ExecutionLimits`.

The live orchestrator (:class:`router.agentic.orchestrator.RecursiveOrchestrator`)
recurses via plain function calls (``AgenticRouter._solve(prompt, depth, ...)``)
with depth as a bare int parameter, not a persisted task-tree node -- there is
no ``taskId``/``parentTaskId``/``children`` bookkeeping to inspect after the
fact. This is that bookkeeping, unwired.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

from .plan import Plan
from .results import ExecutionResult


class TaskStatus(enum.Enum):
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


_IN_PROGRESS = (TaskStatus.PENDING, TaskStatus.PLANNING, TaskStatus.RUNNING)
_FAILED = (TaskStatus.FAILED, TaskStatus.TERMINATED)


@dataclass
class AgentTask:
    task_id: str
    parent_task_id: Optional[str]
    root_task_id: str
    prompt: str
    depth: int = 0
    allocated_budget: float = 0.0
    status: TaskStatus = TaskStatus.PENDING
    plan: Optional[Plan] = None
    result: Optional[ExecutionResult] = None
    children: list["AgentTask"] = field(default_factory=list)

    def aggregate_children(self) -> ExecutionResult:
        """Roll every child's result into one.

        * Raises if a child is still in progress (no result yet, not failed) --
          aggregating a partial tree would under-report cost and success.
        * A child that failed, was terminated, or finished without a result
          makes the aggregate unsuccessful; the first child ``error_code`` is
          propagated (``"child_failed"`` if none was set).
        * No children -> unsuccessful with ``error_code="no_children"``.
        * Cost and tokens are summed; latency is the **max**, since siblings
          run as a fan-out, not in sequence.
        """
        pending = [c.task_id for c in self.children
                   if c.result is None and c.status in _IN_PROGRESS]
        if pending:
            raise RuntimeError(f"cannot aggregate {self.task_id!r}: children still running {pending}")
        if not self.children:
            return ExecutionResult(success=False, error_code="no_children")

        results = [c.result for c in self.children if c.result is not None]
        failed = [c for c in self.children
                  if c.result is None or c.status in _FAILED or not c.result.success]
        error_code = None
        if failed:
            error_code = next((c.result.error_code for c in failed
                               if c.result is not None and c.result.error_code), "child_failed")
        return ExecutionResult(
            output="\n".join(r.output for r in results),
            actual_cost=sum(r.actual_cost for r in results),
            actual_latency=max((r.actual_latency for r in results), default=0.0),
            input_tokens=sum(r.input_tokens for r in results),
            output_tokens=sum(r.output_tokens for r in results),
            success=not failed,
            error_code=error_code,
        )
