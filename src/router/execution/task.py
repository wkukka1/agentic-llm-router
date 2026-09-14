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
        """Roll every child's result into one. No cost/latency weighting by
        model or step type -- a straight sum/concat, good enough until a real
        aggregation policy is needed."""
        outputs = [c.result.output for c in self.children if c.result is not None]
        return ExecutionResult(
            output="\n".join(outputs),
            actual_cost=sum(c.result.actual_cost for c in self.children if c.result),
            actual_latency=sum(c.result.actual_latency for c in self.children if c.result),
            input_tokens=sum(c.result.input_tokens for c in self.children if c.result),
            output_tokens=sum(c.result.output_tokens for c in self.children if c.result),
            success=all(c.result.success for c in self.children if c.result),
        )
