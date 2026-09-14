"""Depth/fanout limits and a shared money ledger for one root task's whole
orchestration tree.

Nothing currently enforces a budget across recursive orchestration --
``router.agentic.AgenticRouter.max_depth`` is the one limit that exists today
(a plain recursion-depth cap, no cost tracking). ``BudgetLedger`` is real,
working bookkeeping (reserve/commit/release with no double-spend); it's just
not threaded through :mod:`router.agentic` or :mod:`router.execution.orchestrator`
yet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutionLimits:
    max_cost: float = float("inf")
    max_latency: float = float("inf")
    max_depth: int = 2
    max_fanout: int = 8
    max_replans: int = 1


class BudgetLedger:
    """Tracks ``spent + reserved <= total`` for one root task's tree.

    ``reserve`` is a hold (e.g. a plan's estimated cost) that must be
    ``commit``\\ ted (replaced with the actual amount) or ``release``\\ d
    (freed, on failure/skip) -- never left dangling, or ``remaining()``
    permanently under-reports what's actually spendable.
    """

    def __init__(self, root_task_id: str, total: float):
        self.root_task_id = root_task_id
        self.total = float(total)
        self.spent = 0.0
        self._reservations: dict[str, float] = {}

    @property
    def reserved(self) -> float:
        return sum(self._reservations.values())

    def remaining(self) -> float:
        return self.total - self.spent - self.reserved

    def reserve(self, task_id: str, amount: float) -> bool:
        amount = float(amount)
        if amount > self.remaining():
            return False
        self._reservations[task_id] = self._reservations.get(task_id, 0.0) + amount
        return True

    def commit(self, task_id: str, actual: float) -> None:
        self._reservations.pop(task_id, None)
        self.spent += float(actual)

    def release(self, task_id: str) -> None:
        self._reservations.pop(task_id, None)
