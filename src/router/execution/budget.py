"""Depth/fanout limits and a shared money ledger for one root task's whole
orchestration tree.

Nothing currently enforces a budget across recursive orchestration --
``router.agentic.AgenticRouter.max_depth`` / ``max_calls`` are the limits that
exist today (no money tracking). ``BudgetLedger`` is real, working bookkeeping
(reserve/commit/release with no double-spend); it's just not threaded through
:mod:`router.agentic` yet.
"""

from __future__ import annotations

import math
import threading
import warnings
from dataclasses import dataclass


@dataclass
class ExecutionLimits:
    """Caps for one orchestration tree. ``max_cost`` is the authority on spend:
    an :class:`~router.execution.orchestrator.Orchestrator` refuses a ledger
    whose ``total`` exceeds it. ``max_depth`` is the depth of the deepest node
    allowed to exist -- the same convention
    :class:`router.agentic.router.AgenticRouter` uses for its own (unrelated,
    unwired-together) ``max_depth`` (XA-08); see
    :meth:`~router.execution.orchestrator.Orchestrator.can_spawn`."""

    max_cost: float = float("inf")
    max_latency: float = float("inf")
    max_depth: int = 2
    max_fanout: int = 8
    max_replans: int = 1


def _amount(value: float, what: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0:
        # a NaN would poison remaining() (every later comparison is False, so the
        # budget stops binding); a negative one would mint budget
        raise ValueError(f"{what} must be a finite, non-negative amount, got {value!r}")
    return v


class BudgetLedger:
    """Tracks ``spent + reserved <= total`` for one root task's tree.

    ``reserve`` is a hold (e.g. a plan's estimated cost) that must be
    ``commit``\\ ted (replaced with the actual amount) or ``release``\\ d
    (freed, on failure/skip) -- never left dangling, or ``remaining()``
    permanently under-reports what's actually spendable. All operations are
    guarded by a lock, so sibling tasks may share one ledger across threads.
    """

    def __init__(self, root_task_id: str, total: float):
        self.root_task_id = root_task_id
        self.total = _amount(total, "total") if not math.isinf(float(total)) else float(total)
        self.spent = 0.0
        self._reservations: dict[str, float] = {}
        self._lock = threading.RLock()

    @property
    def reserved(self) -> float:
        with self._lock:
            return sum(self._reservations.values())

    def remaining(self) -> float:
        with self._lock:
            return self.total - self.spent - self.reserved

    def reserve(self, task_id: str, amount: float) -> bool:
        amount = _amount(amount, "reserve amount")
        with self._lock:          # check-then-act must be atomic
            if amount > self.remaining():
                return False
            self._reservations[task_id] = self._reservations.get(task_id, 0.0) + amount
            return True

    def commit(self, task_id: str, actual: float) -> None:
        actual = _amount(actual, "committed cost")
        with self._lock:
            if task_id not in self._reservations:
                warnings.warn(
                    f"BudgetLedger.commit: no reservation for task {task_id!r} "
                    "(mistyped id, or spend that was never held)",
                    stacklevel=2,
                )
            self._reservations.pop(task_id, None)
            self.spent += actual

    def release(self, task_id: str) -> None:
        with self._lock:
            self._reservations.pop(task_id, None)
