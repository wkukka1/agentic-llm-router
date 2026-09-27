"""Structured trace returned by :class:`~router.agentic.router.AgenticRouter`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

__all__ = ["SubCall", "AgenticResult", "total_cost"]


@dataclass
class SubCall:
    """One model invocation the orchestrator made (a leaf), or an orchestrated
    branch whose ``children`` are the sub-calls. ``error`` is set when the call
    failed; ``answer`` is then a placeholder."""

    prompt: str
    model_id: str
    predicted_quality: float
    answer: str
    mode: str = "single"                # "single" | "orchestrated"
    depth: int = 0
    cost: Optional[float] = None
    children: list["SubCall"] = field(default_factory=list)
    error: Optional[str] = None

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def leaves(self):
        """Nodes that made an actual model call. A branch's ``cost`` already
        sums its descendants, so totals must add leaves only."""
        return (n for n in self.walk() if not n.children)


def total_cost(nodes: Iterable[SubCall]) -> Optional[float]:
    """Sum ``nodes``' costs (pass ``.leaves()`` -- a branch's own ``cost``
    already includes its descendants, so summing branches double-counts).

    A failed node (``error`` set) contributes ``0`` -- it didn't complete, so
    nothing was billed. A *successful* node with ``cost is None`` (a real
    call whose client couldn't price it, e.g. no profile configured) makes
    the total ``None`` -- unresolved, not free (AGENTICROUTER-002: `None`
    must not collapse to `0.0`, or an unpriced run silently reports as
    free)."""
    total = 0.0
    for n in nodes:
        if n.cost is not None:
            total += n.cost
        elif n.error is None:
            return None
    return total


@dataclass
class AgenticResult:
    prompt: str
    mode: str                            # "single" | "orchestrated"
    answer: str
    triage_reason: str
    selected_model_id: Optional[str] = None
    predicted_quality: Optional[float] = None
    steps: list[SubCall] = field(default_factory=list)
    orchestrator: Optional[str] = None
    cost: Optional[float] = None

    def models_used(self) -> list[str]:
        """Models actually invoked. In orchestrated mode ``selected_model_id``
        only produced the triage signal, so it is listed only if some sub-call
        used it."""
        seen: list[str] = []
        for s in self.steps:
            for node in s.leaves():
                if node.model_id not in seen:
                    seen.append(node.model_id)
        if self.mode == "single" and self.selected_model_id and self.selected_model_id not in seen:
            seen.insert(0, self.selected_model_id)
        return seen

    def errors(self) -> list[SubCall]:
        """Sub-calls that failed (their answers are placeholders)."""
        return [n for s in self.steps for n in s.walk() if n.error is not None]

    def summary(self) -> str:
        if self.mode == "single":
            q = "?" if self.predicted_quality is None else f"{self.predicted_quality:.2f}"
            return f"single -> {self.selected_model_id} (q~{q}): {self.triage_reason}"
        n = sum(1 for s in self.steps for _ in s.leaves())
        return (f"orchestrated ({self.orchestrator}) -> {n} sub-call(s) across "
                f"{self.models_used()}: {self.triage_reason}")
