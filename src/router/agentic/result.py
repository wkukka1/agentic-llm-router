"""Structured trace returned by :class:`~router.agentic.router.AgenticRouter`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = ["SubCall", "AgenticResult"]


@dataclass
class SubCall:
    """One model invocation the orchestrator made."""

    prompt: str
    model_id: str
    predicted_quality: float
    answer: str
    mode: str = "single"                # "single" | "orchestrated"
    depth: int = 0
    cost: Optional[float] = None
    children: list["SubCall"] = field(default_factory=list)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


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
    cost: float = 0.0

    def models_used(self) -> list[str]:
        seen: list[str] = []
        for s in self.steps:
            for node in s.walk():
                if node.model_id not in seen:
                    seen.append(node.model_id)
        if self.selected_model_id and self.selected_model_id not in seen:
            seen.insert(0, self.selected_model_id)
        return seen

    def summary(self) -> str:
        if self.mode == "single":
            return (f"single -> {self.selected_model_id} "
                    f"(q~{self.predicted_quality:.2f}): {self.triage_reason}")
        n = sum(1 for s in self.steps for _ in s.walk())
        return (f"orchestrated ({self.orchestrator}) -> {n} sub-call(s) across "
                f"{self.models_used()}: {self.triage_reason}")
