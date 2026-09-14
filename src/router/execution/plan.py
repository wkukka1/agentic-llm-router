"""A ``Plan`` is a sequence of ``PlanStep``\\ s an orchestrator commits to
before executing -- not built by anything yet. The live
:class:`router.agentic.orchestrator.RecursiveOrchestrator` decomposes and
dispatches directly, with no intermediate plan object to forecast or approve
first; that's the gap this module (and :mod:`router.execution.forecast`)
scaffolds."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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


@dataclass
class Plan:
    plan_id: str
    task_id: str
    steps: list[PlanStep] = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_quality: float = 0.0
    estimated_latency: float = 0.0
    utility: float = 0.0
    status: PlanStatus = PlanStatus.DRAFT
    revision: int = 0
