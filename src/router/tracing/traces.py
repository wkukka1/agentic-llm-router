"""``ExecutionTrace``: one served request, resolved back to the exact router
version, artifact, and policy that decided it, plus what actually happened.

Nothing emits one of these today. The closest existing thing is a training
run's ``run.json`` (git sha, config, metrics -- see
``training.nirt.train.fit``), which records a *fitting* run, not a *serving*
decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from ..context import RoutingRequest
    from ..decision import RoutingDecision
    from decompose.signals import PromptSignals
    from ..execution.results import ExecutionResult


@dataclass
class ExecutionTrace:
    trace_id: str
    request_id: str
    task_id: str
    router_version: str
    artifact_id: str
    policy_version: str
    request: "RoutingRequest"
    signals: "PromptSignals"
    decision: "RoutingDecision"
    selected_model_id: str
    result: "ExecutionResult"
    timestamp: str
