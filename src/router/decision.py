"""The output of a ranking pass (:class:`ModelScore`) and of a full routing
decision (:class:`RoutingDecision`) -- the production-shaped counterparts to
:class:`router.routing.base.RoutingResult`, which is what's actually returned
by ``Router.route()`` today. ``RoutingResult`` carries a whole batch's scores
as one dense matrix (efficient for offline evaluation over many queries);
these carry one request's decision as a richer, per-field record (a
``reason``, ``confidence``, ``destination``). ``RoutingDecision`` is live: it
is produced by :class:`~router.policy.DefaultRoutingPolicy` and returned from
:class:`~router.router.RoutingPipeline`'s ``route`` method."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from .llm.profile import LLMProfile


@dataclass
class ModelScore:
    model: "LLMProfile"
    score: float
    expected_quality: float = 0.0
    expected_cost: float = 0.0
    expected_latency: float = 0.0
    confidence: float = 1.0
    supported: bool = True


class DecisionDestination(enum.Enum):
    USER = "user"
    PARENT_ORCHESTRATOR = "parent_orchestrator"
    NEW_ORCHESTRATOR = "new_orchestrator"


@dataclass
class RoutingDecision:
    request_id: str
    ranked_models: list[ModelScore] = field(default_factory=list)
    destination: DecisionDestination = DecisionDestination.USER
    reason: str = ""
    confidence: float = 1.0
    artifact_id: Optional[str] = None
    policy_version: str = ""
