"""Per-request routing constraints and the utility function scored against them.

Scaffolding for the production router design (``router-design.mermaid``) --
nothing in the existing NIRT-based routing/agentic code constructs these yet.
See ``docs/architecture.md`` for what's wired up today vs. what's scaffolding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from .decision import ModelScore


@dataclass
class OptimizationObjective:
    """Weights the four terms a routing utility trades off. ``utility()`` is
    the one place ``u(m, i, lambda)`` is defined -- offline signal analysis
    (``evaluation``) must import this rather than reimplementing it, or
    offline results silently stop matching what the router actually optimizes."""

    quality_weight: float = 1.0
    cost_weight: float = 0.0
    latency_weight: float = 0.0
    risk_weight: float = 0.0

    def utility(self, score: "ModelScore") -> float:
        risk = 1.0 - score.confidence
        return (
            self.quality_weight * score.expected_quality
            - self.cost_weight * score.expected_cost
            - self.latency_weight * score.expected_latency
            - self.risk_weight * risk
        )


@dataclass
class RoutingConstraints:
    """Hard and soft limits a candidate must clear before it's scored.

    ``required_capabilities`` is checked before scoring
    (``Router.filter_candidates``). ``max_cost`` / ``max_latency`` /
    ``min_quality`` are hard filters applied to the *scored* candidates by
    ``RoutingPipeline.route`` (``min_quality`` needs the prediction), compared
    against ``ModelScore.expected_cost`` / ``expected_latency`` /
    ``expected_quality``. ``objective`` is the soft utility tradeoff applied to
    whatever candidates survive."""

    max_cost: float | None = None
    max_latency: float | None = None
    min_quality: float | None = None
    required_capabilities: list[str] = field(default_factory=list)
    objective: OptimizationObjective = field(default_factory=OptimizationObjective)
