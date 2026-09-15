"""``RoutingPolicy``: turns a batch of :class:`~router.decision.ModelScore`\\ s
into a :class:`~router.decision.RoutingDecision` and a destination.

Sits one layer above :class:`router.routing.base.Router`: a ``Router``
answers "how good is model m for query q"; a ``RoutingPolicy`` answers "given
those scores, what do we actually do" -- pick a model and return to the user,
hand back to the parent orchestrator, or spin up a new one. Nothing
constructs a concrete ``RoutingPolicy`` yet; :meth:`decide` and
:meth:`select_destination` are the two hooks a real implementation fills in.
"""

from __future__ import annotations

import abc
import math
from typing import TYPE_CHECKING

from .decision import DecisionDestination, RoutingDecision

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from .context import RoutingContext
    from .decision import ModelScore


class RoutingPolicy(abc.ABC):
    name: str = "policy"
    version: str = "0"
    escalation_threshold: float = 0.0

    @abc.abstractmethod
    def decide(self, context: "RoutingContext", scores: list["ModelScore"]) -> RoutingDecision:
        raise NotImplementedError

    @abc.abstractmethod
    def select_destination(self, context: "RoutingContext",
                           scores: list["ModelScore"]) -> DecisionDestination:
        raise NotImplementedError


class DefaultRoutingPolicy(RoutingPolicy):
    """Argmax on ``context.request.constraints.objective.utility(score)``,
    always destined for the user -- no escalation, no policy versioning
    beyond the literal string below. The one concrete ``RoutingPolicy`` that
    exists, so :class:`~router.router.RoutingPipeline` has something real to
    compose with; a production policy would at minimum use
    :attr:`escalation_threshold` to pick
    :class:`~router.decision.DecisionDestination.PARENT_ORCHESTRATOR` when
    every candidate scores too low, which this does not.

    Reads the objective from ``context`` (per-request) rather than carrying
    its own fixed one -- ``RoutingConstraints.objective`` exists precisely so
    the cost/quality/latency tradeoff can vary per request; a policy-level
    default would silently ignore it.
    """

    name = "default"
    version = "0"

    def decide(self, context: "RoutingContext", scores: list["ModelScore"]) -> RoutingDecision:
        objective = context.request.constraints.objective

        def key(score: "ModelScore") -> float:
            u = float(objective.utility(score))
            return u if math.isfinite(u) else -math.inf

        ranked = sorted(scores, key=key, reverse=True)
        top = ranked[0] if ranked else None
        return RoutingDecision(
            request_id=context.request.request_id,
            ranked_models=ranked,
            destination=self.select_destination(context, scores),
            reason="highest utility" if top else "no candidates scored",
            confidence=top.confidence if top else 0.0,
            policy_version=self.version,
        )

    def select_destination(self, context: "RoutingContext",
                           scores: list["ModelScore"]) -> DecisionDestination:
        return DecisionDestination.USER
