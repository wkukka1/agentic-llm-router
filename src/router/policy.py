"""``RoutingPolicy``: turns a batch of :class:`~router.decision.ModelScore`\\ s
into a :class:`~router.decision.RoutingDecision` and a destination.

Sits one layer above :class:`router.routing.base.Router`: a ``Router``
answers "how good is model m for query q"; a ``RoutingPolicy`` answers "given
those scores, what do we actually do" -- pick a model and return to the user,
hand back to the parent orchestrator, or spin up a new one. :class:`DefaultRoutingPolicy`
below is a concrete implementation, constructed by default in
:class:`~router.router.RoutingPipeline` and live on its ``route`` path --
:meth:`decide` and :meth:`select_destination` are the two hooks it fills in.
:class:`MarginRoutingPolicy` is the cost-aware alternative: same ranking,
cheapest candidate within a margin of the best.
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


class MarginRoutingPolicy(RoutingPolicy):
    """Take the cheapest candidate whose utility is within ``margin`` of the best.

    The measured problem with :class:`DefaultRoutingPolicy` is that argmax on a
    ranker whose scores are dominated by global model capability picks the
    strongest model almost every time, which scores about the same as having no
    router at all while costing the same. The ranking is not the issue: the
    model that should have been picked is usually rank 2. The decision rule is.

    So this rule keeps the ranking and changes only the selection: among the
    candidates the ranker rates within ``margin`` of its best, take the cheapest.
    A *relative* margin, not an absolute score threshold -- a predicted 0.7
    means different things on an easy and a hard query, and an absolute bar
    measured worse than no router at all, while the relative one measured
    better (see docs/prompt_decomposition_experiments.md).

    What a given margin buys is pool-dependent, so the numbers live in
    docs/prompt_decomposition_experiments.md rather than in copies here; use
    ``evaluation.prompt_decomposition.policy_eval.margin_for_target_saving`` to
    turn a savings target into a margin for a specific pool.

    ``escalation_threshold`` is honoured here rather than ignored: when even
    the best candidate is rated below it, the request goes to the parent
    orchestrator instead of being answered by a model nobody believes in.
    """

    name = "margin"
    version = "1"

    def __init__(self, margin: float = 0.02, escalation_threshold: float = 0.0) -> None:
        if margin < 0:
            raise ValueError(f"margin must be >= 0, got {margin}")
        self.margin = float(margin)
        self.escalation_threshold = float(escalation_threshold)

    def _ranked(self, context: "RoutingContext",
                scores: list["ModelScore"]) -> list["ModelScore"]:
        objective = context.request.constraints.objective

        def key(score: "ModelScore") -> float:
            u = float(objective.utility(score))
            return u if math.isfinite(u) else -math.inf

        return sorted(scores, key=key, reverse=True)

    def _margin_for(self, context: "RoutingContext") -> float:
        """The request's quality budget, or this policy's default.

        Cost sensitivity is a property of the *request*, not of the deployment:
        the same service answers "summarise this changelog" and "write the
        migration", and they do not deserve the same budget. `None` on the
        constraints means the caller did not say, so the policy's own value
        stands.
        """
        asked = getattr(context.request.constraints, "quality_margin", None)
        if asked is None:
            return self.margin
        asked = float(asked)
        if asked < 0:
            raise ValueError(f"quality_margin must be >= 0, got {asked}")
        return asked

    def decide(self, context: "RoutingContext", scores: list["ModelScore"]) -> RoutingDecision:
        objective = context.request.constraints.objective
        margin = self._margin_for(context)
        ranked = self._ranked(context, scores)
        chosen = None
        if ranked:
            best = float(objective.utility(ranked[0]))
            within = [s for s in ranked if float(objective.utility(s)) >= best - margin]
            # ties on cost keep the better-ranked model: `within` is already in
            # rank order and `min` is stable.
            chosen = min(within, key=lambda s: s.expected_cost)
        return RoutingDecision(
            request_id=context.request.request_id,
            ranked_models=ranked,
            selected_model_id=chosen.model.model_id if chosen else None,
            destination=self.select_destination(context, scores),
            reason=(f"cheapest within {margin:g} utility of the best"
                    if chosen else "no candidates scored"),
            confidence=chosen.confidence if chosen else 0.0,
            policy_version=self.version,
        )

    def select_destination(self, context: "RoutingContext",
                           scores: list["ModelScore"]) -> DecisionDestination:
        ranked = self._ranked(context, scores)
        if not ranked:
            return DecisionDestination.USER
        objective = context.request.constraints.objective
        if float(objective.utility(ranked[0])) < self.escalation_threshold:
            return DecisionDestination.PARENT_ORCHESTRATOR
        return DecisionDestination.USER
