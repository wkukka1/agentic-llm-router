"""``Router``: decompose -> filter -> rank -> decide, the top-level entry
point in the production router design (``docs/architecture.md``).

This is the diagram's top-level ``Router`` -- the class name was previously
free only after :class:`router.routing.base.Router` (the per-``(query,
model)`` scorer) was renamed to :class:`router.routing.base.RouterModel` to
make room for it. Still missing the diagram's ``agentFactory`` wiring
(``orchestrator_factory`` below is typed but nothing constructs an
``Orchestrator`` from it yet -- that's :mod:`router.execution`, out of scope
for this pass).

Composes real, working pieces (:class:`~decompose.decomposer.PromptDecomposer`,
:class:`~router.routing.base.RouterModel`, :class:`~router.llm.registry.LLMRegistry`,
:class:`~router.policy.RoutingPolicy`) end to end. What it does *not* do is
dispatch the call or hand off to an orchestrator -- that's
:mod:`router.agentic` (already wired, simpler) or
:mod:`router.execution.orchestrator` (scaffolding, not wired). This class
only produces a :class:`~router.decision.RoutingDecision`.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np

from decompose.classifiers.base import ClassificationInput
from decompose.decomposer import PromptDecomposer

from .context import RoutingContext, RoutingRequest
from .decision import ModelScore, RoutingDecision
from .policy import DefaultRoutingPolicy, RoutingPolicy

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from .execution.orchestrator import OrchestratorFactory
    from .llm.registry import LLMRegistry
    from .routing.base import RouterModel


class Router:
    def __init__(
        self,
        router_model: "RouterModel",
        *,
        name: str = "pipeline",
        version: str = "0",
        prompt_decomposer: Optional[PromptDecomposer] = None,
        routing_policy: Optional[RoutingPolicy] = None,
        llm_registry: Optional["LLMRegistry"] = None,
        orchestrator_factory: Optional["OrchestratorFactory"] = None,
    ):
        self.name = name
        self.version = version
        self.router_model = router_model
        self.prompt_decomposer = prompt_decomposer or PromptDecomposer()
        self.routing_policy = routing_policy or DefaultRoutingPolicy()
        self.llm_registry = llm_registry
        self.orchestrator_factory = orchestrator_factory

    def build_context(self, request: RoutingRequest) -> RoutingContext:
        signals = self.prompt_decomposer.extract_signals(
            ClassificationInput(prompt=request.prompt, conversation=request.conversation)
        )
        candidates = self.llm_registry.list() if self.llm_registry is not None else []
        return RoutingContext(request=request, signals=signals, candidates=candidates)

    def route(self, request: RoutingRequest) -> RoutingDecision:
        """Decompose -> filter -> rank -> decide.

        Ranking needs a text-capable :attr:`router_model`
        (:attr:`~router.routing.base.RouterModel.can_route_text`, e.g.
        ``NIRTRouter``/``KNNRouter``) since a live prompt has no prebuilt
        ``query_id``. This is real end-to-end glue, not a stub -- it's the
        one piece of the production design that composes entirely out of
        already-working code (``PromptDecomposer``, ``RouterModel.route_text``,
        ``DefaultRoutingPolicy``).

        Only the raw predicted-quality scores come from ``route_text`` --
        cost/latency tradeoffs are applied once, by ``routing_policy`` via
        ``request.constraints.objective``, not also inside ``RouterModel``'s own
        ``lam``-weighted selection (which this deliberately leaves at 0 to
        avoid scoring cost twice).

        ``expected_cost`` is the router's own per-model cost vector
        (:attr:`~router.routing.base.RouterModel.default_model_costs`, train-mean
        USD/query) when it has one -- the same scale ``RouterModel.route(lam=...)``
        trades against. A kept candidate outside the pool is priced at the pool
        maximum on that same scale; the profile's ``output_cost_per_token`` is
        used only when the router carries no cost signal at all.
        ``RoutingDecision.artifact_id`` is the router model's artifact id, if it
        was built from one.

        Non-finite scores are dropped (a model with no prediction must never
        win). Candidates kept by ``UnsupportedCandidatePolicy.SCORE_WITH_PRIOR``
        are scored with the pool-mean prediction, ``confidence=0`` and
        ``supported=False``. The hard limits in ``request.constraints``
        (``min_quality`` / ``max_cost`` / ``max_latency``) are applied to the
        scored candidates before the policy ranks them.

        **Latency is not wired up yet.** No signal source populates
        ``ModelScore.expected_latency`` anywhere in this package (unlike
        ``expected_cost``, which has a router/profile fallback), so it's
        permanently ``0.0`` -- a non-default ``max_latency`` filters nothing,
        and a non-default ``objective.latency_weight`` has zero effect on
        ranking. ``route()`` warns rather than silently no-opping when either
        is set to something other than its default (RC-01).
        """
        from .routing.base import UnsupportedCandidatePolicy

        constraints = request.constraints
        if constraints.max_latency is not None:
            warnings.warn(
                "RoutingConstraints.max_latency is set, but no latency signal "
                "exists anywhere in router -- ModelScore.expected_latency is "
                "always 0.0, so this hard limit filters nothing (RC-01)",
                stacklevel=2,
            )
        if constraints.objective.latency_weight != 0.0:
            warnings.warn(
                "OptimizationObjective.latency_weight is set, but no latency "
                "signal exists anywhere in router -- ModelScore.expected_latency "
                "is always 0.0, so this weight has no effect on ranking (RC-01)",
                stacklevel=2,
            )

        context = self.build_context(request)
        if self.llm_registry is None:
            # no registry at all -> the router's bare ids are the candidates. An
            # *empty* registry is a real "no candidates" and must not fall back.
            context.candidates = [_StubProfile(m) for m in self.router_model.model_ids]
        filtered = self.router_model.filter_candidates(context.candidates, request.constraints)

        result = self.router_model.route_text([request.prompt])
        row = result.score_frame().iloc[0]
        finite = row[np.isfinite(row.to_numpy(np.float64))]
        prior = float(finite.mean()) if len(finite) else float("nan")
        pool_costs = self.router_model.default_model_costs
        cost_of = (dict(zip(self.router_model.model_ids, map(float, pool_costs)))
                   if pool_costs is not None else {})
        # a candidate outside the pool has no entry in ``cost_of``; price it at the
        # pool max (pessimistic, same choice as ``_mean_train_costs``) so it stays on
        # the router's USD/query scale -- the profile's per-token price is ~1000x smaller
        off_pool_cost = max(cost_of.values()) if cost_of else None
        with_prior = (self.router_model.unsupported_policy
                      is UnsupportedCandidatePolicy.SCORE_WITH_PRIOR)

        scores = []
        for cand in filtered:
            model_id = cand.model_id
            if model_id in row.index:
                value, confidence, supported = float(row[model_id]), 1.0, True
            elif with_prior:
                value, confidence, supported = prior, 0.0, False
            else:
                continue
            if not math.isfinite(value):
                continue
            if model_id in cost_of:
                expected_cost = cost_of[model_id]
            elif off_pool_cost is not None:
                expected_cost = off_pool_cost
            else:
                expected_cost = getattr(cand, "output_cost_per_token", 0.0)
            scores.append(ModelScore(
                model=cand,
                score=value,
                expected_quality=value,
                expected_cost=expected_cost,
                confidence=confidence,
                supported=supported,
            ))
        scores = _apply_hard_limits(scores, request.constraints)
        decision = self.routing_policy.decide(context, scores)
        if decision.artifact_id is None:
            artifact = getattr(self.router_model, "artifact", None)
            decision.artifact_id = getattr(artifact, "artifact_id", None)
        return decision


def _apply_hard_limits(scores: list[ModelScore], constraints) -> list[ModelScore]:
    """Drop scored candidates that violate ``min_quality`` / ``max_cost`` /
    ``max_latency``. Applied after scoring because ``min_quality`` needs the
    prediction; an unset (``None``) limit filters nothing."""
    min_q = getattr(constraints, "min_quality", None)
    max_c = getattr(constraints, "max_cost", None)
    max_l = getattr(constraints, "max_latency", None)
    return [
        s for s in scores
        if (min_q is None or s.expected_quality >= min_q)
        and (max_c is None or s.expected_cost <= max_c)
        and (max_l is None or s.expected_latency <= max_l)
    ]


class _StubProfile:
    """Wraps a bare ``model_id`` string as a minimal candidate when no
    :class:`~router.llm.registry.LLMRegistry` was given -- lets
    :meth:`Router.route` run against any existing ``RouterModel`` (which
    only knows string ids) without requiring the newer ``LLMProfile`` layer
    to be populated first."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.capabilities: list[str] = []
        self.output_cost_per_token = 0.0
