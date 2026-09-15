"""``RoutingPipeline``: decompose -> filter -> rank -> decide, the top-level
entry point in the production router design.

Named ``RoutingPipeline`` here, not ``Router`` -- the diagram's top-level
pipeline class is called ``Router``, but this codebase already has an
actively-used, tested ``Router`` (:class:`router.routing.base.Router`, the
per-``(query, model)`` scorer). Reusing the name for a different class in the
same package would be a namespace collision in practice even where Python
technically allows it (``router.router.Router`` vs
``router.routing.base.Router``), so this is deliberately renamed rather than
shadowing it -- see ``docs/architecture.md``.

Composes real, working pieces (:class:`~decompose.decomposer.PromptDecomposer`,
:class:`~router.routing.base.Router`, :class:`~router.llm.registry.LLMRegistry`,
:class:`~router.policy.RoutingPolicy`) end to end. What it does *not* do is
dispatch the call or hand off to an orchestrator -- that's
:mod:`router.agentic` (already wired, simpler) or
:mod:`router.execution.orchestrator` (scaffolding, not wired). This class
only produces a :class:`~router.decision.RoutingDecision`.
"""

from __future__ import annotations

import math
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
    from .routing.base import Router


class RoutingPipeline:
    def __init__(
        self,
        router_model: "Router",
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
        (:attr:`~router.routing.base.Router.can_route_text`, e.g.
        ``NIRTRouter``/``KNNRouter``) since a live prompt has no prebuilt
        ``query_id``. This is real end-to-end glue, not a stub -- it's the
        one piece of the production design that composes entirely out of
        already-working code (``PromptDecomposer``, ``Router.route_text``,
        ``DefaultRoutingPolicy``).

        Only the raw predicted-quality scores come from ``route_text`` --
        cost/latency tradeoffs are applied once, by ``routing_policy`` via
        ``request.constraints.objective``, not also inside ``Router``'s own
        ``lam``-weighted selection (which this deliberately leaves at 0 to
        avoid scoring cost twice).

        ``expected_cost`` is the router's own per-model cost vector
        (:attr:`~router.routing.base.Router.default_model_costs`, train-mean
        USD/query) when it has one -- the same scale ``Router.route(lam=...)``
        trades against -- and falls back to the profile's
        ``output_cost_per_token`` only when the router carries no cost signal.

        Non-finite scores are dropped (a model with no prediction must never
        win). Candidates kept by ``UnsupportedCandidatePolicy.SCORE_WITH_PRIOR``
        are scored with the pool-mean prediction, ``confidence=0`` and
        ``supported=False``. The hard limits in ``request.constraints``
        (``min_quality`` / ``max_cost`` / ``max_latency``) are applied to the
        scored candidates before the policy ranks them.
        """
        from .routing.base import UnsupportedCandidatePolicy

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
            scores.append(ModelScore(
                model=cand,
                score=value,
                expected_quality=value,
                expected_cost=cost_of.get(model_id, getattr(cand, "output_cost_per_token", 0.0)),
                confidence=confidence,
                supported=supported,
            ))
        scores = _apply_hard_limits(scores, request.constraints)
        return self.routing_policy.decide(context, scores)


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
    :meth:`RoutingPipeline.route` run against any existing ``Router`` (which
    only knows string ids) without requiring the newer ``LLMProfile`` layer
    to be populated first."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.capabilities: list[str] = []
        self.output_cost_per_token = 0.0
