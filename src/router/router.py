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

from typing import TYPE_CHECKING, Optional

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
        """
        context = self.build_context(request)
        candidates = context.candidates or [
            _StubProfile(m) for m in self.router_model.model_ids
        ]
        filtered = self.router_model.filter_candidates(candidates, request.constraints)
        filtered_ids = {c.model_id for c in filtered}

        result = self.router_model.route_text([request.prompt])
        row = result.score_frame().iloc[0]
        by_id = {c.model_id: c for c in filtered}
        scores = [
            ModelScore(
                model=by_id[model_id],
                score=float(value),
                expected_quality=float(value),
                expected_cost=getattr(by_id[model_id], "output_cost_per_token", 0.0),
                confidence=1.0,
                supported=True,
            )
            for model_id, value in row.items()
            if model_id in filtered_ids
        ]
        return self.routing_policy.decide(context, scores)


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
