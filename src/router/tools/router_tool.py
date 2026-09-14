"""``RouterTool``: exposes a :class:`~router.routing.base.Router` as a
:class:`Tool`, so an orchestrator can call routing recursively without ever
re-entering full agentic dispatch.

``MODEL_SELECTION`` mode (:class:`router.context.RoutingMode`) is what
prevents unbounded re-entry: :meth:`route_for_model_selection` always sets it,
and (once :class:`~router.router.RoutingPipeline` exists) the pipeline must
refuse to hand a ``MODEL_SELECTION`` decision back out to a *new*
orchestrator -- it can only return a plain model choice. Today's working
equivalent is simpler and already enforces this by construction:
:func:`router.agentic.orchestrator.build_router_tools`'s ``route_query`` tool
calls ``agent.route_decision(prompt)`` directly, which only ever returns a
:class:`~router.routing.base.RoutingResult`, never something that could
itself trigger another orchestrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..context import RoutingMode, RoutingRequest
from .base import Tool

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from ..decision import RoutingDecision
    from ..routing.base import Router


class RouterTool(Tool):
    name = "router"
    version = "0"

    def __init__(self, router: "Router"):
        self.router = router

    def execute(self, input: RoutingRequest) -> "RoutingDecision":
        return self.route_for_model_selection(input)

    def route_for_model_selection(self, request: RoutingRequest) -> "RoutingDecision":
        """Route ``request`` for model selection only: a plain pick, never a
        hand-off. Real ranking (:meth:`Router.route`/``route_text``) works
        today; building the richer :class:`~router.decision.RoutingDecision`
        record from a :class:`~router.routing.base.RoutingResult` is the piece
        this stubs -- it needs :class:`~router.policy.RoutingPolicy`, which
        nothing implements yet.
        """
        request.mode = RoutingMode.MODEL_SELECTION
        raise NotImplementedError(
            "RouterTool needs a RoutingPolicy to turn a RoutingResult into a "
            "RoutingDecision; router.agentic.AgenticRouter.route_decision(prompt) "
            "is the working equivalent today."
        )
