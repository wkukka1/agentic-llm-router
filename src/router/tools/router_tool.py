"""``RouterTool``: intended to expose a :class:`~router.routing.base.Router`
as a :class:`Tool` so an orchestrator can call routing recursively without
ever re-entering full agentic dispatch -- design scaffolding, not a working
guard today (XA-07).

``route_for_model_selection`` unconditionally raises ``NotImplementedError``
below, and ``MODEL_SELECTION`` mode (:class:`router.context.RoutingMode`) is
never read anywhere outside this file and ``router.context`` -- nothing
enforces it. The live tool-originated recursion path,
:func:`router.agentic.orchestrator.build_router_tools`'s ``route_and_answer``,
calls ``agent._solve(query, depth + 1)`` directly: a full re-triage that CAN
build a fresh ``AgentExecutor`` and recurse into another orchestrator, up to
``AgenticRouter.max_depth``. The only guard that actually exists today is that
bare depth check inside :meth:`router.agentic.router.AgenticRouter._solve` --
there is no mode flag distinguishing a tool-initiated call from the root call.
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
        today, and :class:`~router.policy.DefaultRoutingPolicy` can turn scores
        into a :class:`~router.decision.RoutingDecision` -- but this method
        itself, and the ``MODEL_SELECTION`` mode/recursion-guard story this
        class's module docstring describes, remain unimplemented and unwired.
        ``router.agentic.AgenticRouter.route_decision(prompt)`` is the live
        equivalent, with only a bare depth counter as its guard, not this.

        The implementation must route a *copy*
        (``dataclasses.replace(request, mode=RoutingMode.MODEL_SELECTION)``),
        never mutate the caller's request -- a caller that falls back after an
        error would otherwise keep a request stuck in ``MODEL_SELECTION``.
        """
        raise NotImplementedError(
            "RouterTool.route_for_model_selection is unimplemented scaffolding; "
            "router.agentic.AgenticRouter.route_decision(prompt) "
            "is the working equivalent today."
        )
