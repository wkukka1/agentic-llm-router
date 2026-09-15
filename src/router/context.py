"""The request/context split: :class:`RoutingRequest` is raw caller input,
:class:`RoutingContext` is what :class:`~router.router.RoutingPipeline`
builds from it (decomposed signals + filtered candidates) before ranking.

``Conversation``/``Message``/``MessageRole`` and ``PromptSignals`` live in
the standalone top-level :mod:`decompose` package, not here -- re-exported
below for convenience since ``RoutingRequest``/``RoutingContext`` reference
them directly."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from decompose.conversation import Conversation, Message, MessageRole
from decompose.signals import PromptSignals

from .constraints import RoutingConstraints

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from .llm.profile import LLMProfile

__all__ = [
    "Conversation", "Message", "MessageRole", "PromptSignals",
    "RoutingMode", "RoutingRequest", "RoutingContext",
]


class RoutingMode(enum.Enum):
    """``MODEL_SELECTION`` is *intended* to be what a
    :class:`~router.tools.router_tool.RouterTool` passes when an orchestrator
    calls the router as a tool, to prevent the router from ever recursing back
    into full agentic dispatch (XA-07) -- but nothing sets or reads this value
    today: :meth:`RouterTool.route_for_model_selection` is unimplemented, and
    the live tool-originated recursion path
    (``router.agentic.orchestrator.build_router_tools``'s ``route_and_answer``)
    does call back into full dispatch, with only a bare depth counter as its
    guard. This mode is design scaffolding, not an enforced guard."""

    NORMAL = "normal"
    MODEL_SELECTION = "model_selection"


@dataclass
class RoutingRequest:
    request_id: str
    prompt: str
    conversation: Optional[Conversation] = None
    constraints: RoutingConstraints = field(default_factory=RoutingConstraints)
    mode: RoutingMode = RoutingMode.NORMAL
    parent_task_id: Optional[str] = None
    depth: int = 0


@dataclass
class RoutingContext:
    request: RoutingRequest
    signals: PromptSignals
    candidates: list["LLMProfile"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
