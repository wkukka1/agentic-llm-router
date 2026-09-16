"""Conversation history -- lives here, not in ``router.context``, so this
package has zero dependency on ``router`` (``ClassificationInput`` needs it
too). ``router.context.RoutingRequest`` imports it from here instead."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class MessageRole(enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: MessageRole
    content: str
    tokens: int = 0


@dataclass
class Conversation:
    conversation_id: str
    messages: list[Message] = field(default_factory=list)
