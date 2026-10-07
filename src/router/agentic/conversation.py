"""Multi-turn input for :class:`~router.agentic.router.AgenticRouter`.

``AgenticRouter.run`` accepts a bare prompt, a chat message list
(``[{"role": ..., "content": ...}]``) or a
:class:`decompose.conversation.Conversation`. These helpers split such an input
into the three things the router needs:

* **routing text** -- what the router *scores*: the user turns joined with
  blank lines. That is exactly how the training data rendered a multi-turn
  conversation (``training.data.normalize.render_prompt`` over Arena's
  ``str(list)`` of user turns), so a live conversation lands where the router
  was trained. ``router`` may not import ``training``, so the rendering is
  restated here; ``test_routing_text_matches_training_render_prompt`` pins it.
* **latest user turn** -- what triage and decomposition judge: the request
  being answered now, not the whole history.
* **history** -- every message before that turn, sent to the routed model
  ahead of the latest turn (:func:`with_history`) so it answers in context.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Union

from decompose.conversation import Conversation

__all__ = [
    "AgenticInput",
    "as_messages",
    "routing_text",
    "latest_user_turn",
    "history_before_latest",
    "history_chars",
    "with_history",
]

Message = dict[str, str]
AgenticInput = Union[str, Sequence[Mapping[str, str]], Conversation]


def _sanitize(text: str) -> str:
    """Drop lone surrogates -- same as ``training.data.normalize.sanitize_text``."""
    return str(text).encode("utf-8", "replace").decode("utf-8", "replace")


def as_messages(x: AgenticInput) -> list[Message]:
    """Normalize any accepted input to ``[{"role", "content"}]``. A ``str`` is
    one user message. The last message must be a non-empty user turn -- that
    is the request being answered."""
    if isinstance(x, str):
        msgs = [{"role": "user", "content": x}]
    elif isinstance(x, Conversation):
        msgs = [{"role": m.role.value, "content": m.content} for m in x.messages]
    else:
        msgs = []
        for m in x:
            if "role" not in m or "content" not in m:
                raise ValueError(f"message needs 'role' and 'content': {dict(m)!r}")
            msgs.append({"role": str(m["role"]), "content": str(m["content"])})
    if not msgs or msgs[-1]["role"] != "user" or not msgs[-1]["content"].strip():
        raise ValueError("the last message must be a non-empty user turn")
    return msgs


def routing_text(messages: Sequence[Message]) -> str:
    """The user turns joined with ``"\\n\\n"`` -- the training-time rendering."""
    return _sanitize("\n\n".join(m["content"] for m in messages if m["role"] == "user"))


def latest_user_turn(messages: Sequence[Message]) -> str:
    return messages[-1]["content"]


def history_before_latest(messages: Sequence[Message]) -> tuple[Message, ...]:
    """Every message before the latest user turn (system message included)."""
    return tuple(dict(m) for m in messages[:-1])


def history_chars(history: Sequence[Mapping[str, str]]) -> int:
    """Total characters of a message history's contents (for token estimates)."""
    return sum(len(str(m.get("content", ""))) for m in history)


def with_history(history: Sequence[Message], prompt: str) -> Union[str, list[Message]]:
    """What a routed model is sent: ``prompt`` alone when there is no history
    (single-turn calls are unchanged), else the history plus ``prompt`` as the
    final user turn."""
    if not history:
        return prompt
    return [*map(dict, history), {"role": "user", "content": prompt}]
