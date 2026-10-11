"""``ProviderAdapter``: the per-provider ``complete``/``stream`` implementation
an :class:`~router.llm.client.LLMClient` delegates to."""

from __future__ import annotations

import abc
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from ..client import LLMResponse

#: a bare user prompt, or a chat message list (``[{"role": ..., "content": ...}]``,
#: system prompt / history included)
Prompt = str | Sequence[Mapping[str, str]]


class ProviderAdapter(abc.ABC):
    """``client_kwargs`` (timeout, base_url, default ``max_tokens`` ...) configure
    the SDK client; ``generation_kwargs`` on each call (``max_tokens``,
    ``temperature``, ``stop`` ...) configure one request."""

    def __init__(self, model: str, *, api_key: str | None = None, **client_kwargs):
        self.model = model
        self.api_key = api_key
        self.client_kwargs = client_kwargs

    @abc.abstractmethod
    def complete(self, prompt: Prompt, **generation_kwargs) -> LLMResponse:
        raise NotImplementedError

    @abc.abstractmethod
    def stream(self, prompt: Prompt, **generation_kwargs) -> Iterator[str]:
        """Yield text chunks as they arrive."""
        raise NotImplementedError
