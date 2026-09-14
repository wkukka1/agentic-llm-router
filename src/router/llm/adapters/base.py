"""``ProviderAdapter``: the per-provider ``complete``/``stream`` implementation
an :class:`~router.llm.client.LLMClient` delegates to."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from ..client import LLMResponse


class ProviderAdapter(abc.ABC):
    @abc.abstractmethod
    def complete(self, prompt: str) -> "LLMResponse":
        raise NotImplementedError

    @abc.abstractmethod
    def stream(self, prompt: str) -> "LLMResponse":
        raise NotImplementedError
