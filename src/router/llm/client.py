"""``LLMClient``: the ``complete``/``stream`` interface a
:class:`~router.llm.registry.LLMRegistry` entry resolves to via
:class:`LLMClientFactory`, delegating to a
:class:`~router.llm.adapters.base.ProviderAdapter`.

See :mod:`router.llm` for how this relates to the working
:mod:`router.agentic.llm_clients` layer (``invoke(prompt)``, no
``complete``/``stream`` split, no adapter indirection) -- that one is what
the live agentic flow actually calls today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from .adapters.base import ProviderAdapter
    from .profile import LLMProfile


@dataclass
class LLMResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency: float = 0.0


class LLMClient:
    """Wraps one :class:`~router.llm.adapters.base.ProviderAdapter` and just
    delegates -- the adapter is where a provider's actual request/response
    shape gets translated to :class:`LLMResponse`."""

    def __init__(self, adapter: "ProviderAdapter"):
        self._adapter = adapter

    def complete(self, prompt: str) -> LLMResponse:
        return self._adapter.complete(prompt)

    def stream(self, prompt: str) -> LLMResponse:
        return self._adapter.stream(prompt)


class LLMClientFactory:
    """``LLMProfile -> LLMClient``, picking the adapter by ``profile.provider``.

    Adapter modules are imported lazily (inside :meth:`client_for`, not at
    module load) so importing this factory never requires every provider's
    optional dependency to be installed.
    """

    _PROVIDERS = {"openai", "anthropic", "google"}

    def client_for(self, profile: "LLMProfile", *, api_key: Optional[str] = None) -> LLMClient:
        provider = profile.provider.lower()
        if provider == "openai":
            from .adapters.openai import OpenAIAdapter as Adapter
        elif provider == "anthropic":
            from .adapters.anthropic import AnthropicAdapter as Adapter
        elif provider == "google":
            from .adapters.google import GoogleAdapter as Adapter
        else:
            raise ValueError(
                f"no ProviderAdapter for provider={profile.provider!r}; "
                f"supported: {sorted(self._PROVIDERS)}"
            )
        return LLMClient(Adapter(profile.model_id, api_key=api_key))
