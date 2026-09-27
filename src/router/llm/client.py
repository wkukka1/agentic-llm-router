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

import importlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Mapping, Optional, Union

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from .adapters.base import Prompt, ProviderAdapter
    from .profile import LLMProfile


@dataclass
class LLMResponse:
    """``None`` means *unknown* (e.g. a provider that omits usage while
    streaming) -- distinct from a genuinely free (``0``) call."""

    content: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost: Optional[float] = None
    latency: Optional[float] = None
    #: provider-specific raw response object / metadata (finish reason,
    #: tool-call payloads, moderation flags, ...). Nothing reads these today;
    #: kept so an adapter can attach them without another migration.
    raw: Optional[object] = None
    meta: Optional[dict] = None


class LLMClient:
    """Wraps one :class:`~router.llm.adapters.base.ProviderAdapter`. The
    adapter translates the provider's request/response shape; the client
    times the call and prices it from the ``profile``'s per-token costs when
    the adapter reports token counts but no cost."""

    def __init__(self, adapter: "ProviderAdapter", profile: Optional["LLMProfile"] = None):
        self._adapter = adapter
        self.profile = profile

    def complete(self, prompt: "Prompt", **generation_kwargs) -> LLMResponse:
        t0 = time.perf_counter()
        resp = self._adapter.complete(prompt, **generation_kwargs)
        if resp.latency is None:
            resp.latency = time.perf_counter() - t0
        if resp.cost is None:
            resp.cost = self.price(resp.input_tokens, resp.output_tokens)
        return resp

    def stream(self, prompt: "Prompt", **generation_kwargs) -> Iterator[str]:
        return self._adapter.stream(prompt, **generation_kwargs)

    def price(self, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
        """``in * input_cost_per_token + out * output_cost_per_token``, or
        ``None`` without a profile or token counts, or when the price for a
        side that was actually used (a nonzero token count) is unset -- an
        unpriced profile must not be indistinguishable from a free one."""
        if self.profile is None or (input_tokens is None and output_tokens is None):
            return None
        in_price = self.profile.input_cost_per_token
        out_price = self.profile.output_cost_per_token
        if (input_tokens and in_price is None) or (output_tokens and out_price is None):
            return None
        return float((input_tokens or 0) * (in_price or 0.0)
                     + (output_tokens or 0) * (out_price or 0.0))


#: serving provider -> ``"module:Class"``; imported lazily so importing the
#: factory never requires every provider SDK
#:
#: A different, non-interacting provider vocabulary from
#: ``router.agentic.llm_clients.guess_provider`` (LangChain names) and
#: ``configs/model_registry.yaml`` (model-author names) -- no translation
#: exists between any of the three (XA-10). Harmless today since
#: ``LLMClientFactory``/this module have no production caller (see the module
#: docstring above); would need reconciling before anything ever does feed
#: ``guess_provider``'s output here.
_ADAPTERS: dict[str, str] = {
    "openai": "router.llm.adapters.openai:OpenAIAdapter",
    "anthropic": "router.llm.adapters.anthropic:AnthropicAdapter",
    "google": "router.llm.adapters.google:GoogleAdapter",
}

AdapterSpec = Union[str, type]


def _adapter_class(spec: AdapterSpec) -> type:
    if isinstance(spec, type):
        return spec
    module, _, cls = spec.partition(":")
    return getattr(importlib.import_module(module), cls)


class LLMClientFactory:
    """``LLMProfile -> LLMClient``, picking the adapter by the profile's
    *serving* provider (``serving_provider``, else ``provider``).

    ``adapters`` extends / overrides the provider -> adapter table;
    ``fallback`` (e.g. an OpenAI-compatible endpoint adapter) serves any
    provider not in it. **Adapters** (the SDK client construction) are cached
    per ``(provider, model, api_key, client_kwargs)`` so they aren't rebuilt on
    every request; the returned :class:`LLMClient` is always built fresh
    around the ``profile`` passed in, so a re-priced profile (see
    ``LLMRegistry.register(..., replace=True)``) is never served stale prices
    from an earlier call for the same provider/model.
    """

    def __init__(self, adapters: Optional[Mapping[str, AdapterSpec]] = None, *,
                 fallback: Optional[AdapterSpec] = None):
        self._adapters: dict[str, AdapterSpec] = {**_ADAPTERS, **dict(adapters or {})}
        self._fallback = fallback
        self._cache: dict[tuple, "ProviderAdapter"] = {}

    def client_for(self, profile: "LLMProfile", *, api_key: Optional[str] = None,
                   **client_kwargs) -> LLMClient:
        provider = (getattr(profile, "serving_provider", None) or profile.provider).lower()
        key = (provider, profile.model_id, api_key, repr(sorted(client_kwargs.items())))
        adapter = self._cache.get(key)
        if adapter is None:
            spec = self._adapters.get(provider, self._fallback)
            if spec is None:
                raise ValueError(
                    f"no ProviderAdapter for provider={provider!r}; "
                    f"supported: {sorted(self._adapters)} (pass adapters= or fallback=)"
                )
            Adapter = _adapter_class(spec)
            adapter = Adapter(profile.model_id, api_key=api_key, **client_kwargs)
            self._cache[key] = adapter
        return LLMClient(adapter, profile)
