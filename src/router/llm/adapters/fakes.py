"""Deterministic, no-network :class:`~router.llm.adapters.base.ProviderAdapter`
implementations -- for tests and for pools with nothing real behind them yet.

Ports of the old ``router.agentic.llm_clients.EchoClient``/``CallableClient``
onto the adapter-based :class:`~router.llm.client.LLMClient` contract.
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional

from ..client import LLMResponse
from .base import Prompt, ProviderAdapter


class EchoAdapter(ProviderAdapter):
    """No network, deterministic. Returns a short marker string -- enough to
    thread through an orchestrator and assert on in tests."""

    def __init__(self, model: str = "echo", *, prefix: str = "answer",
                 api_key: Optional[str] = None, **client_kwargs):
        super().__init__(model, api_key=api_key, **client_kwargs)
        self.prefix = prefix

    def complete(self, prompt: Prompt, **generation_kwargs) -> LLMResponse:
        snippet = " ".join(str(prompt).split())[:160]
        return LLMResponse(content=f"[{self.prefix}::{self.model}] {snippet}", cost=0.0)

    def stream(self, prompt: Prompt, **generation_kwargs) -> Iterator[str]:
        yield self.complete(prompt, **generation_kwargs).content


class CallableAdapter(ProviderAdapter):
    """Wrap any ``(prompt, **kw) -> str`` (or ``-> LLMResponse``)."""

    def __init__(self, model: str, fn: Callable[..., object], *,
                 api_key: Optional[str] = None, **client_kwargs):
        super().__init__(model, api_key=api_key, **client_kwargs)
        self._fn = fn

    def complete(self, prompt: Prompt, **generation_kwargs) -> LLMResponse:
        out = self._fn(prompt, **generation_kwargs)
        if isinstance(out, LLMResponse):
            return out
        return LLMResponse(content=str(out))

    def stream(self, prompt: Prompt, **generation_kwargs) -> Iterator[str]:
        yield self.complete(prompt, **generation_kwargs).content
