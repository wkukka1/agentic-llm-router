"""``LLMClient``: the ``complete``/``stream`` interface a
:class:`~router.llm.registry.LLMRegistry` entry resolves to, delegating to a
:class:`~router.llm.adapters.base.ProviderAdapter`.

See :mod:`router.llm` for how this relates to the working
:mod:`router.agentic.llm_clients` layer (``invoke(prompt)``, no
``complete``/``stream`` split, no adapter indirection) -- that one is what
the live agentic flow actually calls today.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .cost import price_tokens

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from .adapters.base import Prompt, ProviderAdapter
    from .profile import LLMProfile


@dataclass
class LLMResponse:
    """``None`` means *unknown* (e.g. a provider that omits usage while
    streaming) -- distinct from a genuinely free (``0``) call."""

    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    latency: float | None = None
    #: provider-specific raw response object / metadata (finish reason,
    #: tool-call payloads, moderation flags, ...). Nothing reads these today;
    #: kept so an adapter can attach them without another migration.
    raw: object | None = None
    meta: dict | None = None


class LLMClient:
    """Wraps one :class:`~router.llm.adapters.base.ProviderAdapter`. The
    adapter translates the provider's request/response shape; the client
    times the call and prices it from the ``profile``'s per-token costs when
    the adapter reports token counts but no cost."""

    def __init__(self, adapter: ProviderAdapter, profile: LLMProfile | None = None):
        self._adapter = adapter
        self.profile = profile

    def complete(self, prompt: Prompt, **generation_kwargs) -> LLMResponse:
        t0 = time.perf_counter()
        resp = self._adapter.complete(prompt, **generation_kwargs)
        if resp.latency is None:
            resp.latency = time.perf_counter() - t0
        if resp.cost is None:
            resp.cost = self.price(resp.input_tokens, resp.output_tokens)
        return resp

    def stream(self, prompt: Prompt, *,
               on_done: Callable[[LLMResponse], None] | None = None,
               **generation_kwargs) -> Generator[str, None, LLMResponse]:
        """Yield the adapter's text chunks as they arrive; when the stream ends,
        build the final :class:`LLMResponse` -- the adapter's generator return
        value when it has one (usage included), else the joined chunks -- with
        ``latency`` and ``meta["ttft"]`` (time to first chunk) filled in and
        priced like :meth:`complete`. That response is the generator's return
        value and is also passed to ``on_done``: bookkeeping hangs off the end
        of the stream, never in front of the first token."""
        t0 = time.perf_counter()
        ttft: float | None = None
        chunks: list[str] = []
        it = self._adapter.stream(prompt, **generation_kwargs)
        while True:
            try:
                chunk = next(it)
            except StopIteration as stop:
                resp = stop.value
                break
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks.append(chunk)
            yield chunk
        if not isinstance(resp, LLMResponse):
            resp = LLMResponse(content="".join(chunks))
        if resp.latency is None:
            resp.latency = time.perf_counter() - t0
        resp.meta = {**(resp.meta or {}), "ttft": ttft if ttft is not None else resp.latency}
        if resp.cost is None:
            resp.cost = self.price(resp.input_tokens, resp.output_tokens)
        if on_done is not None:
            on_done(resp)
        return resp

    def price(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        """``in * input_cost_per_token + out * output_cost_per_token``, or
        ``None`` without a profile or token counts, or when the price for a
        side that was actually used (a nonzero token count) is unset -- an
        unpriced profile must not be indistinguishable from a free one."""
        if self.profile is None:
            return None
        return price_tokens(input_tokens, output_tokens,
                            self.profile.input_cost_per_token, self.profile.output_cost_per_token)
