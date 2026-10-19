"""Anthropic provider adapter: the Messages API through the official ``anthropic`` SDK.

``evaluation.judge.client.JudgeClient`` also calls the Anthropic API, but only
for offline pairwise-judge labeling (``docs/anchor_judge.md``) and can't be
imported from ``router``; this is the serving-side adapter.

* ``max_tokens`` is required by the Messages API: taken from the call's
  ``generation_kwargs``, else ``client_kwargs["max_tokens"]`` (both reach the
  adapter through ``client_kwargs`` / ``LLMClient.complete``), else ``ValueError``.
* A chat-list prompt's leading ``system`` messages become the ``system=``
  parameter; the rest is sent as ``messages``.
* **Prompt caching** is on by default (``cache=False`` turns it off): the API
  caches by content-block breakpoint, not automatically, so the adapter marks
  the request's earliest block cacheable -- the system prompt's block if
  there is one (the stable part, reused by every call), else the first
  message's block. Anthropic's own prefix matching then finds and reuses the
  longest cached prefix on every later call sharing that same lead content --
  lower time to first token and ~0.1x input price on the cached part.
  Prefixes under the model's minimum (512-4096 tokens) silently don't cache.
* ``LLMResponse.input_tokens`` counts every prompt token (uncached + cache
  read + cache write); the split is in ``meta``. With ``input_cost_per_token``
  / ``output_cost_per_token`` in ``client_kwargs`` the adapter prices the call
  itself, cache-aware (reads 0.1x, 5-minute writes 1.25x); otherwise ``cost``
  is ``None`` and :class:`~router.llm.client.LLMClient` prices it from the
  profile at the full input rate.
* ``stop_reason == "refusal"`` raises ``RuntimeError`` -- a refusal is not an
  answer to route on.
* ``client=`` injects a prebuilt SDK client (tests, or sharing one connection
  pool across adapters); otherwise one ``anthropic.Anthropic`` is built lazily
  per adapter from ``api_key`` / ``timeout`` / ``max_retries`` / ``base_url``.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from ..client import LLMResponse
from ..cost import price_tokens
from .base import Prompt, ProviderAdapter

#: Anthropic cache pricing relative to the base input rate
_CACHE_READ_MULTIPLIER = 0.1
_CACHE_WRITE_MULTIPLIER = 1.25          # 5-minute TTL (the default ``ephemeral``)

_SDK_CLIENT_KEYS = ("timeout", "max_retries", "base_url")


class AnthropicAdapter(ProviderAdapter):
    def __init__(self, model: str, *, api_key: str | None = None, **client_kwargs):
        super().__init__(model, api_key=api_key, **client_kwargs)
        self._client_obj = client_kwargs.get("client")

    # -- SDK client ------------------------------------------------------
    @property
    def _client(self):
        if self._client_obj is None:
            import anthropic

            kw = {k: self.client_kwargs[k] for k in _SDK_CLIENT_KEYS if k in self.client_kwargs}
            if self.api_key is not None:
                kw["api_key"] = self.api_key
            self._client_obj = anthropic.Anthropic(**kw)
        return self._client_obj

    # -- request / response translation ---------------------------------
    def _request(self, prompt: Prompt, gen: dict) -> dict[str, Any]:
        gen = dict(gen)
        max_tokens = gen.pop("max_tokens", None) or self.client_kwargs.get("max_tokens")
        if not max_tokens:
            raise ValueError(
                "AnthropicAdapter needs max_tokens (pass it per call or as a client kwarg)"
            )
        if isinstance(prompt, str):
            system, messages = None, [{"role": "user", "content": prompt}]
        else:
            msgs = [dict(m) for m in prompt]
            lead = 0
            while lead < len(msgs) and msgs[lead].get("role") == "system":
                lead += 1
            system = "\n\n".join(m["content"] for m in msgs[:lead]) or None
            messages = [{"role": m["role"], "content": m["content"]} for m in msgs[lead:]]
        cache = bool(self.client_kwargs.get("cache", True))
        if system and cache:
            system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        elif cache and messages:
            first = messages[0]
            messages[0] = {**first, "content": [
                {"type": "text", "text": first["content"], "cache_control": {"type": "ephemeral"}}
            ]}
        req: dict[str, Any] = {"model": self.model, "max_tokens": int(max_tokens),
                               "messages": messages, **gen}
        if system:
            req["system"] = system
        return req

    def _to_response(self, message) -> LLMResponse:
        if getattr(message, "stop_reason", None) == "refusal":
            details = getattr(message, "stop_details", None)
            raise RuntimeError(f"{self.model} refused the request ({details!r})")
        text = "".join(b.text for b in message.content if getattr(b, "type", None) == "text")
        usage = message.usage
        uncached = int(getattr(usage, "input_tokens", 0) or 0)
        read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        return LLMResponse(
            content=text,
            input_tokens=uncached + read + write,
            output_tokens=out,
            cost=self._cost(uncached, read, write, out),
            raw=message,
            meta={
                "stop_reason": message.stop_reason,
                "cache_read_input_tokens": read,
                "cache_creation_input_tokens": write,
                "request_id": getattr(message, "_request_id", None),
            },
        )

    def _cost(self, uncached: int, read: int, write: int, out: int) -> float | None:
        effective_in = uncached + _CACHE_READ_MULTIPLIER * read + _CACHE_WRITE_MULTIPLIER * write
        return price_tokens(effective_in, out, self.client_kwargs.get("input_cost_per_token"),
                            self.client_kwargs.get("output_cost_per_token"))

    # -- ProviderAdapter -------------------------------------------------
    def complete(self, prompt: Prompt, **generation_kwargs) -> LLMResponse:
        return self._to_response(self._client.messages.create(**self._request(prompt, generation_kwargs)))

    def stream(self, prompt: Prompt, **generation_kwargs) -> Generator[str, None, LLMResponse]:
        """Yield text deltas; the generator's return value (``StopIteration.value``)
        is the final :class:`LLMResponse`, with usage -- see :meth:`LLMClient.stream`."""
        with self._client.messages.stream(**self._request(prompt, generation_kwargs)) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        return self._to_response(final)
