"""Ollama provider adapter: a local ``ollama serve`` instance through the
official ``ollama`` SDK.

One adapter is one connection to the local server, not one connection per
model -- ``model`` passed to :meth:`OllamaAdapter.__init__` is only a
*default*; every call to :meth:`complete` / :meth:`stream` can override it
with ``model=``, so a single adapter (even one built with no default model
at all) can serve every locally-pulled model. There is no serve-map config
file: the ``model`` string is always the real Ollama tag (e.g.
``"qwen2.5:3b"``), passed straight through to the server.

* ``generation_kwargs`` reach the Ollama call almost unchanged: ``max_tokens``
  is renamed to Ollama's ``num_predict``; ``format`` (a JSON-schema dict, or
  the literal ``"json"``, for structured output) and ``tools`` (function/tool
  specs) are forwarded as top-level ``chat()`` arguments; everything else
  (``temperature``, ``top_p``, ``top_k``, ``stop``, ``seed``, ...) is merged
  into Ollama's ``options`` dict. No allowlist beyond the ``max_tokens``
  rename -- a new Ollama option works without touching this file.
* Local inference is genuinely free: ``LLMResponse.cost`` is always ``0.0``,
  never ``None`` (``None`` means *unknown* elsewhere in this module -- not
  the case here).
* ``client=`` injects a prebuilt ``ollama.Client`` (tests, or sharing one
  connection across several adapters bound to different default models --
  see :meth:`~router.agentic.llm_clients.ClientRegistry.ollama`); otherwise
  one is built lazily from ``host`` (default ``http://localhost:11434``).
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from ..client import LLMResponse
from .base import Prompt, ProviderAdapter

#: top-level ``ollama.Client.chat()`` kwargs, forwarded as-is rather than
#: folded into ``options``
_TOP_LEVEL_KEYS = ("format", "tools")


def _messages(prompt: Prompt) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [{"role": m["role"], "content": m["content"]} for m in prompt]


class OllamaAdapter(ProviderAdapter):
    def __init__(self, model: str | None = None, *, host: str | None = None,
                 api_key: str | None = None, **client_kwargs):
        super().__init__(model, api_key=api_key, **client_kwargs)
        self._host = host or client_kwargs.get("host") or "http://localhost:11434"
        self._client_obj = client_kwargs.get("client")

    # -- SDK client ------------------------------------------------------
    @property
    def _client(self):
        if self._client_obj is None:
            import ollama

            self._client_obj = ollama.Client(host=self._host)
        return self._client_obj

    # -- request / response translation ---------------------------------
    def _resolve_model(self, model: str | None) -> str:
        resolved = model or self.model
        if not resolved:
            raise ValueError(
                "OllamaAdapter needs a model (pass model= per call, or bind one at construction)"
            )
        return resolved

    def _request(self, gen: dict) -> dict[str, Any]:
        gen = dict(gen)
        req: dict[str, Any] = {k: gen.pop(k) for k in _TOP_LEVEL_KEYS if k in gen}
        if "max_tokens" in gen:
            gen["num_predict"] = gen.pop("max_tokens")
        if gen:
            req["options"] = gen
        return req

    def _to_response(self, resp) -> LLMResponse:
        return LLMResponse(
            content=resp.message.content,
            input_tokens=getattr(resp, "prompt_eval_count", None),
            output_tokens=getattr(resp, "eval_count", None),
            cost=0.0,
            raw=resp,
            meta={
                "done_reason": getattr(resp, "done_reason", None),
                "total_duration": getattr(resp, "total_duration", None),
            },
        )

    # -- ProviderAdapter -------------------------------------------------
    def complete(self, prompt: Prompt, *, model: str | None = None,
                 **generation_kwargs) -> LLMResponse:
        resolved = self._resolve_model(model)
        resp = self._client.chat(model=resolved, messages=_messages(prompt),
                                  **self._request(generation_kwargs))
        return self._to_response(resp)

    def stream(self, prompt: Prompt, *, model: str | None = None,
               **generation_kwargs) -> Generator[str, None, LLMResponse]:
        """Yield text deltas; the generator's return value (``StopIteration.value``)
        is the final :class:`LLMResponse`, with usage -- see :meth:`LLMClient.stream`."""
        resolved = self._resolve_model(model)
        final = None
        text_parts: list[str] = []
        for chunk in self._client.chat(model=resolved, messages=_messages(prompt), stream=True,
                                        **self._request(generation_kwargs)):
            piece = chunk.message.content
            if piece:
                text_parts.append(piece)
                yield piece
            if getattr(chunk, "done", False):
                final = chunk
        # the terminal chunk's own `message.content` is typically empty (every
        # text delta already streamed) -- the final LLMResponse's content is
        # the joined deltas, not that chunk's own (likely empty) piece
        resp = self._to_response(final)
        resp.content = "".join(text_parts)
        return resp
