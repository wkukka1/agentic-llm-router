"""``LangChainAdapter``: the one :class:`~router.llm.adapters.base.ProviderAdapter`
that actually reaches a network today -- the ``openai``/``anthropic``/``google``
adapters are unbuilt stubs. Wraps a LangChain chat model (an instance, or a
model string for ``langchain.chat_models.init_chat_model``).

``guess_provider``/``message_text`` live here (moved from the old
``router.agentic.llm_clients``, which re-exports them for the handful of
unrelated callers -- ``router.agentic.triage``/``decompose`` -- that only
ever needed ``message_text`` to flatten a LangChain message, never the old
``invoke``-based client contract).
"""

from __future__ import annotations

import re
from typing import Iterator, Mapping, Optional

from ..client import LLMResponse
from .base import Prompt, ProviderAdapter

#: LangChain `model_provider` names, for `init_chat_model`. A different,
#: non-interacting provider vocabulary from `router.llm.client`'s `_ADAPTERS`
#: (`openai`/`anthropic`/`google` only) and `configs/model_registry.yaml`'s
#: `provider:` field (model-author names) -- no translation exists between
#: any of the three (XA-10).
_PROVIDER_HINTS = [
    ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"), ("davinci", "openai"),
    ("claude", "anthropic"),
    ("gemini", "google_genai"), ("palm", "google_genai"), ("bison", "google_genai"),
    ("mistral", "mistralai"), ("mixtral", "mistralai"),
    ("llama", "groq"), ("command", "cohere"), ("qwen", "together"),
]


def guess_provider(model_id: str) -> Optional[str]:
    """Best-effort LangChain ``model_provider`` from a bare model id -- a last
    resort. Each hint must start its own token and end on a boundary; a hint
    may be followed by a version number such as ``llama3`` / ``qwen2.5`` /
    ``gemini15``, but not by more letters (``o1-mini`` matches ``o1``,
    ``yolo1`` does not, ``gpt4all-13b-snoozy`` does not).
    Prefer an explicit ``provider=``/``model=`` per pool id: a substring
    can't tell which host actually serves a given model."""
    low = str(model_id).lower()
    for needle, provider in _PROVIDER_HINTS:
        if re.search(rf"(?:^|[^a-z0-9]){re.escape(needle)}(?:\d+(?:\.\d+)*)?(?:$|[^a-z0-9])", low):
            return provider
    return None


def message_text(msg: object) -> str:
    """Plain text of a LangChain message (or anything else).

    ``AIMessage.content`` may be a list of content blocks (tool use, thinking,
    multimodal output); ``str()`` of that is a Python repr, not an answer. Text
    blocks are joined and other block types skipped."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and block.get("type", "text") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


class LangChainAdapter(ProviderAdapter):
    """``model`` is the provider's API model name (or a pool id ``guess_provider``
    can read a provider hint off of). ``input_cost_per_token``/
    ``output_cost_per_token`` price the call from the response's
    ``usage_metadata`` directly -- not through an ``LLMProfile`` -- since
    router pool ids are frequently not real API model names with a profile
    entry (:class:`~router.llm.client.LLMClient` only falls back to
    profile-based pricing when an adapter leaves ``cost=None``, so this is
    fully compatible with that mechanism)."""

    def __init__(
        self,
        model: str,
        chat_model: object = None,
        *,
        provider: Optional[str] = None,
        input_cost_per_token: Optional[float] = None,
        output_cost_per_token: Optional[float] = None,
        api_key: Optional[str] = None,
        **client_kwargs,
    ):
        super().__init__(model, api_key=api_key, **client_kwargs)
        self._chat = chat_model
        self._spec = (model, provider or guess_provider(model), client_kwargs)
        self._prices = (input_cost_per_token, output_cost_per_token)

    def _model(self):
        if self._chat is None:
            from langchain.chat_models import init_chat_model  # optional dep

            name, provider, kw = self._spec
            if self.api_key is not None:
                kw = {**kw, "api_key": self.api_key}
            self._chat = init_chat_model(name, model_provider=provider, **kw)
        return self._chat

    def complete(self, prompt: Prompt, **generation_kwargs) -> LLMResponse:
        msg = self._model().invoke(prompt, **generation_kwargs)
        in_tok, out_tok = self._tokens(getattr(msg, "usage_metadata", None))
        return LLMResponse(
            content=message_text(msg),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost=self._cost(in_tok, out_tok),
        )

    def stream(self, prompt: Prompt, **generation_kwargs) -> Iterator[str]:
        for chunk in self._model().stream(prompt, **generation_kwargs):
            yield message_text(chunk)

    @staticmethod
    def _tokens(usage) -> tuple[Optional[int], Optional[int]]:
        if usage is None:
            return None, None
        get = usage.get if isinstance(usage, Mapping) else (lambda k, d=None: getattr(usage, k, d))
        return get("input_tokens"), get("output_tokens")

    def _cost(self, in_tok: Optional[int], out_tok: Optional[int]) -> Optional[float]:
        """``None`` (unknown) when there are no token counts at all, or when a
        side with a nonzero token count has no configured price -- an unpriced
        side must not be silently treated as free (LANGCHAINADAPTER-002)."""
        in_price, out_price = self._prices
        if in_tok is None and out_tok is None:
            return None
        if (in_tok and in_price is None) or (out_tok and out_price is None):
            return None
        return float((in_tok or 0) * (in_price or 0.0) + (out_tok or 0) * (out_price or 0.0))
