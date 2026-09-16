"""LLM clients: how the orchestrator actually *calls* a routed model.

A :class:`Router` decides *which* ``model_id`` should answer; a client turns
that id into a callable LLM. Clients are pluggable so the same orchestrator runs
against real APIs in production and against a deterministic echo in tests.

* :class:`EchoClient`      -- no network, deterministic. The default.
* :class:`CallableClient`  -- wrap any ``str -> str`` function.
* :class:`LangChainClient` -- a LangChain chat model (an instance, or a model
  string for ``langchain.chat_models.init_chat_model``).

:class:`ClientRegistry` maps ``model_id -> client`` and synthesises a default
client for ids it has never seen, so an unfamiliar candidate pool never crashes
the orchestrator. When the registry was given real clients, synthesising an
echo client for a missing id warns (its answers are placeholders).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

__all__ = [
    "LLMResponse",
    "LLMClient",
    "EchoClient",
    "CallableClient",
    "LangChainClient",
    "ClientRegistry",
    "guess_provider",
    "message_text",
]


# --------------------------------------------------------------------------- #
# response                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class LLMResponse:
    text: str
    model_id: str
    cost: Optional[float] = None
    raw: object = None
    meta: dict = field(default_factory=dict)


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


class LLMClient:
    """Interface: ``invoke(prompt, **kw) -> LLMResponse``. ``model_id`` is set."""

    model_id: str = "?"

    def invoke(self, prompt: str, **kw) -> LLMResponse:  # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# concrete clients                                                            #
# --------------------------------------------------------------------------- #
class EchoClient(LLMClient):
    """Deterministic, no-network. Returns a short marker string -- enough to
    thread through an orchestrator and assert on in tests."""

    def __init__(self, model_id: str, *, prefix: str = "answer"):
        self.model_id = str(model_id)
        self._prefix = prefix

    def invoke(self, prompt: str, **kw) -> LLMResponse:
        snippet = " ".join(str(prompt).split())[:160]
        return LLMResponse(
            text=f"[{self._prefix}::{self.model_id}] {snippet}",
            model_id=self.model_id,
            cost=0.0,
            meta={"echo": True},
        )


class CallableClient(LLMClient):
    """Wrap any ``(prompt: str, **kw) -> str`` (or ``-> LLMResponse``)."""

    def __init__(self, model_id: str, fn: Callable[..., object]):
        self.model_id = str(model_id)
        self._fn = fn

    def invoke(self, prompt: str, **kw) -> LLMResponse:
        out = self._fn(prompt, **kw)
        if isinstance(out, LLMResponse):
            return out
        return LLMResponse(text=str(out), model_id=self.model_id)


class LangChainClient(LLMClient):
    """A LangChain chat model. Pass an already-built model, or a string for
    ``init_chat_model`` (optionally with ``provider=``). Lazily imported.

    ``model=`` is the provider's API model name when it differs from the pool
    id (RouterBench / IRT-Router ids are usually not valid API names).
    ``input_cost_per_token`` / ``output_cost_per_token`` price the call from
    the response's ``usage_metadata``; without them ``cost`` stays ``None``
    (unknown) and the token counts are still recorded in ``meta``.
    """

    def __init__(
        self,
        model_id: str,
        chat_model: object = None,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        input_cost_per_token: Optional[float] = None,
        output_cost_per_token: Optional[float] = None,
        **init_kwargs,
    ):
        self.model_id = str(model_id)
        self._chat = chat_model
        self._spec = (model or model_id, provider or guess_provider(model or model_id), init_kwargs)
        self._prices = (input_cost_per_token, output_cost_per_token)

    def _model(self):
        if self._chat is None:
            from langchain.chat_models import init_chat_model  # optional dep

            name, provider, kw = self._spec
            self._chat = init_chat_model(name, model_provider=provider, **kw)
        return self._chat

    def invoke(self, prompt: str, **kw) -> LLMResponse:
        msg = self._model().invoke(prompt)
        meta = getattr(msg, "response_metadata", {}) or {}
        usage = getattr(msg, "usage_metadata", None)
        return LLMResponse(
            text=message_text(msg),
            model_id=self.model_id,
            cost=self._cost(usage),
            raw=msg,
            meta={"response_metadata": meta, "usage_metadata": usage},
        )

    def _cost(self, usage) -> Optional[float]:
        in_price, out_price = self._prices
        if usage is None or (in_price is None and out_price is None):
            return None
        get = usage.get if isinstance(usage, Mapping) else (lambda k, d=None: getattr(usage, k, d))
        in_tok, out_tok = get("input_tokens"), get("output_tokens")
        if in_tok is None and out_tok is None:
            return None
        return float((in_tok or 0) * (in_price or 0.0) + (out_tok or 0) * (out_price or 0.0))


# --------------------------------------------------------------------------- #
# provider guessing                                                           #
# --------------------------------------------------------------------------- #
# LangChain `model_provider` names, for `init_chat_model` (LangChainClient
# below) only. Two other, different provider vocabularies exist in this
# codebase -- `router.llm.client`'s `_ADAPTERS` (`openai`/`anthropic`/`google`
# only) and `configs/model_registry.yaml`'s `provider:` field (model-author
# names, e.g. `meta`, `mistralai`, `zhipu`) -- with no translation between any
# of the three (XA-10). They don't interact today: `guess_provider`'s output
# never reaches `router.llm.client` or the registry.
_PROVIDER_HINTS = [
    ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"), ("davinci", "openai"),
    ("claude", "anthropic"),
    ("gemini", "google_genai"), ("palm", "google_genai"), ("bison", "google_genai"),
    ("mistral", "mistralai"), ("mixtral", "mistralai"),
    ("llama", "groq"), ("command", "cohere"), ("qwen", "together"),
]


def guess_provider(model_id: str) -> Optional[str]:
    """Best-effort LangChain ``model_provider`` from a bare model id -- a last
    resort. Each hint must start a token (``o1-mini`` matches ``o1``,
    ``yolo1`` does not). Prefer an explicit ``provider=`` / ``model=`` per pool
    id (``ClientRegistry.langchain(pool, **{id: {...}})``): a substring can't
    tell which host actually serves a given model."""
    low = str(model_id).lower()
    for needle, provider in _PROVIDER_HINTS:
        if re.search(rf"(?:^|[^a-z0-9]){re.escape(needle)}", low):
            return provider
    return None


# --------------------------------------------------------------------------- #
# registry                                                                    #
# --------------------------------------------------------------------------- #
def _echo_factory(model_id: str) -> LLMClient:
    return EchoClient(model_id)


class ClientRegistry:
    """``model_id -> LLMClient``, with a factory for unknown ids.

    ``default_factory(model_id) -> LLMClient`` is called (and the result
    cached) the first time an unregistered id is looked up. It defaults to
    :class:`EchoClient`, so an orchestrator over an unfamiliar pool still runs;
    ids that got a synthesised client are listed in :attr:`synthesized`.
    """

    def __init__(
        self,
        clients: Optional[Mapping[str, LLMClient]] = None,
        *,
        default_factory: Optional[Callable[[str], LLMClient]] = None,
    ):
        self._clients: dict[str, LLMClient] = dict(clients or {})
        self._default_factory = default_factory or _echo_factory
        self._has_real_clients = bool(self._clients)
        self.synthesized: set[str] = set()

    @classmethod
    def echo(cls, model_ids=()) -> "ClientRegistry":
        return cls({m: EchoClient(m) for m in model_ids})

    @classmethod
    def langchain(cls, model_ids=(), **per_model_kwargs) -> "ClientRegistry":
        """Every id served by a :class:`LangChainClient` (provider guessed unless
        ``per_model_kwargs[id]`` gives ``provider=`` / ``model=``)."""
        return cls(
            {m: LangChainClient(m, **per_model_kwargs.get(m, {})) for m in model_ids},
            default_factory=lambda mid: LangChainClient(mid),
        )

    def register(self, model_id: str, client: LLMClient) -> None:
        self._clients[str(model_id)] = client

    def get(self, model_id: str) -> LLMClient:
        mid = str(model_id)
        if mid not in self._clients:
            client = self._default_factory(mid)
            if self._default_factory is _echo_factory and self._has_real_clients:
                warnings.warn(
                    f"no client registered for model {mid!r}; answering with an EchoClient "
                    "placeholder",
                    stacklevel=2,
                )
            self._clients[mid] = client
            self.synthesized.add(mid)
        return self._clients[mid]

    def __contains__(self, model_id: str) -> bool:
        return str(model_id) in self._clients

    def invoke(self, model_id: str, prompt: str, **kw) -> LLMResponse:
        return self.get(model_id).invoke(prompt, **kw)
