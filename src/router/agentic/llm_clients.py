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
the orchestrator.
"""

from __future__ import annotations

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
    ``init_chat_model`` (optionally with ``provider=``). Lazily imported."""

    def __init__(
        self,
        model_id: str,
        chat_model: object = None,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **init_kwargs,
    ):
        self.model_id = str(model_id)
        self._chat = chat_model
        self._spec = (model or model_id, provider or guess_provider(model_id), init_kwargs)

    def _model(self):
        if self._chat is None:
            from langchain.chat_models import init_chat_model  # optional dep

            name, provider, kw = self._spec
            self._chat = init_chat_model(name, model_provider=provider, **kw)
        return self._chat

    def invoke(self, prompt: str, **kw) -> LLMResponse:
        msg = self._model().invoke(prompt)
        text = getattr(msg, "content", msg)
        meta = getattr(msg, "response_metadata", {}) or {}
        usage = getattr(msg, "usage_metadata", None)
        return LLMResponse(
            text=text if isinstance(text, str) else str(text),
            model_id=self.model_id,
            raw=msg,
            meta={"response_metadata": meta, "usage_metadata": usage},
        )


# --------------------------------------------------------------------------- #
# provider guessing                                                           #
# --------------------------------------------------------------------------- #
_PROVIDER_HINTS = [
    ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"), ("davinci", "openai"),
    ("claude", "anthropic"),
    ("gemini", "google_genai"), ("palm", "google_genai"), ("bison", "google_genai"),
    ("mistral", "mistralai"), ("mixtral", "mistralai"),
    ("llama", "groq"), ("command", "cohere"), ("qwen", "together"),
]


def guess_provider(model_id: str) -> Optional[str]:
    """Best-effort LangChain ``model_provider`` from a bare model id."""
    low = str(model_id).lower()
    for needle, provider in _PROVIDER_HINTS:
        if needle in low:
            return provider
    return None


# --------------------------------------------------------------------------- #
# registry                                                                    #
# --------------------------------------------------------------------------- #
class ClientRegistry:
    """``model_id -> LLMClient``, with a factory for unknown ids.

    ``default_factory(model_id) -> LLMClient`` is called (and the result
    cached) the first time an unregistered id is looked up. It defaults to
    :class:`EchoClient`, so an orchestrator over an unfamiliar pool still runs.
    """

    def __init__(
        self,
        clients: Optional[Mapping[str, LLMClient]] = None,
        *,
        default_factory: Optional[Callable[[str], LLMClient]] = None,
    ):
        self._clients: dict[str, LLMClient] = dict(clients or {})
        self._default_factory = default_factory or (lambda mid: EchoClient(mid))

    @classmethod
    def echo(cls, model_ids=()) -> "ClientRegistry":
        return cls({m: EchoClient(m) for m in model_ids})

    @classmethod
    def langchain(cls, model_ids=(), **per_model_kwargs) -> "ClientRegistry":
        """Every id served by a :class:`LangChainClient` (provider guessed)."""
        return cls(
            {m: LangChainClient(m, **per_model_kwargs.get(m, {})) for m in model_ids},
            default_factory=lambda mid: LangChainClient(mid),
        )

    def register(self, model_id: str, client: LLMClient) -> None:
        self._clients[str(model_id)] = client

    def get(self, model_id: str) -> LLMClient:
        mid = str(model_id)
        if mid not in self._clients:
            self._clients[mid] = self._default_factory(mid)
        return self._clients[mid]

    def __contains__(self, model_id: str) -> bool:
        return str(model_id) in self._clients

    def invoke(self, model_id: str, prompt: str, **kw) -> LLMResponse:
        return self.get(model_id).invoke(prompt, **kw)
