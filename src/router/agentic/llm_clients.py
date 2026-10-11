"""LLM clients: how the orchestrator actually *calls* a routed model.

A :class:`~router.routing.base.RouterModel` decides *which* ``model_id`` should
answer; :class:`~router.llm.client.LLMClient` turns that id into a callable
LLM via a :class:`~router.llm.adapters.base.ProviderAdapter`. Clients are
pluggable so the same orchestrator runs against real APIs in production and
against a deterministic echo in tests.

* ``EchoAdapter``      (:mod:`router.llm.adapters.fakes`) -- no network,
  deterministic. The default.
* ``CallableAdapter``  (:mod:`router.llm.adapters.fakes`) -- wrap any
  ``str -> str`` function.
* ``LangChainAdapter`` (:mod:`router.llm.adapters.langchain_adapter`) -- a
  LangChain chat model (an instance, or a model string for
  ``langchain.chat_models.init_chat_model``).
* ``OllamaAdapter``     (:mod:`router.llm.adapters.ollama`) -- a local
  ``ollama serve`` instance; no serve-map, ids are real Ollama tags.

:class:`ClientRegistry` maps ``model_id -> LLMClient`` and synthesises a
default client for ids it has never seen, so an unfamiliar candidate pool
never crashes the orchestrator. When the registry was given real clients,
synthesising an *echo* client for a missing id raises instead -- a registry
that is answering real requests must not silently start faking answers for
an id it was never given a client for.

``guess_provider``/``message_text`` now live in
:mod:`router.llm.adapters.langchain_adapter`; re-exported here for
:mod:`router.agentic.triage`/:mod:`router.agentic.decompose`, which only ever
needed ``message_text`` to flatten a LangChain message.
"""

from __future__ import annotations

from typing import Callable, Generator, Mapping, Optional, Sequence

import numpy as np

from ..llm.adapters.fakes import EchoAdapter
from ..llm.adapters.langchain_adapter import LangChainAdapter, guess_provider, message_text
from ..llm.client import LLMClient, LLMResponse

__all__ = [
    "LLMResponse",
    "LLMClient",
    "ClientRegistry",
    "guess_provider",
    "message_text",
]


# --------------------------------------------------------------------------- #
# registry                                                                    #
# --------------------------------------------------------------------------- #
def _echo_factory(model_id: str) -> LLMClient:
    return LLMClient(EchoAdapter(model_id))


class ClientRegistry:
    """``model_id -> LLMClient``, with a factory for unknown ids.

    ``default_factory(model_id) -> LLMClient`` is called (and the result
    cached) the first time an unregistered id is looked up. It defaults to an
    echo-backed client, so an orchestrator over an unfamiliar pool still runs;
    ids that got a synthesised client are listed in :attr:`synthesized`.

    That echo default is only safe when the whole registry is echo (nothing
    real to fall back from -- demos, tests). Once it holds any real client,
    an unregistered id raises :class:`LookupError` instead of synthesising an
    echo client: a mix of real and echo answers in the same run would look
    like every model answered when one didn't (CLIENTREGISTRY-001).
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
        return cls({m: LLMClient(EchoAdapter(m)) for m in model_ids})

    @classmethod
    def langchain(cls, model_ids=(), **per_model_kwargs) -> "ClientRegistry":
        """Every id served by a :class:`~router.llm.adapters.langchain_adapter.LangChainAdapter`
        (provider guessed unless ``per_model_kwargs[id]`` gives ``provider=``
        / ``model=``)."""
        def _build(pool_id: str, kwargs: dict) -> LLMClient:
            api_model = kwargs.pop("model", pool_id)
            return LLMClient(LangChainAdapter(api_model, **kwargs))

        return cls(
            {m: _build(m, dict(per_model_kwargs.get(m, {}))) for m in model_ids},
            default_factory=lambda mid: LLMClient(LangChainAdapter(mid)),
        )

    @classmethod
    def ollama(cls, model_ids: Sequence[str] = (), *, host: Optional[str] = None,
               client: object = None, **client_kwargs) -> "ClientRegistry":
        """Every id served through :class:`~router.llm.adapters.ollama.OllamaAdapter`,
        all sharing one ``ollama.Client`` connection -- ``model_ids`` are real
        Ollama tags (e.g. ``"qwen2.5:3b"``), not pool ids needing translation
        (no serve-map file, unlike :meth:`anthropic`). Ids not passed in still
        work through the default factory, which reuses the same connection.
        ``client=`` injects a prebuilt ``ollama.Client`` (tests; otherwise one
        is built from ``host``, default ``http://localhost:11434``)."""
        from ..llm.adapters.ollama import OllamaAdapter

        if client is None:
            import ollama

            client = ollama.Client(host=host or "http://localhost:11434")

        def build(mid: str) -> LLMClient:
            return LLMClient(OllamaAdapter(mid, client=client, **client_kwargs))

        return cls({m: build(m) for m in model_ids}, default_factory=build)

    @classmethod
    def anthropic(cls, serve_map: Mapping, *, pool: Sequence[str] = (),
                  **client_kwargs) -> "ClientRegistry":
        """Every pool id served by a Claude model through
        :class:`~router.llm.adapters.anthropic.AnthropicAdapter`, per a serve
        map (``configs/serve_anthropic.yaml``): ``pool`` maps a pool id to a
        tier, ``tiers`` a tier to a Claude model id, ``models`` a Claude id to
        its per-token prices; ``default_tier`` serves ids not listed (``None``
        -> ``KeyError``). One adapter (one SDK client) per Claude model, shared
        by every pool id on that tier. ``client_kwargs`` (``api_key=``,
        ``timeout=``, ``client=`` ...) reach every adapter; the map's
        ``max_tokens`` is the per-call default."""
        from ..llm.adapters.anthropic import AnthropicAdapter
        from ..llm.profile import LLMProfile

        tiers, models = dict(serve_map["tiers"]), dict(serve_map.get("models") or {})
        tier_of = dict(serve_map.get("pool") or {})
        default_tier = serve_map.get("default_tier")
        api_key = client_kwargs.pop("api_key", None)
        if serve_map.get("max_tokens") is not None:
            client_kwargs.setdefault("max_tokens", int(serve_map["max_tokens"]))
        adapters: dict[str, AnthropicAdapter] = {}

        def build(pool_id: str) -> LLMClient:
            tier = tier_of.get(pool_id, default_tier)
            if tier is None:
                raise KeyError(f"pool model {pool_id!r} has no tier in the serve map "
                               "and there is no default_tier")
            claude_id = tiers[tier]
            prices = dict(models.get(claude_id) or {})
            if claude_id not in adapters:
                adapters[claude_id] = AnthropicAdapter(claude_id, api_key=api_key,
                                                       **prices, **client_kwargs)
            profile = LLMProfile(name=pool_id, provider="anthropic", model_id=claude_id,
                                 input_cost_per_token=prices.get("input_cost_per_token"),
                                 output_cost_per_token=prices.get("output_cost_per_token"))
            return LLMClient(adapters[claude_id], profile)

        return cls({m: build(m) for m in pool}, default_factory=build)

    def register(self, model_id: str, client: LLMClient) -> None:
        self._clients[str(model_id)] = client
        # a client added directly is exactly the "real client" case the
        # echo-fallback error below exists to protect (CLIENTREGISTRY-001)
        self._has_real_clients = True

    def get(self, model_id: str) -> LLMClient:
        mid = str(model_id)
        if mid not in self._clients:
            if self._default_factory is _echo_factory and self._has_real_clients:
                raise LookupError(
                    f"no client registered for model {mid!r}; refusing to answer with an "
                    "echo placeholder alongside real clients -- register() it or pass a "
                    "different default_factory"
                )
            client = self._default_factory(mid)
            self._clients[mid] = client
            self.synthesized.add(mid)
        return self._clients[mid]

    def can_serve(self, model_id: str) -> bool:
        """Whether :meth:`get` would return a client for ``model_id`` without
        raising -- i.e. whether a router may legally select it. Side-effect-free
        (never calls the factory, never mutates :attr:`_clients`/:attr:`synthesized`),
        so it's safe to probe every candidate in a pool speculatively, not just the
        one that ends up chosen (CLIENTREGISTRY-001's eligibility counterpart)."""
        mid = str(model_id)
        if mid in self._clients:
            return True
        return not (self._default_factory is _echo_factory and self._has_real_clients)

    def eligible_mask(self, model_ids: Sequence[str]) -> np.ndarray:
        """``[len(model_ids)]`` bool array, :meth:`can_serve` per id, in order --
        the ``eligible=`` vector for ``RouterModel.route``/``route_text`` over
        this registry's servable models."""
        return np.array([self.can_serve(m) for m in model_ids], dtype=bool)

    def __contains__(self, model_id: str) -> bool:
        return str(model_id) in self._clients

    def complete(self, model_id: str, prompt: str, **kw) -> LLMResponse:
        return self.get(model_id).complete(prompt, **kw)

    def stream(self, model_id: str, prompt, **kw) -> Generator[str, None, LLMResponse]:
        """:meth:`LLMClient.stream` for ``model_id`` (``on_done=`` passes through)."""
        return self.get(model_id).stream(prompt, **kw)
