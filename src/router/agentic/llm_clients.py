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

:class:`ClientRegistry` maps ``model_id -> LLMClient`` and synthesises a
default client for ids it has never seen, so an unfamiliar candidate pool
never crashes the orchestrator. When the registry was given real clients,
synthesising an echo client for a missing id warns (its answers are
placeholders).

``guess_provider``/``message_text`` now live in
:mod:`router.llm.adapters.langchain_adapter`; re-exported here for
:mod:`router.agentic.triage`/:mod:`router.agentic.decompose`, which only ever
needed ``message_text`` to flatten a LangChain message.
"""

from __future__ import annotations

import warnings
from typing import Callable, Mapping, Optional

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

    def register(self, model_id: str, client: LLMClient) -> None:
        self._clients[str(model_id)] = client

    def get(self, model_id: str) -> LLMClient:
        mid = str(model_id)
        if mid not in self._clients:
            client = self._default_factory(mid)
            if self._default_factory is _echo_factory and self._has_real_clients:
                warnings.warn(
                    f"no client registered for model {mid!r}; answering with an echo "
                    "placeholder",
                    stacklevel=2,
                )
            self._clients[mid] = client
            self.synthesized.add(mid)
        return self._clients[mid]

    def __contains__(self, model_id: str) -> bool:
        return str(model_id) in self._clients

    def complete(self, model_id: str, prompt: str, **kw) -> LLMResponse:
        return self.get(model_id).complete(prompt, **kw)
