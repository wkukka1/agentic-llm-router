"""``ClientRegistry`` (router/agentic/llm_clients.py) factory methods. A stub
stands in for ``ollama.Client`` (passed as ``client=``), so these need no
local Ollama server and no network."""

from __future__ import annotations

from router.agentic.llm_clients import ClientRegistry
from router.llm.adapters.ollama import OllamaAdapter


class _StubOllamaClient:
    """Never actually called here -- only identity matters: every registered
    adapter must share this one object."""


def test_ollama_registers_each_id_sharing_one_client():
    stub = _StubOllamaClient()
    registry = ClientRegistry.ollama(["qwen2.5:3b", "phi3.5"], client=stub)

    for model_id in ("qwen2.5:3b", "phi3.5"):
        assert model_id in registry
        adapter = registry.get(model_id)._adapter
        assert isinstance(adapter, OllamaAdapter)
        assert adapter.model == model_id
        assert adapter._client is stub


def test_ollama_default_factory_serves_an_unregistered_tag_on_the_same_client():
    stub = _StubOllamaClient()
    registry = ClientRegistry.ollama(["qwen2.5:3b"], client=stub)

    unseen = registry.get("smollm2:1.7b")._adapter
    assert isinstance(unseen, OllamaAdapter)
    assert unseen.model == "smollm2:1.7b"
    assert unseen._client is stub
    assert "smollm2:1.7b" in registry.synthesized
