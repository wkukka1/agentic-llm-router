"""Regressions for known_bugs/LLMClient, LLMClientFactory, LLMResponse fixes."""

from __future__ import annotations

from router.llm.adapters.fakes import EchoAdapter
from router.llm.client import LLMClient, LLMClientFactory, LLMResponse
from router.llm.profile import LLMProfile


def _profile(**overrides) -> LLMProfile:
    kwargs = dict(name="m", provider="echo", model_id="m")
    kwargs.update(overrides)
    return LLMProfile(**kwargs)


def test_unpriced_profile_makes_price_unknown_not_zero():
    """LLMCLIENT-001: a profile whose prices were never set must price a call
    ``None`` (unknown), not ``0.0`` (indistinguishable from genuinely free)."""
    client = LLMClient(adapter=None, profile=_profile())
    assert client.price(1000, 1000) is None


def test_price_only_requires_the_side_actually_used():
    """A profile priced on only one side (e.g. output-only) must still price
    a call that uses only that side -- LLMCLIENTFACTORY-001's repro needs
    ``price(0, 1000)`` to resolve from ``output_cost_per_token`` alone."""
    client = LLMClient(adapter=None, profile=_profile(output_cost_per_token=5e-6))
    assert client.price(0, 1000) == 5e-6 * 1000
    assert client.price(1000, 0) is None  # needs the unset input price


def test_client_factory_serves_a_repriced_profile_with_fresh_prices():
    """LLMCLIENTFACTORY-001: the adapter/SDK client may be cached, but a
    later ``client_for`` call with a re-priced profile for the same
    provider/model must not keep the first profile's prices."""
    factory = LLMClientFactory(adapters={"echo": EchoAdapter})
    old = _profile(output_cost_per_token=1e-6)
    new = _profile(output_cost_per_token=5e-6)

    factory.client_for(old)
    client = factory.client_for(new)

    assert client.price(0, 1000) == 5e-6 * 1000


def test_llm_response_carries_raw_and_meta():
    """LLMRESPONSE-001: ``raw``/``meta`` fields dropped during the migration
    from ``agentic.llm_clients.LLMResponse`` must exist again."""
    resp = LLMResponse(content="ok", raw={"finish_reason": "stop"}, meta={"provider": "echo"})
    assert resp.raw == {"finish_reason": "stop"}
    assert resp.meta == {"provider": "echo"}
    assert LLMResponse(content="ok").raw is None
    assert LLMResponse(content="ok").meta is None
