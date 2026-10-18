"""Regressions for known_bugs/LLMClient, LLMResponse fixes."""

from __future__ import annotations

from router.llm.adapters.fakes import EchoAdapter
from router.llm.client import LLMClient, LLMResponse
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


def test_llm_response_carries_raw_and_meta():
    """LLMRESPONSE-001: ``raw``/``meta`` fields dropped during the migration
    from ``agentic.llm_clients.LLMResponse`` must exist again."""
    resp = LLMResponse(content="ok", raw={"finish_reason": "stop"}, meta={"provider": "echo"})
    assert resp.raw == {"finish_reason": "stop"}
    assert resp.meta == {"provider": "echo"}
    assert LLMResponse(content="ok").raw is None
    assert LLMResponse(content="ok").meta is None


def _drain(gen):
    chunks = []
    try:
        while True:
            chunks.append(next(gen))
    except StopIteration as stop:
        return chunks, stop.value


def test_stream_returns_a_timed_priced_response_and_calls_on_done():
    """Adapters whose ``stream`` returns nothing (echo, LangChain) still get a
    final response built from the chunks, with time-to-first-token."""
    done = []
    client = LLMClient(EchoAdapter("m"))
    chunks, resp = _drain(client.stream("hi", on_done=done.append))
    assert chunks and resp.content == "".join(chunks)
    assert resp.latency is not None and resp.meta["ttft"] <= resp.latency
    assert done == [resp]


def test_stream_prices_the_adapters_final_usage_from_the_profile():
    from router.llm.adapters.base import ProviderAdapter

    class _Adapter(ProviderAdapter):
        def complete(self, prompt, **kw):  # pragma: no cover - unused
            raise NotImplementedError

        def stream(self, prompt, **kw):
            yield "a"
            yield "b"
            return LLMResponse(content="ab", input_tokens=10, output_tokens=2)

    profile = LLMProfile(name="m", provider="x", model_id="m",
                         input_cost_per_token=1.0, output_cost_per_token=10.0)
    chunks, resp = _drain(LLMClient(_Adapter("m"), profile).stream("hi"))
    assert chunks == ["a", "b"] and resp.cost == 30.0 and "ttft" in resp.meta
