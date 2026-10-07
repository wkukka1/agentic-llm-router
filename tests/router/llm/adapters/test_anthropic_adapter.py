"""``AnthropicAdapter`` (router/llm/adapters/anthropic.py). A stub stands in for
``anthropic.Anthropic`` (passed as ``client=``), so these need no API key and no
network; the one live test is opt-in (``ANTHROPIC_LIVE_TEST=1``)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from router.llm.adapters.anthropic import AnthropicAdapter
from router.llm.client import LLMClient


def _message(text="hello", *, stop_reason="end_turn", input_tokens=10, output_tokens=5,
             cache_read=0, cache_write=0):
    msg = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                              cache_read_input_tokens=cache_read,
                              cache_creation_input_tokens=cache_write),
    )
    msg._request_id = "req_123"
    return msg


class _Stream:
    def __init__(self, chunks, final):
        self.text_stream = iter(chunks)
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


class _StubClient:
    def __init__(self, message=None, chunks=("hel", "lo")):
        self.requests: list[dict] = []
        self._message = message or _message()
        self._chunks = chunks
        self.messages = SimpleNamespace(create=self._create, stream=self._stream)

    def _create(self, **kw):
        self.requests.append(kw)
        return self._message

    def _stream(self, **kw):
        self.requests.append(kw)
        return _Stream(list(self._chunks), self._message)


def test_complete_builds_request_and_reads_usage():
    stub = _StubClient()
    resp = AnthropicAdapter("claude-haiku-4-5", client=stub, max_tokens=256).complete("hi")
    req = stub.requests[0]
    assert req["model"] == "claude-haiku-4-5" and req["max_tokens"] == 256
    # no system prompt: cache_control marks the first (only) message's block --
    # not a top-level request key, which the SDK doesn't recognize at all
    assert req["messages"] == [{"role": "user", "content": [
        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
    ]}]
    assert "cache_control" not in req
    assert "system" not in req
    assert resp.content == "hello"                      # text blocks only
    assert (resp.input_tokens, resp.output_tokens) == (10, 5)
    assert resp.meta["stop_reason"] == "end_turn" and resp.meta["request_id"] == "req_123"


def test_generation_max_tokens_overrides_client_default():
    stub = _StubClient()
    AnthropicAdapter("m", client=stub, max_tokens=256).complete("hi", max_tokens=32, temperature=0.0)
    assert stub.requests[0]["max_tokens"] == 32 and stub.requests[0]["temperature"] == 0.0


def test_missing_max_tokens_is_an_error():
    with pytest.raises(ValueError, match="max_tokens"):
        AnthropicAdapter("m", client=_StubClient()).complete("hi")


def test_leading_system_messages_are_lifted():
    stub = _StubClient()
    msgs = [{"role": "system", "content": "be terse"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"}]
    AnthropicAdapter("m", client=stub, max_tokens=8).complete(msgs)
    req = stub.requests[0]
    # the system prompt is the stable, top-of-request block -- it gets the
    # cache breakpoint, not the (still plain-string) messages
    assert req["system"] == [{"type": "text", "text": "be terse",
                              "cache_control": {"type": "ephemeral"}}]
    assert [m["role"] for m in req["messages"]] == ["user", "assistant", "user"]
    assert req["messages"][0]["content"] == "a"


def test_cache_can_be_disabled():
    stub = _StubClient()
    AnthropicAdapter("m", client=stub, max_tokens=8, cache=False).complete("hi")
    req = stub.requests[0]
    assert req["messages"] == [{"role": "user", "content": "hi"}]
    assert "cache_control" not in req


def test_cache_disabled_leaves_system_as_a_plain_string():
    stub = _StubClient()
    msgs = [{"role": "system", "content": "be terse"}, {"role": "user", "content": "a"}]
    AnthropicAdapter("m", client=stub, max_tokens=8, cache=False).complete(msgs)
    assert stub.requests[0]["system"] == "be terse"


def test_refusal_raises():
    stub = _StubClient(_message("", stop_reason="refusal"))
    with pytest.raises(RuntimeError, match="refus"):
        AnthropicAdapter("m", client=stub, max_tokens=8).complete("hi")


def test_cost_uses_cache_aware_prices_when_given():
    stub = _StubClient(_message(input_tokens=100, output_tokens=10, cache_read=1000, cache_write=200))
    resp = AnthropicAdapter("m", client=stub, max_tokens=8,
                            input_cost_per_token=1e-6, output_cost_per_token=5e-6).complete("hi")
    # uncached 100 @1x + reads 1000 @0.1x + writes 200 @1.25x, + 10 out @5e-6
    assert resp.cost == pytest.approx(100e-6 + 1000 * 0.1e-6 + 200 * 1.25e-6 + 10 * 5e-6)
    assert resp.input_tokens == 1300                       # every prompt token, cached or not
    assert resp.meta["cache_read_input_tokens"] == 1000
    assert resp.meta["cache_creation_input_tokens"] == 200


def test_without_prices_cost_is_left_to_the_client():
    stub = _StubClient()
    assert AnthropicAdapter("m", client=stub, max_tokens=8).complete("hi").cost is None


def test_stream_yields_text_and_returns_the_final_response():
    stub = _StubClient()
    gen = AnthropicAdapter("m", client=stub, max_tokens=8).stream("hi")
    chunks = []
    with pytest.raises(StopIteration) as stop:
        while True:
            chunks.append(next(gen))
    assert chunks == ["hel", "lo"]
    assert stop.value.value.content == "hello" and stop.value.value.output_tokens == 5


def test_llm_client_complete_times_the_call():
    client = LLMClient(AnthropicAdapter("m", client=_StubClient(), max_tokens=8))
    resp = client.complete("hi")
    assert resp.latency is not None and resp.latency >= 0


@pytest.mark.skipif(os.environ.get("ANTHROPIC_LIVE_TEST") != "1",
                    reason="live API test; set ANTHROPIC_LIVE_TEST=1 (and credentials) to run")
def test_live_haiku_round_trip():
    resp = LLMClient(AnthropicAdapter("claude-haiku-4-5", max_tokens=32)).complete(
        "Reply with the single word: pong")
    assert "pong" in resp.content.lower()
    assert resp.input_tokens and resp.output_tokens
