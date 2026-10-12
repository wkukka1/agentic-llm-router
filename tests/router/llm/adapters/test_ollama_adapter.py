"""``OllamaAdapter`` (router/llm/adapters/ollama.py). A stub stands in for
``ollama.Client`` (passed as ``client=``), so these need no local Ollama
server and no network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from router.llm.adapters.ollama import OllamaAdapter


def _response(content="hello", *, prompt_eval_count=10, eval_count=5,
              done_reason="stop", total_duration=123):
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
        done_reason=done_reason,
        total_duration=total_duration,
    )


def _chunk(text, *, done=False, **kw):
    ns = SimpleNamespace(message=SimpleNamespace(content=text), done=done)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class _StubClient:
    def __init__(self, response=None, chunks=None):
        self.requests: list[dict] = []
        self._response = response or _response()
        self._chunks = chunks if chunks is not None else [
            _chunk("hel"), _chunk("lo"),
            _chunk("", done=True, prompt_eval_count=10, eval_count=5,
                   done_reason="stop", total_duration=123),
        ]

    def chat(self, **kw):
        self.requests.append(kw)
        if kw.get("stream"):
            return iter(self._chunks)
        return self._response


def test_complete_builds_request_and_reads_usage():
    stub = _StubClient()
    resp = OllamaAdapter("qwen2.5:3b", client=stub).complete("hi")
    req = stub.requests[0]
    assert req["model"] == "qwen2.5:3b"
    assert req["messages"] == [{"role": "user", "content": "hi"}]
    assert "options" not in req
    assert resp.content == "hello"
    assert (resp.input_tokens, resp.output_tokens) == (10, 5)
    assert resp.cost == 0.0                               # local inference: known, not unknown
    assert resp.meta["done_reason"] == "stop"


def test_model_kwarg_overrides_bound_default():
    stub = _StubClient()
    OllamaAdapter("default-model", client=stub).complete("hi", model="qwen2.5:3b")
    assert stub.requests[0]["model"] == "qwen2.5:3b"


def test_bare_adapter_needs_model_per_call():
    stub = _StubClient()
    resp = OllamaAdapter(client=stub).complete("hi", model="phi3.5")
    assert stub.requests[0]["model"] == "phi3.5"
    assert resp.content == "hello"


def test_missing_model_is_an_error():
    stub = _StubClient()
    with pytest.raises(ValueError):
        OllamaAdapter(client=stub).complete("hi")


def test_generation_kwargs_map_into_options():
    stub = _StubClient()
    OllamaAdapter("m", client=stub).complete("hi", max_tokens=32, temperature=0.0)
    assert stub.requests[0]["options"] == {"num_predict": 32, "temperature": 0.0}


def test_format_and_tools_stay_top_level_not_in_options():
    stub = _StubClient()
    schema = {"type": "object"}
    tools = [{"type": "function", "function": {"name": "f"}}]
    OllamaAdapter("m", client=stub).complete("hi", format=schema, tools=tools)
    req = stub.requests[0]
    assert req["format"] == schema and req["tools"] == tools
    assert "options" not in req


def test_chat_list_prompt_keeps_every_role_including_system():
    stub = _StubClient()
    prompt = [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]
    OllamaAdapter("m", client=stub).complete(prompt)
    assert stub.requests[0]["messages"] == prompt


def test_stream_yields_deltas_and_final_response_has_joined_content():
    stub = _StubClient()
    gen = OllamaAdapter("m", client=stub).stream("hi")
    chunks = []
    try:
        while True:
            chunks.append(next(gen))
    except StopIteration as stop:
        resp = stop.value
    assert chunks == ["hel", "lo"]
    assert resp.content == "hello"
    assert (resp.input_tokens, resp.output_tokens) == (10, 5)
    assert resp.cost == 0.0
