"""``LangChainAdapter`` (router/llm/adapters/langchain_adapter.py) -- the one
adapter that actually reaches a network today. A stub ``chat_model`` stands
in for the real LangChain model so these tests need no ``langchain`` install
and no network access."""

from __future__ import annotations

from router.llm.adapters.langchain_adapter import LangChainAdapter


class _StubChatModel:
    """Records the kwargs it was called with; returns a canned message/chunks."""

    def __init__(self):
        self.invoke_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def invoke(self, prompt, **kwargs):
        self.invoke_calls.append(kwargs)
        return type("Msg", (), {"content": "ok"})()

    def stream(self, prompt, **kwargs):
        self.stream_calls.append(kwargs)
        yield type("Msg", (), {"content": "a"})()
        yield type("Msg", (), {"content": "b"})()


def test_complete_forwards_generation_kwargs_to_invoke():
    chat = _StubChatModel()
    adapter = LangChainAdapter("gpt-4", chat_model=chat)

    adapter.complete("hi", temperature=0, max_tokens=200, stop=["\n"])

    assert chat.invoke_calls == [{"temperature": 0, "max_tokens": 200, "stop": ["\n"]}]


def test_stream_forwards_generation_kwargs_to_stream():
    chat = _StubChatModel()
    adapter = LangChainAdapter("gpt-4", chat_model=chat)

    list(adapter.stream("hi", temperature=0.5))

    assert chat.stream_calls == [{"temperature": 0.5}]


def test_complete_with_no_generation_kwargs_still_works():
    chat = _StubChatModel()
    adapter = LangChainAdapter("gpt-4", chat_model=chat)

    resp = adapter.complete("hi")

    assert resp.content == "ok"
    assert chat.invoke_calls == [{}]
