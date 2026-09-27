"""``LangChainAdapter`` (router/llm/adapters/langchain_adapter.py) -- the one
adapter that actually reaches a network today. A stub ``chat_model`` stands
in for the real LangChain model so these tests need no ``langchain`` install
and no network access -- except the two ``init_chat_model`` tests, which patch
``langchain.chat_models`` and skip when ``langchain`` is not installed."""

from __future__ import annotations

import pytest

from router.llm.adapters.langchain_adapter import LangChainAdapter, guess_provider


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


def test_guess_provider_requires_a_right_hand_boundary_too():
    """MOD-001 (router.llm.adapters): a hint matching mid-token (inside a
    longer alnum run) must not count as a match -- only RA-19's existing
    left-boundary guard existed before this fix."""
    assert guess_provider("gpt4all-13b-snoozy") is None
    # existing passing cases must still match -- a boundary on the right,
    # not "no match at all"
    assert guess_provider("o1-mini") == "openai"
    assert guess_provider("gpt-4o") == "openai"
    assert guess_provider("yolo1") is None
    assert guess_provider("claude-3-5-sonnet") == "anthropic"
    assert guess_provider("some-random-model") is None


def test_guess_provider_allows_a_trailing_version_number_after_the_hint():
    """A hint may be followed by a version number (llama3, qwen2.5, gemini15)
    -- real configured ids -- but not by more letters (gpt4all)."""
    assert guess_provider("gpt4all-13b-snoozy") is None
    assert guess_provider("llama31_8b_instruct") == "groq"
    assert guess_provider("llama2-70b-steerlm-chat") == "groq"
    assert guess_provider("qwen1.5-7b-chat") == "together"
    assert guess_provider("qwen2.5-0.5b-instruct") == "together"
    assert guess_provider("qwen2.5-7b-instruct") == "together"
    assert guess_provider("qwen25_7b_instruct") == "together"
    assert guess_provider("qwen3-235b-a22b-2507") == "together"
    assert guess_provider("gemini15_flash") == "google_genai"
    # previously pinned cases still hold
    assert guess_provider("o1-mini") == "openai"
    assert guess_provider("gpt-4o") == "openai"
    assert guess_provider("yolo1") is None
    assert guess_provider("claude-3-5-sonnet") == "anthropic"
    assert guess_provider("some-random-model") is None


def test_model_forwards_api_key_to_init_chat_model(monkeypatch):
    """LANGCHAINADAPTER-001: an explicit api_key must reach init_chat_model,
    not be silently dropped in favour of ambient credentials."""
    pytest.importorskip("langchain")
    import langchain.chat_models

    captured = {}

    def fake_init_chat_model(name, *, model_provider=None, **kw):
        captured["name"] = name
        captured["model_provider"] = model_provider
        captured.update(kw)
        return _StubChatModel()

    monkeypatch.setattr(langchain.chat_models, "init_chat_model", fake_init_chat_model)

    adapter = LangChainAdapter("gpt-4", provider="openai", api_key="sk-test")
    adapter._model()

    assert captured["api_key"] == "sk-test"


def test_model_omits_api_key_when_not_given(monkeypatch):
    """No api_key= given -- must not pass a spurious api_key=None that could
    shadow init_chat_model's own default resolution."""
    pytest.importorskip("langchain")
    import langchain.chat_models

    captured = {}

    def fake_init_chat_model(name, *, model_provider=None, **kw):
        captured.update(kw)
        return _StubChatModel()

    monkeypatch.setattr(langchain.chat_models, "init_chat_model", fake_init_chat_model)

    adapter = LangChainAdapter("gpt-4", provider="openai")
    adapter._model()

    assert "api_key" not in captured


class _PricedChatModel(_StubChatModel):
    """Like _StubChatModel, but its response carries usage_metadata."""

    def invoke(self, prompt, **kwargs):
        msg = super().invoke(prompt, **kwargs)
        msg.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
        return msg


def test_complete_prices_nothing_when_only_one_side_is_configured():
    """LANGCHAINADAPTER-002: a half-priced adapter (only input_cost_per_token
    given) must not silently price the unset output side at $0."""
    chat = _PricedChatModel()
    adapter = LangChainAdapter("gpt-4", chat_model=chat, input_cost_per_token=1e-6)

    resp = adapter.complete("hi")

    assert resp.cost is None


def test_complete_prices_normally_when_both_sides_are_configured():
    chat = _PricedChatModel()
    adapter = LangChainAdapter(
        "gpt-4", chat_model=chat, input_cost_per_token=1e-6, output_cost_per_token=2e-6,
    )

    resp = adapter.complete("hi")

    assert resp.cost == pytest.approx(100 * 1e-6 + 50 * 2e-6)
