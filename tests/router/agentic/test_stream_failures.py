"""stream() has the same failure semantics as run(): a blank answer is an error; a failing on_result is logged."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from router.agentic import AgenticRouter, ClientRegistry, LLMClient, LLMResponse
from router.agentic.router import flush_bookkeeping
from router.llm.adapters.base import ProviderAdapter
from router.routing import RouterModel

POOL = ["small", "big"]


class _Router(RouterModel):
    kind = "fake"
    can_route_text = True

    def __init__(self):
        super().__init__(POOL, name="fake")

    def predict_scores(self, query_ids):
        return pd.DataFrame([[0.5, 0.5] for _ in query_ids], index=[str(q) for q in query_ids], columns=POOL)

    def predict_scores_text(self, texts, *, encoder=None):
        return pd.DataFrame([[0.9, 0.9] for _ in texts], index=list(range(len(texts))), columns=POOL)


def _clients(text: str) -> ClientRegistry:
    class _Words(ProviderAdapter):
        def complete(self, prompt, **kw):
            return LLMResponse(content=text)

        def stream(self, prompt, **kw):
            yield from text.split()
            return LLMResponse(content=text, cost=0.1, input_tokens=1, output_tokens=1)

    return ClientRegistry({m: LLMClient(_Words(m)) for m in POOL})


def _drain(gen):
    chunks = []
    try:
        while True:
            chunks.append(next(gen))
    except StopIteration as stop:
        return chunks, stop.value


def test_a_blank_streamed_answer_is_an_error_like_a_blank_run_answer():
    agent = AgenticRouter(_Router(), _clients("  "))
    with pytest.raises(RuntimeError, match="no answer"):
        agent.run("What is 2+2?")
    with pytest.raises(RuntimeError, match="no answer"):
        _drain(agent.stream("What is 2+2?"))


def test_a_failing_stream_on_result_is_logged_not_lost(caplog):
    def boom(result):
        raise ValueError("sink down")

    agent = AgenticRouter(_Router(), _clients("four"))
    with caplog.at_level(logging.ERROR):
        _drain(agent.stream("What is 2+2?", on_result=boom))
        flush_bookkeeping(timeout=30)
    assert "result sink failed" in caplog.text
