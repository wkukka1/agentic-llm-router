"""Pure filtering/extraction logic tested against small inline fixtures --
no live network calls (download_modern_pool_battles' own HF fetch is
exercised manually, not in CI, same discipline as public_benchmarks.py)."""

from __future__ import annotations

import pandas as pd

from training.data.modern_pool_battles import (
    filter_modern_pool_battles,
    query_text_from_conversation,
)


def _user_turn(text):
    return {"role": "user", "content": [{"type": "text", "text": text, "image": None}]}


def _asst_turn(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text, "image": None}]}


def test_query_text_from_conversation_finds_first_user_text():
    convo = [_user_turn("Compare Tormenta20 with DnD5e"), _asst_turn("...")]
    assert query_text_from_conversation(convo) == "Compare Tormenta20 with DnD5e"


def test_query_text_from_conversation_none_without_a_text_turn():
    convo = [{"role": "user", "content": [{"type": "image", "text": None, "image": "..."}]}]
    assert query_text_from_conversation(convo) is None


def test_filter_modern_pool_battles_keeps_only_both_sides_aliased():
    raw = pd.DataFrame([
        {"id": "1", "model_a": "deepseek-v3-0324", "model_b": "gemini-2.5-pro",
         "winner": "model_a", "conversation_a": [_user_turn("q1")]},
        {"id": "2", "model_a": "deepseek-v3-0324", "model_b": "gpt-4.1-2025-04-14",  # not aliased
         "winner": "model_a", "conversation_a": [_user_turn("q2")]},
        {"id": "3", "model_a": "qwen3-235b-a22b-instruct-2507", "model_b": "deepseek-r1-0528",
         "winner": "tie", "conversation_a": [_user_turn("q3")]},
    ])
    out = filter_modern_pool_battles(raw)
    assert set(out["query_id"]) == {"1", "3"}
    row1 = out[out["query_id"] == "1"].iloc[0]
    assert row1["model_a"] == "deepseek-v3-0324" and row1["model_b"] == "gemini-2.5-pro"
    assert row1["score_a"] == 1.0
    row3 = out[out["query_id"] == "3"].iloc[0]
    assert row3["model_a"] == "qwen3-235b-a22b-2507"  # aliased, not the raw dataset slug
    assert row3["score_a"] == 0.5


def test_filter_modern_pool_battles_drops_both_bad():
    raw = pd.DataFrame([
        {"id": "1", "model_a": "deepseek-v3-0324", "model_b": "gemini-2.5-pro",
         "winner": "both_bad", "conversation_a": [_user_turn("q1")]},
    ])
    assert filter_modern_pool_battles(raw).empty


def test_filter_modern_pool_battles_drops_rows_without_query_text():
    raw = pd.DataFrame([
        {"id": "1", "model_a": "deepseek-v3-0324", "model_b": "gemini-2.5-pro",
         "winner": "model_a",
         "conversation_a": [{"role": "user", "content": [{"type": "image", "text": None}]}]},
    ])
    assert filter_modern_pool_battles(raw).empty
