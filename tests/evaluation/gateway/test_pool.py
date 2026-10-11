"""The routing-data prompt pool: which LMArena rows become prompts, and how
the judge uses a reference solution and labels "matches the best"."""

from __future__ import annotations

import pandas as pd

from evaluation.gateway import pool
from evaluation.gateway.grade import judge_prompt, parse_verdicts


def msg(role, text, kind="text"):
    return {"role": role, "content": [{"type": kind, "text": text}]}


def test_arena_keeps_single_turn_english_text_coding_prompts_once(tmp_path, monkeypatch):
    good = "How do I reverse a linked list in Python without recursion?"
    rows = [
        (True, "en", [msg("user", good), msg("assistant", "...")], {"turns": 1}),
        (True, "en", [msg("user", good), msg("assistant", "...")], {"turns": 1}),       # duplicate vote
        (False, "en", [msg("user", "Write me a poem about the sea, please")], {"turns": 1}),  # not code
        (True, "de", [msg("user", "Wie sortiere ich eine Liste in Python?")], {"turns": 1}),
        (True, "en", [msg("user", "ok")], {"turns": 1}),                                   # too short
        (True, "en", [msg("user", good + " (two)"), msg("assistant", "a"), msg("user", "more")], {"turns": 2}),
        (True, "en", [msg("user", "What is in this screenshot of my code?", kind="image")], {"turns": 1}),
    ]
    path = tmp_path / "raw" / "arena-0.parquet"
    path.parent.mkdir()
    pd.DataFrame(rows, columns=["is_code", "language", "conversation_a", "conv_metadata"]).to_parquet(path)
    monkeypatch.setattr(pool, "_download", lambda url, dest: path)
    (row,) = pool.arena_rows(10, pool_dir=tmp_path)
    assert (row["prompt"], row["source"], row["group"], row["cwd"]) == (good, "arena", "arena", None)


def test_judge_gets_the_reference_and_reports_matches_best():
    text = judge_prompt("fix the bug", {"A": "patch a", "B": "patch b"}, reference="--- a/x.py\n+++ b/x.py")
    assert "=== Reference solution ===" in text and "maintainers" in text
    assert "Reference solution" not in judge_prompt("fix the bug", {"A": "x"})
    verdict = parse_verdicts('{"A": {"score": 9, "matches_best": true, "reason": "r"}, '
                             '"B": {"score": 4, "matches_best": "maybe", "reason": "q"}}', ["A", "B"])
    assert verdict == {"A": (0.9, "r", True), "B": (0.4, "q", None)}
