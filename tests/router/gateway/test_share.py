"""Opt-in usage sharing: off by default, nothing identifying in a record,
and never in the way of a request."""

from __future__ import annotations

import json

import pytest

from router.gateway import cli, share


@pytest.fixture(autouse=True)
def fresh_consent(monkeypatch):
    monkeypatch.setattr(share, "_consent", {"mtime": None, "on": False})
    monkeypatch.setattr(share, "_embed", lambda text: [1] * share.DIMS)  # no encoder download in tests


LOGGED = {"ts": "2026-10-17T13:45:12+00:00", "claude_session": "abc123", "conversation": "8885ac1c",
          "requested_model": "claude-opus-5-5", "routed_model": "claude-haiku-5-5",
          "reason": "cheapest within 0.05 utility of the best", "status": 400, "subagent": False,
          "error": "invalid_request_error: messages.3: the secret token sk-ant-xyz is wrong",
          "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01, "cost_if_requested_usd": 0.05,
          "correction": True, "thread": "continue", "latency_s": 1.2}


def test_nothing_is_kept_unless_the_user_opts_in():
    assert not share.enabled()
    share.record(LOGGED, "fix the bug in /Users/me/secret-project/app.py")
    share.flush(5)
    assert share.queued() == []


def test_a_record_holds_no_text_ids_paths_or_exact_times():
    share.set_enabled(True)
    prompt = "fix the bug in /Users/me/secret-project/app.py, my key is sk-ant-xyz"
    share.record(LOGGED, prompt)
    share.flush(5)
    (rec,) = share.queued()
    text = json.dumps(rec)
    for leak in ("secret-project", "sk-ant-xyz", "abc123", "8885ac1c", "13:45", "Users"):
        assert leak not in text, leak
    assert rec["day"] == "2026-10-17" and rec["error_type"] == "invalid_request_error"
    assert rec["decision"] == "router" and rec["correction"] is True and rec["thread"] == "continue"
    assert rec["prompt_chars"] == 64 and len(rec["embedding"]) == share.DIMS


def test_subagents_and_tool_steps_add_nothing():
    share.set_enabled(True)
    share.record({**LOGGED, "subagent": True}, "search for x")
    share.record(LOGGED, None)  # a tool-loop step: no typed message
    share.flush(5)
    assert share.queued() == []


def test_the_compressed_embedding_is_small_integers_from_a_fixed_projection():
    import numpy as np

    v = np.random.default_rng(0).normal(size=384)
    a, b = share._compress(v), share._compress(v)
    assert a == b and len(a) == 64 and all(-127 <= x <= 127 for x in a)


def test_the_cli_turns_it_on_and_off_and_deletes(capsys):
    assert cli.main(["share"]) == 0 and "sharing      off" in capsys.readouterr().out
    cli.main(["share", "on"])
    assert share.enabled()
    share.record(LOGGED, "hello there")
    share.flush(5)
    cli.main(["share", "show"])
    assert "1 record(s)" in capsys.readouterr().out
    cli.main(["share", "off"])
    assert not share.enabled() and len(share.queued()) == 1  # kept until deleted
    cli.main(["share", "delete"])
    assert share.queued() == []


def test_setup_asks_once_and_only_a_yes_opts_in(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_doctor", lambda: 0)
    monkeypatch.setattr(cli, "_healthy", lambda url: False)
    monkeypatch.setattr(cli, "_fetch_router", lambda source, target: None)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "")
    cli.main(["setup"])
    assert any("Share anonymous usage data" in q for q in asked) and not share.enabled()
    asked.clear()
    cli.main(["setup"])
    assert not any("Share anonymous" in q for q in asked)  # asked once: the answer is remembered
    cli.main(["setup", "--share"])
    assert share.enabled()
