"""The ops agent's read-only query script: correct numbers from a known log."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".claude/skills/aar-gateway-ops/scripts/query.py"


@pytest.fixture
def query(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("query", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    old = (datetime.now(UTC) - timedelta(days=30)).isoformat(timespec="seconds")
    base = {"claude_session": "abc123def456", "requested_model": "claude-opus-5-5", "routed_model": "claude-opus-5-5",
            "status": 200, "cost_usd": 1.0, "cost_if_requested_usd": 2.0, "cost_if_strongest_usd": 2.0}
    rows = [
        {**base, "ts": now},
        {**base, "ts": now, "routed_model": "claude-haiku-5-5", "cost_usd": 0.2, "cost_if_requested_usd": 2.0,
         "cost_if_strongest_usd": 2.0, "subagent": True, "cache_creation_input_tokens": 500},
        {**base, "ts": now, "status": 404, "cost_usd": None, "cost_if_requested_usd": None, "cost_if_strongest_usd": None,
         "error": "not_found_error: No thread state was found", "thread": "continue"},
        {**base, "ts": now, "status": 400, "cost_usd": None, "cost_if_requested_usd": None, "cost_if_strongest_usd": None,
         "error": "invalid_request_error: thread: differing field(s): model", "thread": "continue"},
        {**base, "ts": old},
    ]
    log = tmp_path / "requests.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(log))
    return mod


def test_savings_are_against_the_model_asked_for(query):
    s = query.savings(query.load(7))
    assert s["vs_no_router"]["usage_usd"] == 1.2 and s["vs_no_router"]["baseline_usd"] == 4.0
    assert s["vs_no_router"]["saved_pct"] == 70.0
    assert s["subagents"]["saved_pct"] == 90.0


def test_expired_threads_are_not_failures_but_mismatches_are(query):
    e = query.errors(query.load(7))
    assert e["expired_threads_not_counted"] == 1
    assert e["failed"] == 1 and e["by_error_type"] == {"invalid_request_error": 1}


def test_the_window_drops_old_requests(query):
    assert query.savings(query.load(7))["requests"] == 4
    assert query.savings(query.load(None))["requests"] == 5


def test_session_matches_by_prefix(query):
    s = query.session(query.load(None), "abc123")
    assert s["requests"] == 5 and s["session"] == "abc123"


def test_the_script_never_reads_prompts_or_writes_files(query):
    import ast

    tree = ast.parse(SCRIPT.read_text())
    tree.body = tree.body[1:]  # the module docstring describes the rules; the code must obey them
    code = " ".join(ast.dump(n) for n in ast.walk(tree))
    assert "turns.jsonl" not in code  # no prompt text, not even by name
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("write_text", "unlink", "replace", "write")
                   for n in ast.walk(tree))


def test_the_contracts_name_the_installed_command_and_forbid_chaining():
    root = SCRIPT.parents[1]  # .claude/skills/aar-gateway-ops
    skill = (root / "SKILL.md").read_text()
    contracts = (root / "reference" / "contracts.md").read_text()
    assert ".local/share/uv/tools/agentic-llm-router/bin/aar-gateway" in skill
    assert "One plain command per Bash call" in skill and "`;`" in contracts
