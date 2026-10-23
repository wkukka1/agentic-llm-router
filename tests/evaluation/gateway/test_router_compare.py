"""The router bake-off harness without any router: the recorder stub's replies and call log,
attributing one request's upstream calls, requests built from a template, and errors staying
in the denominator."""

from __future__ import annotations

import json

import numpy as np
import pytest
from starlette.testclient import TestClient

from evaluation.gateway import pool_eval as pe
from evaluation.gateway import router_compare as rc

TEMPLATE = {"body": {"model": "claude-opus-5-5", "stream": True, "max_tokens": 32000,
                     "thinking": {"type": "enabled", "budget_tokens": 31999},
                     "system": [{"type": "text", "text": "You are Claude Code."}],
                     "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
                     "metadata": {"user_id": "user_abc_account_def_session_1111"},
                     "messages": [{"role": "user", "content": [
                         {"type": "text", "text": "<system-reminder>ctx</system-reminder>"},
                         {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]},
                         {"role": "system", "content": [{"type": "text", "text": "deferred tools"}]}]},
            "headers": {"anthropic-beta": "claude-code-20250219", "user-agent": "claude-cli/9.9"}}


@pytest.fixture
def stub():
    rc._calls.clear()
    with TestClient(rc.recorder_app()) as client:
        yield client
    rc._calls.clear()


def test_anthropic_stream_and_json_replies_are_readable_and_logged_without_prompt_text(stub):
    body = rc.build_request(TEMPLATE, "SECRET PROMPT", model="claude-haiku-5-5", bid="0123456789abcdef")
    r = stub.post("/v1/messages", content=json.dumps(body))
    assert rc.reply_ok("anthropic", r.status_code, r.text) is None
    r = stub.post("/v1/messages", content=json.dumps({**body, "stream": False}))
    assert r.json()["model"] == "claude-haiku-5-5" and rc.reply_ok("anthropic", 200, r.text) is None
    calls = rc.calls_since(0)
    assert [c["model"] for c in calls] == ["claude-haiku-5-5"] * 2
    assert all(c["bench_id"] == "0123456789abcdef" for c in calls)
    assert "SECRET" not in json.dumps(calls)


def test_openai_replies_and_count_tokens(stub):
    body = rc.openai_request("q", model="claude-sonnet-5-5", bid="fedcba9876543210")
    r = stub.post("/v1/chat/completions", content=json.dumps(body))
    assert rc.reply_ok("openai", r.status_code, r.text) is None
    r = stub.post("/v1/chat/completions", content=json.dumps({**body, "stream": True}))
    assert rc.reply_ok("openai", r.status_code, r.text) is None
    r = stub.post("/v1/messages/count_tokens", content=json.dumps(body))
    assert r.json()["input_tokens"] >= 1
    assert [c["kind"] for c in rc.calls_since(0)] == ["openai", "openai", "count_tokens"]


def test_attribution_one_call_is_a_pick_none_or_many_is_an_anomaly():
    one = [{"endpoint": "/v1/messages", "model": "claude-sonnet-5-5", "kind": "anthropic", "bench_id": "x"},
           {"endpoint": "/v1/messages/count_tokens", "model": "claude-opus-5-5", "kind": "count_tokens"}]
    got = rc.attribute(one, "x")
    assert got["tier"] == "claude-sonnet-5-5" and got["error"] is None and got["mapping_reason"] == "bench id matched"
    assert rc.attribute([], "x")["error"] == "no_upstream_call"
    two = [one[0], {**one[0], "model": "claude-opus-5-5"}]
    got = rc.attribute(two, "x")
    assert got["error"] == "multiple_upstream_calls" and got["tier"] is None


def test_unknown_models_fail_closed():
    for model in ("gpt-4.1", "claude-opus-5", "openai/claude-opus-5-5", ""):
        got = rc.attribute([{"endpoint": "/v1/messages", "model": model, "kind": "anthropic"}], "x")
        assert got["error"] == "unknown_model" and got["tier"] is None
    assert rc.attribute([{"endpoint": "/v1/messages", "model": "anthropic/claude-opus-5-5",
                          "kind": "anthropic", "bench_id": None}], "x")["tier"] == "claude-opus-5-5"


def test_build_request_swaps_the_prompt_and_isolates_each_request():
    a = rc.build_request(TEMPLATE, "explain X", model="claude-sonnet-5-5", bid="aaaaaaaaaaaaaaaa")
    b = rc.build_request(TEMPLATE, "explain X", model="claude-sonnet-5-5", bid="bbbbbbbbbbbbbbbb", thinking=False)
    texts = [blk["text"] for blk in a["messages"][0]["content"]]
    assert texts == ["<system-reminder>ctx</system-reminder>", "explain X"]
    assert [m["role"] for m in a["messages"]] == ["user", "system"]  # the trailing system message is kept
    assert a["model"] == "claude-sonnet-5-5" and "thinking" in a and "thinking" not in b
    assert a["system"][-1]["text"] == "aar-bench-id: aaaaaaaaaaaaaaaa"
    assert a["system"][0]["text"] == "You are Claude Code."  # CCR reads system[1] for its subagent tag
    assert a["metadata"]["user_id"] != b["metadata"]["user_id"]
    assert TEMPLATE["body"]["messages"][0]["content"][1]["text"] == "hi"  # the template is untouched
    headers = rc.request_headers(TEMPLATE, "aaaaaaaaaaaaaaaa")
    assert headers["anthropic-beta"] == "claude-code-20250219" and headers["x-api-key"] == "aar-bench-dummy"


def test_a_template_without_the_placeholder_is_refused():
    broken = {"body": {**TEMPLATE["body"], "messages": [{"role": "user", "content": "hello"}]}}
    with pytest.raises(ValueError):
        rc.build_request(broken, "p", model="claude-opus-5-5", bid="aaaaaaaaaaaaaaaa")


def _prompts():
    return [{"prompt_hash": f"p{i}", "prompt": f"q{i}", "source": "own",
             "score": np.array(s, float), "cost": np.array([1.0, 5.0, 10.0]),
             "matches": np.array([x >= max(s) for x in s])}
            for i, s in enumerate([(0.9, 0.9, 0.9), (0.2, 0.9, 0.9), (0.2, 0.3, 0.9), (0.9, 0.9, 0.9)])]


def test_errors_stay_in_the_denominator_charged_as_opus():
    prompts = _prompts()
    rows = [{"prompt_hash": "p0", "tier": "claude-haiku-5-5", "error": None},
            {"prompt_hash": "p1", "tier": None, "error": "unknown_model"},
            {"prompt_hash": "p2", "tier": "claude-opus-5-5", "error": None}]  # p3 missing entirely
    picks, failed = rc.picks_of(rows, prompts)
    assert list(picks) == [0, 2, 2, 2] and list(failed) == [False, True, False, True]
    (row,) = rc.score_rows(prompts, {"r": rows})
    assert row["n_prompts"] == 4 and row["n_errors"] == 2 and row["unknown_models"] == 1
    assert sum(row["mix"].values()) == 4
    assert row["actual_opus_share"] == 0.25  # errors are not counted as Opus picks


def test_a_router_that_always_forwards_opus_matches_always_opus():
    prompts = _prompts()
    rows = [{"prompt_hash": p["prompt_hash"], "tier": "claude-opus-5-5", "error": None} for p in prompts]
    mine = rc.score_rows(prompts, {"opus forwarder": rows})[0]
    ref = {r["policy"]: r for r in pe.evaluate(prompts, picks=pe.policies(prompts, margin=0.0, k=2))}
    assert mine["quality"] == ref["always opus"]["quality"]
    assert mine["est_cost_usd"] == ref["always opus"]["usage_usd"]


def test_paired_bootstrap_sees_no_difference_between_identical_policies():
    prompts = _prompts()
    same = np.array([2, 2, 2, 2])
    got = rc.paired_bootstrap(prompts, same, same, n=200)
    assert got["quality"]["delta"] == 0.0 and got["quality"]["verdict"] == "no detectable difference"
    cheaper = rc.paired_bootstrap(prompts, np.array([0, 0, 0, 0]), same, n=200)
    assert cheaper["est_cost_usd"]["hi"] < 0


def test_bench_traffic_reaching_the_live_gateway_is_caught_but_the_users_own_is_not(tmp_path, monkeypatch):
    from evaluation.gateway import router_bench as rb

    monkeypatch.setattr(rb.Path, "home", classmethod(lambda cls: tmp_path))
    live = tmp_path / ".aar-gateway"
    live.mkdir()
    (live / "requests.jsonl").write_text('{"claude_session": "old"}\n')
    before = rb._live_sizes()
    rows = [{"bench_id": "0123456789abcdef"}]
    with open(live / "requests.jsonl", "a") as fh:
        fh.write('{"claude_session": "users-own-1"}\n')
    assert rb._bench_traffic_in_live(before, rows) == []
    with open(live / "requests.jsonl", "a") as fh:
        fh.write(json.dumps({"claude_session": rc._session_uuid("0123456789abcdef")[:12]}) + "\n")
    assert rb._bench_traffic_in_live(before, rows) == [str(live / "requests.jsonl")]


def test_savings_are_measured_against_the_model_asked_for():
    prompts = _prompts()  # costs 1 / 5 / 10 per prompt
    haiku = [{"prompt_hash": p["prompt_hash"], "tier": "claude-haiku-5-5", "error": None} for p in prompts]
    (vs_opus,) = rc.score_rows(prompts, {"r": haiku})
    assert vs_opus["no_router_usd"] == 40.0 and vs_opus["saved_usd"] == 36.0 and vs_opus["saved_pct"] == 0.9
    (vs_sonnet,) = rc.score_rows(prompts, {"r": haiku}, baseline=1)
    assert vs_sonnet["saved_usd"] == 16.0
    assert vs_sonnet["dq_vs_no_router"] == pytest.approx(np.mean([0.9, 0.2, 0.2, 0.9]) - np.mean([0.9, 0.9, 0.3, 0.9]))
    opus = [{**r, "tier": "claude-opus-5-5"} for r in haiku]
    assert rc.score_rows(prompts, {"r": opus})[0]["saved_usd"] == 0.0
