"""A subagent's answer coming back to the main agent: the main agent's
request that carries an Agent call's ``tool_result`` is logged with it
(``returned``), under the same spawn id as the subagent's own requests, so a
session's turns show each subagent, its model, and the answer it returned."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from test_subagents import AUTH, MAIN_BILLING, TOOL, subagent_body

from aar_gateway import cli, history, interceptor, rollup, server, share
from aar_gateway.config import load_settings
from aar_gateway.subagents import DELEGATE_PROMPT

TASK = "Find where CLAUDE_PRICES is defined"
OTHER = "Write unit tests for pricing.py"
ANSWER = "src/aar_gateway/pricing.py:12 defines CLAUDE_PRICES."
SESSION = {**AUTH, "x-claude-code-session-id": "session-one-abc"}


def agent_call(call_id, prompt, kind="lookup"):
    return {"type": "tool_use", "id": call_id, "name": "Agent",
            "input": {"subagent_type": kind, "description": "d", "prompt": prompt}}


def result(call_id, text):
    return {"type": "tool_result", "tool_use_id": call_id, "content": [{"type": "text", "text": text}]}


def main_body(messages, model="claude-opus-5-5"):
    return {"model": model, "max_tokens": 64, "tools": [TOOL], "system": [{"type": "text", "text": MAIN_BILLING}],
            "messages": messages}


ASK = [{"role": "user", "content": [{"type": "text", "text": "where are the prices, and test them"}]}]
CALLS = {"role": "assistant", "content": [agent_call("toolu_a", TASK), agent_call("toolu_b", OTHER, "make"),
                                          {"type": "tool_use", "id": "toolu_r", "name": "Read", "input": {}}]}
RESULTS = {"role": "user", "content": [result("toolu_a", ANSWER), result("toolu_b", "3 tests added, all pass"),
                                       {"type": "tool_result", "tool_use_id": "toolu_r", "content": "file text"}]}


def test_each_agent_result_is_a_returned_answer_under_its_tasks_spawn_id():
    found = interceptor.returned_spawns(main_body(ASK + [CALLS, RESULTS]))
    assert [(a["spawn_id"], a["tool_use_id"]) for a in found] == [
        (interceptor._task_key(TASK), "toolu_a"), (interceptor._task_key(OTHER), "toolu_b")]
    assert found[0]["answer_tokens"] == -(-len(ANSWER) // 4)
    assert found[0]["answer_preview"] == ANSWER  # prompt history is on by default
    assert interceptor.returned_spawns(main_body(ASK)) == []  # a typed message returns nothing


def test_an_answer_whose_call_was_not_resent_is_matched_through_the_spawn_hook():
    """A server-side thread sends only the new message: the Agent call is not
    in the body, but the hook saw it, with its tool_use_id."""
    interceptor.configure(load_settings(None), None)
    interceptor.record_spawn("s1", "lookup", TASK, tool_use_id="toolu_a")
    (found,) = interceptor.returned_spawns(main_body([RESULTS]))
    assert (found["spawn_id"], found["tool_use_id"]) == (interceptor._task_key(TASK), "toolu_a")


def test_previews_are_kept_only_with_prompt_history_on():
    history.set_enabled(False)
    try:
        (found, _) = interceptor.returned_spawns(main_body(ASK + [CALLS, RESULTS]))
        assert found["answer_preview"] is None and found["answer_tokens"] > 0
    finally:
        history.set_enabled(True)


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "requests.jsonl"))
    interceptor.set_upstream_client(httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"type": "message", "model": json.loads(req.content)["model"],
                                              "content": [], "usage": {"input_tokens": 10, "output_tokens": 5}}))))
    with TestClient(server.app) as client:
        interceptor.configure(load_settings(None), None)
        yield client
    interceptor.set_upstream_client(None)
    interceptor.configure(load_settings(None), None)


def run_turn(client):
    """One typed turn: the main agent spawns a subagent, which makes two
    requests, and the main agent's next request carries its answer back."""
    client.post("/v1/messages", json=main_body(ASK), headers=SESSION)
    client.post("/api/subagent-spawn", json={"session_id": "session-one-abc", "subagent_type": "lookup",
                                             "prompt": TASK, "tool_use_id": "toolu_a"})
    first = [{"role": "user", "content": [{"type": "text", "text": TASK}]}]
    client.post("/v1/messages", json=subagent_body(TASK, messages=first), headers=SESSION)
    loop = first + [{"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}]
    client.post("/v1/messages", json=subagent_body(TASK, messages=loop), headers=SESSION)
    back = ASK + [{"role": "assistant", "content": [agent_call("toolu_a", TASK)]},
                  {"role": "user", "content": [result("toolu_a", ANSWER)]}]
    client.post("/v1/messages", json=main_body(back), headers=SESSION)


def test_a_subagents_requests_and_its_returned_answer_share_one_spawn_id(gateway):
    run_turn(gateway)
    main, sub1, sub2, back = interceptor.read_records()
    spawn = interceptor._task_key(TASK)
    assert (main["spawn_id"], main["returned"]) == (None, None)
    assert (sub1["spawn_id"], sub2["spawn_id"]) == (spawn, spawn)  # the tool loop keeps it
    assert sub1["task_preview"] == TASK and sub2["task_preview"] is None
    assert [a["spawn_id"] for a in back["returned"]] == [spawn]
    assert not back["subagent"]


def test_a_sessions_turns_show_each_subagent_and_the_answer_it_returned(gateway):
    run_turn(gateway)
    (turn,) = interceptor.session_requests("session-one-abc")["turns"]
    (spawned,) = turn["subagents"]
    assert (spawned["role"], spawned["requests"], spawned["returned"]) == ("lookup", 2, True)
    assert spawned["models"] == ["claude-haiku-5-5"] and spawned["model"] == "claude-haiku-5-5"
    assert (spawned["task_preview"], spawned["answer_preview"]) == (TASK, ANSWER)
    assert spawned["answer_tokens"] > 0 and spawned["cost_usd"] > 0
    assert turn["returned"] == 1 and turn["requests"] == 2
    assert turn["subagent_cost_usd"] == pytest.approx(spawned["cost_usd"])


def test_the_summary_counts_answers_returned_per_kind(gateway):
    run_turn(gateway)
    s = interceptor.summarize()
    assert (s["totals"]["spawns"], s["totals"]["spawns_returned"]) == (1, 1)
    assert s["totals"]["subagent_cost_usd"] > 0
    (row,) = s["subagents"]
    assert (row["spawns"], row["tracked_spawns"], row["returned"]) == (1, 1, 1)
    assert row["avg_answer_tokens"] == -(-len(ANSWER) // 4)
    assert s["sessions"][0]["subagent_spawns"] == 1


def test_a_spawn_still_running_is_not_returned():
    records = [{"ts": "2026-10-10T10:00:00+00:00", "claude_session": "s", "new_turn": True, "status": 200},
               {"ts": "2026-10-10T10:00:05+00:00", "claude_session": "s", "subagent": True, "agent_role": "make",
                "spawn_id": "abc", "routed_model": "claude-sonnet-5-5", "status": 200}]
    (turn,) = interceptor.session_turns(records)
    assert turn["subagents"][0]["returned"] is False


def test_an_archived_rollup_from_before_spawn_ids_still_merges():
    old = rollup.new_state()
    rollup.accumulate(old, {"subagent": True, "agent_role": "lookup", "conversation": "c", "status": 200})
    del old["returns"]
    del old["subagents"]["lookup"]["spawn_ids"]
    new = rollup.new_state()
    rollup.accumulate(new, {"subagent": True, "agent_role": "lookup", "spawn_id": "x", "conversation": "d"})
    rollup.accumulate(new, {"returned": [{"spawn_id": "x", "answer_tokens": 7}]})
    out = rollup.finish(rollup.merge(old, new), recent=[], log="", default_strongest=None)
    (row,) = out["subagents"]
    assert (row["spawns"], row["tracked_spawns"], row["returned"], row["avg_answer_tokens"]) == (2, 1, 1, 7)
    assert rollup.finish(old, recent=[], log="", default_strongest=None)["totals"]["spawns_returned"] == 0


def test_previews_are_never_shared():
    record = {"ts": "2026-10-10T10:00:00+00:00", "task_preview": TASK, "spawn_id": "x",
              "returned": [{"spawn_id": "x", "answer_preview": ANSWER}]}
    shared = json.dumps(share.build_record(record, None, embed=False))
    assert TASK not in shared and ANSWER not in shared


def test_claude_code_is_told_to_delegate_unless_the_user_appends_their_own(monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_STATUSLINE", "0")
    args, _ = cli._claude_command("http://127.0.0.1:8787", [])
    assert args[args.index("--append-system-prompt") + 1] == DELEGATE_PROMPT
    args, _ = cli._claude_command("http://127.0.0.1:8787", ["--append-system-prompt", "mine"])
    assert args.count("--append-system-prompt") == 1 and args[-1] == "mine"
    monkeypatch.setenv("AAR_GATEWAY_AGENTS", "0")
    assert "--append-system-prompt" not in cli._claude_command("http://127.0.0.1:8787", [])[0]


def test_the_hook_passes_the_agent_calls_id(monkeypatch, capsys):
    import io
    import urllib.request

    sent = []

    class Reply:
        def read(self): return b"{}"
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: sent.append(json.loads(req.data)) or Reply())
    event = {"tool_name": "Agent", "session_id": "s1", "tool_use_id": "toolu_a", "tool_input": {"prompt": TASK}}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    cli.main(["hook"])
    assert sent[0]["tool_use_id"] == "toolu_a"
