"""The interceptor end to end with routing on: the non-Anthropic lane keeps
tools, a turn's tool loop stays on one model, a failed reroute falls back,
costs and baselines are logged, and the user's cost preference reaches the
router."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from router.gateway import interceptor, server
from router.gateway.config import load_settings
from router.gateway.pricing import CLAUDE_PRICES, cost_usd

AUTH = {"authorization": "Bearer sk-ant-oat-not-a-real-token"}
TOOL = {"name": "Read", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
OPENAI_TOOL_CALL = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
    {"id": "call_1", "function": {"name": "Read", "arguments": '{"path": "a.py"}'}}]}}],
    "usage": {"prompt_tokens": 30, "completion_tokens": 5}}
OPENAI_STREAM = (
    'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2}}\n\n'
    "data: [DONE]\n\n"
)


def turn(text_or_blocks, model="claude-opus-5", **extra):
    content = text_or_blocks if isinstance(text_or_blocks, list) else [{"type": "text", "text": text_or_blocks}]
    return {"model": model, "max_tokens": 64, "tools": [TOOL],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "start"}]},
                         {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                         {"role": "user", "content": content}], **extra}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "requests.jsonl"))
    cfg = tmp_path / "gateway.yaml"
    cfg.write_text("""
interceptor:
  models:
    claude-opus-5:    {lane: anthropic, proxy: x}
    claude-haiku-4-5: {lane: anthropic, proxy: x}
    local-coder:
      lane: openai
      base_url: http://local.test/v1
      prices: {input: 0, output: 0}
""")
    calls = {"anthropic": [], "openai": []}
    state = {"anthropic_status": {}}

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        if req.url.host == "local.test":
            calls["openai"].append(body)
            if body.get("stream"):
                return httpx.Response(200, text=OPENAI_STREAM, headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=OPENAI_TOOL_CALL)
        calls["anthropic"].append(body)
        status = state["anthropic_status"].get(body.get("model"), 200)
        if status != 200:
            return httpx.Response(status, json={"type": "error",
                                                "error": {"message": state.get("error_message", "nope")}})
        return httpx.Response(200, json={"type": "message", "model": body["model"], "content": [],
                                         "usage": state.get("usage", {"input_tokens": 10, "output_tokens": 2})})

    interceptor.set_upstream_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with TestClient(server.app) as client:
        settings = load_settings(cfg)
        yield client, settings, calls, state
    interceptor.set_upstream_client(None)
    interceptor.configure(load_settings(None), None)


def records():
    return interceptor.read_records()


def test_openai_lane_keeps_tools_and_needs_no_anthropic_login(env):
    client, settings, calls, _ = env
    interceptor.configure(settings, None)
    resp = client.post("/v1/messages", json=turn("open a.py", model="local-coder"))
    assert resp.status_code == 200
    assert resp.json()["content"] == [{"type": "tool_use", "id": "call_1", "name": "Read",
                                       "input": {"path": "a.py"}}]
    (sent,) = calls["openai"]
    assert sent["tools"][0]["function"]["name"] == "Read"
    assert calls["anthropic"] == []
    (rec,) = records()
    assert (rec["lane"], rec["cost_usd"], rec["input_tokens"]) == ("openai", 0.0, 30)


def test_openai_lane_streams_back_as_anthropic_events(env):
    client, settings, calls, _ = env
    interceptor.configure(settings, None)
    resp = client.post("/v1/messages", json=turn("hi", model="local-coder", stream=True))
    text = resp.text
    assert "event: message_start" in text and "event: message_stop" in text
    deltas = [json.loads(line[5:]) for line in text.splitlines()
              if line.startswith("data:") and "text_delta" in line]
    assert "".join(d["delta"]["text"] for d in deltas) == "hello"
    assert records()[0]["output_tokens"] == 2


def test_a_turns_tool_loop_stays_on_the_turns_model(env):
    client, settings, calls, _ = env
    asked = []
    interceptor.configure(settings, lambda body, ctx: asked.append(ctx) or "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("fix the bug"), headers=AUTH)
    client.post("/v1/messages", json=turn([{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]),
                headers=AUTH)
    assert len(asked) == 1 and asked[0]["human_text"] == "fix the bug"
    assert [c["model"] for c in calls["anthropic"]] == ["claude-haiku-4-5", "claude-haiku-4-5"]
    assert records()[1]["reason"] == "same turn as the previous request"


def test_system_reminders_are_not_human_text(env):
    client, settings, _, _ = env
    asked = []
    interceptor.configure(settings, lambda body, ctx: asked.append(ctx["human_text"]))
    client.post("/v1/messages", json=turn([{"type": "text", "text": "<system-reminder>x</system-reminder>"},
                                           {"type": "text", "text": "real question"}]), headers=AUTH)
    assert asked == ["real question"]


def test_a_failed_reroute_falls_back_to_the_model_the_client_asked_for(env):
    client, settings, calls, state = env
    state["anthropic_status"]["claude-haiku-4-5"] = 400
    interceptor.configure(settings, lambda body, ctx: "claude-haiku-4-5")
    resp = client.post("/v1/messages", json=turn("hi"), headers=AUTH)
    assert resp.status_code == 200
    assert [c["model"] for c in calls["anthropic"]] == ["claude-haiku-4-5", "claude-opus-5"]
    (rec,) = records()
    assert rec["fallback"] is True and rec["routed_model"] == "claude-opus-5"


def test_baselines_assume_the_other_model_kept_its_cache(env):
    """Turn 1 on Opus caches a 1,000-token prefix. Turn 2 is moved to Haiku,
    which has to rewrite that prefix -- but the "if you had stayed on Opus"
    baseline must read it from cache, or the switch looks free."""
    client, settings, _, state = env
    interceptor.configure(settings, lambda body, ctx: None)
    state["usage"] = {"input_tokens": 0, "output_tokens": 10, "cache_creation_input_tokens": 1000}
    client.post("/v1/messages", json=turn("first"), headers=AUTH)
    interceptor.set_model_picker(lambda body, ctx: "claude-haiku-4-5")
    state["usage"] = {"input_tokens": 0, "output_tokens": 10, "cache_creation_input_tokens": 1200}
    client.post("/v1/messages", json=turn("second"), headers=AUTH)
    rec = records()[1]
    warm = {"input_tokens": 0, "output_tokens": 10, "cache_creation_input_tokens": 200,
            "cache_read_input_tokens": 1000, "cache_creation": {}}
    assert rec["cost_if_requested_usd"] == pytest.approx(cost_usd(CLAUDE_PRICES["claude-opus-5"], warm))
    assert rec["cost_usd"] == pytest.approx(cost_usd(CLAUDE_PRICES["claude-haiku-4-5"], state["usage"]))
    totals = client.get("/stats").json()["totals"]
    assert totals["rerouted"] == 1 and totals["requests"] == 2


def test_cost_preference_is_saved_and_reaches_the_router(env, tmp_path):
    client, settings, _, _ = env
    seen = []
    interceptor.configure(settings, lambda body, ctx: seen.append(ctx["quality_margin"]))
    save_more = interceptor.PRESETS["save_more"]
    assert client.post("/api/preferences", json={"preset": "save_more"}).json()["quality_margin"] == save_more
    assert json.loads((tmp_path / "preferences.json").read_text())["quality_margin"] == save_more
    client.post("/v1/messages", json=turn("a"), headers=AUTH)
    client.post("/v1/messages", json=turn("b"), headers={**AUTH, interceptor.MARGIN_HEADER: "0.02"})
    assert seen == [save_more, 0.02]
    assert client.post("/api/preferences", json={"quality_margin": 2}).status_code == 400


def test_dashboard_page_is_served(env):
    client, *_ = env
    resp = client.get("/dashboard")
    assert resp.status_code == 200 and "Cost preference" in resp.text


def test_human_text_looks_past_claude_codes_trailing_system_message():
    """Found live: Claude Code appends hook output as a `system` message after
    the user's, and puts reminders inside the same text block as the prompt."""
    body = {"messages": [
        {"role": "user", "content": [{"type": "text", "text":
            "<system-reminder>\ncontext\n</system-reminder>\nWhat is the capital of France?"}]},
        {"role": "system", "content": [{"type": "text", "text": "SessionStart hook success"}]}]}
    assert interceptor.human_text(body) == "What is the capital of France?"


def test_side_requests_without_tools_are_not_routed(env):
    """Found live: Claude Code's title request switches thinking off, which
    Opus/Sonnet 5.5 reject -- routing it only ever bought a 400 and a retry."""
    client, settings, calls, _ = env
    asked = []
    interceptor.configure(settings, lambda body, ctx: asked.append(1) or "claude-haiku-4-5")
    body = {**turn("name this session"), "tools": []}
    client.post("/v1/messages", json=body, headers=AUTH)
    assert asked == [] and calls["anthropic"][0]["model"] == "claude-opus-5"
    assert records()[0]["reason"] == "side request (no tools), not routed"


def test_models_that_reject_the_requests_settings_are_skipped():
    from router.gateway.routing import incompatibility

    agent = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
    assert incompatibility(agent, {"rejects": ["effort"]}) == "rejects output_config.effort"
    assert incompatibility(agent, {"rejects": ["thinking_disabled"]}) is None
    assert incompatibility({"thinking": {"type": "disabled"}}, {"rejects": ["thinking_disabled"]})


def test_a_model_the_upstream_rejects_for_this_client_is_not_tried_again(env):
    """Found live: Anthropic answers 400 to a model the client's Claude Code
    version doesn't support. One fallback is fine; retrying it every turn
    is not."""
    client, settings, calls, state = env
    state["anthropic_status"]["claude-haiku-4-5"] = 400
    seen = []
    interceptor.configure(settings, lambda body, ctx: seen.append(dict(ctx["blocked"])) or "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("one"), headers={**AUTH, "user-agent": "claude-cli/1.0"})
    client.post("/v1/messages", json={**turn("two"), "messages": [{"role": "user", "content": "x"}]},
                headers={**AUTH, "user-agent": "claude-cli/1.0"})
    assert seen[0] == {} and "claude-haiku-4-5" in seen[1]


def test_subagent_hand_backs_and_task_notifications_are_not_new_turns(env):
    """Found live: with background subagents, Claude Code tells the main agent
    each one finished with two user messages of its own. Routing each as a new
    turn moved the main conversation between models mid-task."""
    client, settings, calls, _ = env
    asked = []
    interceptor.configure(settings, lambda body, ctx: asked.append(ctx["human_text"]) or "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("research the gateway with subagents"), headers=AUTH)
    hand_back = ('Another Claude session sent a message:\n<agent-message from="a41b9e">\n[Subagent hand-back] '
                 'The text below is the final report of a subagent.\nsrc/x.py:12\n</agent-message>')
    notification = ("<task-notification>\n<task-id>a41b9e</task-id>\n<status>completed</status>\n"
                    "</task-notification>")
    for text in (hand_back, notification):
        client.post("/v1/messages", json=turn(text), headers=AUTH)
    assert asked == ["research the gateway with subagents"]
    assert [c["model"] for c in calls["anthropic"]] == ["claude-haiku-4-5"] * 3
    assert [r["reason"] for r in records()[1:]] == ["same turn as the previous request"] * 2
    client.post("/v1/messages", json=turn([{"type": "text", "text": notification},
                                           {"type": "text", "text": "now summarise it"}]), headers=AUTH)
    assert asked[-1] == "now summarise it"  # the human's own words still start a turn


def test_a_400_about_the_request_itself_does_not_block_the_model(env):
    """Anthropic answers 400 to a prompt too long for a model: that says
    nothing about the next request, which may be short."""
    client, settings, calls, state = env
    state["anthropic_status"]["claude-haiku-4-5"] = 400
    state["error_message"] = "prompt is too long: 210000 tokens > 200000 maximum"
    seen = []
    interceptor.configure(settings, lambda body, ctx: seen.append(dict(ctx["blocked"])) or "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("one"), headers=AUTH)
    client.post("/v1/messages", json={**turn("two"), "messages": [{"role": "user", "content": "x"}]}, headers=AUTH)
    assert seen == [{}, {}]


def test_a_blocked_model_is_tried_again_after_a_while(env, monkeypatch):
    client, settings, calls, state = env
    state["anthropic_status"]["claude-haiku-4-5"] = 400
    seen = []
    interceptor.configure(settings, lambda body, ctx: seen.append(dict(ctx["blocked"])) or "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("one"), headers=AUTH)
    monkeypatch.setattr(interceptor, "BLOCK_SECONDS", 0.0)
    client.post("/v1/messages", json={**turn("two"), "messages": [{"role": "user", "content": "x"}]}, headers=AUTH)
    assert seen == [{}, {}]


def test_a_new_session_sharing_the_prefix_counts_the_cache_an_earlier_one_wrote(env):
    """Found live: session A cached Claude Code's system prompt + tools on
    Opus; session B (same prefix) was moved to another model and rewrote it.
    Opus would have *read* it -- the baseline must say so, and the router must
    see Opus as warm."""
    client, settings, _, state = env
    seen = []

    def body(first):
        return {**turn("x"), "messages": [{"role": "user", "content": first}]}

    interceptor.configure(settings, lambda b, ctx: seen.append(ctx["warm_tokens"]))
    state["usage"] = {"input_tokens": 0, "output_tokens": 5, "cache_creation_input_tokens": 1000}
    client.post("/v1/messages", json=body("session A"), headers=AUTH)

    interceptor.set_model_picker(lambda b, ctx: seen.append(ctx["warm_tokens"]) or "claude-haiku-4-5")
    state["usage"] = {"input_tokens": 0, "output_tokens": 5, "cache_creation_input_tokens": 1100}
    client.post("/v1/messages", json=body("session B"), headers=AUTH)

    assert seen == [{}, {"claude-opus-5": 1000}]
    rec = records()[1]
    opus_warm = {"input_tokens": 0, "output_tokens": 5, "cache_creation_input_tokens": 100,
                 "cache_read_input_tokens": 1000, "cache_creation": {}}
    assert rec["cost_if_requested_usd"] == pytest.approx(cost_usd(CLAUDE_PRICES["claude-opus-5"], opus_warm))


def test_status_line_shows_the_model_that_actually_answered(env):
    client, settings, _, _ = env
    interceptor.configure(settings, lambda body, ctx: "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("hi"), headers={**AUTH, "x-claude-code-session-id": "s-1"})
    line = client.get("/api/statusline", params={"session_id": "s-1"}).json()["line"]
    assert "claude-haiku-4-5 (routed)" in line and "less plan usage" in line


def test_session_id_falls_back_to_metadata_user_id(env):
    client, settings, _, _ = env
    interceptor.configure(settings, None)
    body = {**turn("hi"), "metadata": {"user_id": "user_abc_account_x_session_s-2"}}
    client.post("/v1/messages", json=body, headers=AUTH)
    assert "claude-opus-5 (as asked)" in client.get("/api/statusline", params={"session_id": "s-2"}).json()["line"]


def test_an_unreachable_provider_is_a_logged_502_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "requests.jsonl"))

    def down(req):
        raise httpx.ConnectError("no route to host")

    interceptor.set_upstream_client(httpx.AsyncClient(transport=httpx.MockTransport(down)))
    with TestClient(server.app) as client:
        interceptor.configure(load_settings(None), None)
        resp = client.post("/v1/messages", json=turn("hi"), headers=AUTH)
    interceptor.set_upstream_client(None)
    assert resp.status_code == 502 and resp.json()["type"] == "error"
    (rec,) = records()
    assert rec["status"] == 502 and "unreachable" in rec["reason"]


def test_capture_is_opt_in_and_keeps_only_the_humans_words_and_folder(env, tmp_path):
    client, settings, _, _ = env
    interceptor.configure(settings, None)
    client.post("/v1/messages", json=turn("first"), headers=AUTH)
    assert not interceptor.turns_path().exists()

    interceptor.configure({**settings, "capture_prompts": True}, None)
    env_block = {"type": "text", "text": "<system-reminder>\n - Primary working directory: /repo/x\n</system-reminder>"}
    body = turn([env_block, {"type": "text", "text": "refactor the loader"}])
    client.post("/v1/messages", json=body, headers=AUTH)
    client.post("/v1/messages", json=turn([{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]),
                headers=AUTH)
    (rec,) = [json.loads(line) for line in interceptor.turns_path().read_text().splitlines()]
    assert rec["prompt"] == "refactor the loader" and rec["cwd"] == "/repo/x"
    assert "system-reminder" not in rec["prompt"]


def test_the_log_says_what_started_a_turn_without_its_text():
    assert interceptor.turn_trigger(None) is None
    assert interceptor.turn_trigger("fix the login bug") == "typed"
    assert interceptor.turn_trigger("<command-name>/compact</command-name> secret") == "command-name"


def test_the_dashboard_is_told_which_models_are_not_claude(env):
    _, settings, _, _ = env
    settings["models"]["llama3.1:latest"] = {"lane": "openai", "base_url": "http://127.0.0.1:11434/v1"}
    interceptor.configure(settings, None)
    notes = interceptor.preferences()["model_notes"]
    assert notes == {"local-coder": "other provider", "llama3.1:latest": "local"}
