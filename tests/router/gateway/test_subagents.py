"""Claude Code subagents through the gateway: recognised by Claude Code's
``cc_is_subagent=true`` marker, given a role (definition, built-in, or task
keywords), and routed to the role's tier -- never above the session's model,
sticky for the subagent's tool loop, logged and summarised per role."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from router.gateway import interceptor, server
from router.gateway.config import load_settings
from router.gateway.subagents import (ROLES, agent_definitions, agent_markdown, classify_task, detect,
                                      role_decision)

AUTH = {"authorization": "Bearer sk-ant-oat-not-a-real-token"}
TOOL = {"name": "Read", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
BILLING = "x-anthropic-billing-header: cc_version=2.1.293; cc_entrypoint=cli; cc_is_subagent=true;"
MAIN_BILLING = "x-anthropic-billing-header: cc_version=2.1.293; cc_entrypoint=cli;"
EXPLORE = "You are a file search specialist for Claude Code, Anthropic's official CLI for Claude."
PLAN = "You are a software architect and planning specialist for Claude Code."


def subagent_body(task, system_prompt="You are an agent.", model="claude-opus-5-5", messages=None, **extra):
    return {"model": model, "max_tokens": 64, "tools": [TOOL],
            "system": [{"type": "text", "text": BILLING}, {"type": "text", "text": system_prompt}],
            "messages": messages or [{"role": "user", "content": [{"type": "text", "text": task}]}], **extra}


@pytest.mark.parametrize("task, role", [
    ("Find where CLAUDE_PRICES is defined", "scout"),
    ("Explore the training pipeline and report the entry points", "scout"),
    ("Label real user prompts with a task type. Read the file, output labels", "extractor"),
    ("Summarize the failing CI log", "extractor"),
    ("Review the diff in src/router for bugs", "reviewer"),
    ("Find bugs in the interceptor", "reviewer"),
    ("Write unit tests for pricing.py", "tester"),
    ("Propose an approach for caching embeddings", "planner"),
    ("Find the root cause of the flaky test", "architect"),
    ("Generate 100 prompts that a real person would send", None),
    ("Write 100 prompts, all examples of the task type SUMMARIZE", None),
    ("Implement the cache and find where it is read", None),
])
def test_tasks_are_classified_by_what_they_ask_for(task, role):
    found = classify_task(task)
    assert (found[0] if found else None) == role


def test_only_requests_marked_as_subagents_are_subagents():
    main = subagent_body("Find X")
    main["system"][0]["text"] = MAIN_BILLING
    assert detect(main, "Find X") is None
    assert detect(subagent_body("Find X"), "Find X")["role"] == "scout"


def test_definition_beats_builtin_beats_task():
    marked = subagent_body("Find X", system_prompt=agent_definitions()["reviewer"]["prompt"])
    assert detect(marked, "Find X") == {"role": "reviewer", "source": "definition", "why": "aar-role: reviewer"}
    explore = detect(subagent_body("Review this", system_prompt=EXPLORE), "Review this")
    assert (explore["role"], explore["source"]) == ("explore", "builtin")
    plan = detect(subagent_body("Find X", system_prompt=PLAN), "Find X")
    assert (plan["role"], plan["source"]) == ("plan", "builtin")
    assert role_decision(plan, subagent_body("Find X"), "claude-opus-5-5", load_settings(None), {},
                         1000)["model"] == "claude-sonnet-5-5"
    unknown = detect(subagent_body("Generate 100 prompts"), "Generate 100 prompts")
    assert unknown == {"role": None, "source": None, "why": None}


def test_every_role_definition_is_recognised_and_has_an_output_budget():
    for name, d in agent_definitions().items():
        assert detect(subagent_body("x", system_prompt=d["prompt"]), "x")["role"] == name
        assert "Output budget:" in d["prompt"]
    assert agent_markdown("scout").startswith("---\nname: scout\n")
    assert set(ROLES) == {"scout", "extractor", "reviewer", "planner", "tester", "architect"}


def test_role_tier_is_a_ceiling_on_cost_not_an_upgrade():
    settings = load_settings(None)
    agent = {"role": "reviewer", "source": "task", "why": "review"}
    body = subagent_body("Review it")
    assert role_decision(agent, body, "claude-opus-5-5", settings, {}, 1000)["model"] == "claude-sonnet-5-5"
    kept = role_decision(agent, body, "claude-haiku-5-5", settings, {}, 1000)
    assert kept["model"] == "claude-haiku-5-5" and "would cost more" in kept["reason"]
    architect = {"role": "architect", "source": "task", "why": "root cause"}
    assert role_decision(architect, body, "claude-opus-5-5", settings, {}, 1000)["model"] == "claude-opus-5-5"


def test_a_role_model_that_cant_take_the_request_is_left_to_the_router():
    settings = load_settings(None)
    settings["subagents"]["roles"] = {"scout": "claude-haiku-4-5"}  # rejects `effort`
    agent = {"role": "scout", "source": "task", "why": "find"}
    body = subagent_body("Find X", output_config={"effort": "high"})
    assert role_decision(agent, body, "claude-opus-5-5", settings, {}, 1000) is None
    assert role_decision({"role": "scout", "source": "task", "why": "find"}, subagent_body("Find X"),
                         "claude-opus-5-5", load_settings(None), {"claude-haiku-5-5": "400"}, 1000) is None
    assert role_decision({"role": None, "source": None, "why": None}, body, "claude-opus-5-5",
                         load_settings(None), {}, 1000) is None


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "requests.jsonl"))
    sent = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        sent.append(body["model"])
        return httpx.Response(200, json={"type": "message", "model": body["model"], "content": [],
                                         "usage": {"input_tokens": 10, "output_tokens": 5,
                                                   "cache_read_input_tokens": 1000}})

    interceptor.set_upstream_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with TestClient(server.app) as client:
        yield client, sent
    interceptor.set_upstream_client(None)
    interceptor.configure(load_settings(None), None)


def test_a_subagent_is_routed_by_role_and_its_tool_loop_stays_there(gateway):
    client, sent = gateway
    picked = []
    interceptor.configure(load_settings(None), lambda body, ctx: picked.append(ctx) or None)
    task = "Find where CLAUDE_PRICES is defined"
    first = [{"role": "user", "content": [{"type": "text", "text": task}]}]
    client.post("/v1/messages", json=subagent_body(task, messages=first), headers=AUTH)
    loop = first + [{"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}]
    client.post("/v1/messages", json=subagent_body(task, messages=loop), headers=AUTH)
    assert sent == ["claude-haiku-5-5", "claude-haiku-5-5"]
    assert picked == []  # the role decided; the general router was not asked
    recs = interceptor.read_records()
    assert [(r["subagent"], r["agent_role"], r["role_source"]) for r in recs] == [(True, "scout", "task")] * 2
    assert recs[1]["reason"] == "same turn as the previous request"

    (row,) = interceptor.summarize()["subagents"]
    assert (row["role"], row["spawns"], row["requests"], row["models"]) == ("scout", 1, 2, {"claude-haiku-5-5": 2})
    assert row["cost_usd"] < row["cost_if_requested_usd"]
    assert row["cache_hit_rate"] == pytest.approx(1000 / 1010)


def test_an_unrecognised_subagent_goes_to_the_general_router(gateway):
    client, sent = gateway
    interceptor.configure(load_settings(None), lambda body, ctx: "claude-sonnet-5-5")
    client.post("/v1/messages", json=subagent_body("Generate 100 prompts"), headers=AUTH)
    assert sent == ["claude-sonnet-5-5"]
    (rec,) = interceptor.read_records()
    assert (rec["subagent"], rec["agent_role"]) == (True, None)


def test_subagent_roles_can_be_turned_off(gateway, tmp_path):
    client, sent = gateway
    cfg = tmp_path / "gateway.yaml"
    cfg.write_text("interceptor:\n  subagents: {enabled: false}\n")
    interceptor.configure(load_settings(cfg), None)
    client.post("/v1/messages", json=subagent_body("Find X"), headers=AUTH)
    assert sent == ["claude-opus-5-5"]


def test_claude_code_is_given_the_roles_unless_the_user_brings_their_own(monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_STATUSLINE", "0")
    args, _ = server._claude_command("http://127.0.0.1:8787", [])
    assert json.loads(args[args.index("--agents") + 1]) == agent_definitions()
    args, _ = server._claude_command("http://127.0.0.1:8787", ["--agents", "{}"])
    assert args.count("--agents") == 1
    monkeypatch.setenv("AAR_GATEWAY_AGENTS", "0")
    assert "--agents" not in server._claude_command("http://127.0.0.1:8787", [])[0]


def test_agents_install_writes_files_and_keeps_existing_ones(tmp_path, capsys):
    (tmp_path / "scout.md").write_text("mine")
    server.main(["agents", "--install", str(tmp_path)])
    assert (tmp_path / "scout.md").read_text() == "mine"
    assert (tmp_path / "reviewer.md").read_text() == agent_markdown("reviewer")


def main_turn(text, model="claude-opus-5-5"):
    return {"model": model, "max_tokens": 64, "tools": [TOOL], "system": [{"type": "text", "text": MAIN_BILLING}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]}


def test_routing_off_keeps_the_asked_model_and_pin_sends_every_agent_turn_to_one(gateway, tmp_path):
    client, sent = gateway
    interceptor.configure(load_settings(None), lambda body, ctx: "claude-haiku-5-5")
    assert client.post("/api/preferences", json={"mode": "off"}).json()["mode"] == "off"
    client.post("/v1/messages", json=main_turn("fix the bug"), headers=AUTH)
    client.post("/v1/messages", json=subagent_body("Find X"), headers=AUTH)
    assert sent == ["claude-opus-5-5", "claude-opus-5-5"]

    prefs = client.post("/api/preferences", json={"mode": "pin", "pinned_model": "claude-sonnet-5-5"}).json()
    assert (prefs["mode"], prefs["pinned_model"]) == ("pin", "claude-sonnet-5-5")
    client.post("/v1/messages", json=main_turn("another question", model="claude-haiku-5-5"), headers=AUTH)
    assert sent[-1] == "claude-sonnet-5-5"  # a pin may be pricier: the user chose it
    assert interceptor.read_records()[-1]["reason"] == "pinned to claude-sonnet-5-5 (dashboard)"

    assert client.post("/api/preferences", json={"mode": "pin", "pinned_model": "gpt-9"}).status_code == 400
    interceptor.configure(load_settings(None), None)  # a restart keeps the choice
    assert interceptor.preferences()["mode"] == "pin"
    kinds = [json.loads(line)["kind"] for line in interceptor.overrides_path().read_text().splitlines()]
    assert kinds == ["routing", "routing"]


def test_a_pinned_model_that_cant_take_the_request_is_not_forced(gateway):
    client, sent = gateway
    interceptor.configure(load_settings(None), None)
    client.post("/api/preferences", json={"mode": "pin", "pinned_model": "claude-haiku-4-5"})
    body = main_turn("hi")
    body["output_config"] = {"effort": "high"}  # Haiku 4.5 rejects it
    client.post("/v1/messages", json=body, headers=AUTH)
    assert sent == ["claude-opus-5-5"]
    assert "can't take this request" in interceptor.read_records()[-1]["reason"]


def test_a_role_tier_can_be_changed_reset_and_set_to_the_sessions_model(gateway):
    client, sent = gateway
    interceptor.configure(load_settings(None), None)
    prefs = client.post("/api/preferences", json={"role": "scout", "model": "claude-sonnet-5-5"}).json()
    scout = next(r for r in prefs["roles"] if r["role"] == "scout")
    assert (scout["model"], scout["default"], scout["overridden"]) == ("claude-sonnet-5-5", "claude-haiku-5-5", True)
    client.post("/v1/messages", json=subagent_body("Find X"), headers=AUTH)
    assert sent == ["claude-sonnet-5-5"]

    client.post("/api/preferences", json={"role": "scout", "model": "session"})
    client.post("/v1/messages", json=subagent_body("Find Y"), headers=AUTH)
    assert sent[-1] == "claude-opus-5-5"

    prefs = client.post("/api/preferences", json={"role": "scout", "model": "default"}).json()
    assert next(r for r in prefs["roles"] if r["role"] == "scout")["overridden"] is False
    client.post("/v1/messages", json=subagent_body("Find Z"), headers=AUTH)
    assert sent[-1] == "claude-haiku-5-5"
    assert client.post("/api/preferences", json={"role": "wizard", "model": "default"}).status_code == 400


def test_allow_upgrade_reaches_the_router_and_is_kept(gateway):
    client, sent = gateway
    seen = []
    interceptor.configure(load_settings(None), lambda body, ctx: seen.append(ctx["allow_upgrade"]))
    client.post("/v1/messages", json=main_turn("hard question", model="claude-haiku-5-5"), headers=AUTH)
    assert client.post("/api/preferences", json={"allow_upgrade": True}).json()["allow_upgrade"] is True
    client.post("/v1/messages", json=main_turn("another hard one", model="claude-haiku-5-5"), headers=AUTH)
    assert seen == [False, True]
    interceptor.configure(load_settings(None), None)
    assert interceptor.preferences()["allow_upgrade"] is True


def test_sessions_sum_plan_usage_and_count_turns_and_subagents():
    base = {"status": 200, "requested_model": "claude-opus-5-5", "cost_if_strongest_usd": 1.0}
    records = [
        {**base, "ts": "2026-10-10T10:00:00+00:00", "claude_session": "s1", "conversation": "c1", "new_turn": True,
         "routed_model": "claude-sonnet-5-5", "cost_usd": 0.4, "cost_if_requested_usd": 0.8},
        {**base, "ts": "2026-10-10T10:01:00+00:00", "claude_session": "s1", "conversation": "c2", "new_turn": True,
         "subagent": True, "routed_model": "claude-haiku-5-5", "cost_usd": 0.01, "cost_if_requested_usd": 0.2},
        {**base, "ts": "2026-10-10T10:02:00+00:00", "claude_session": "s1", "conversation": "c1", "new_turn": True,
         "reason": "side request (no tools), not routed", "routed_model": "claude-opus-5-5", "cost_usd": 0.01,
         "cost_if_requested_usd": None},
        {**base, "ts": "2026-10-09T09:00:00+00:00", "claude_session": None, "conversation": "c0", "new_turn": True,
         "routed_model": "claude-opus-5-5", "cost_usd": 0.5, "cost_if_requested_usd": 0.5},
    ]
    s1, before = interceptor.summarize_sessions(records)
    assert (s1["session"], s1["requests"], s1["turns"], s1["subagent_spawns"], s1["rerouted"]) == ("s1", 3, 1, 1, 2)
    assert s1["cost_usd"] == pytest.approx(0.42) and s1["cost_if_requested_usd"] == pytest.approx(1.01)
    assert s1["models"] == {"claude-sonnet-5-5": 1, "claude-haiku-5-5": 1, "claude-opus-5-5": 1}
    assert (before["session"], before["requests"]) == (None, 1)
