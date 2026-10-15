"""The gateway end to end, with a real logged-in Claude Code.

Opt in: these spend a little of the plan and need ``claude`` logged in, so
they run only with ``AAR_E2E=1``:

    AAR_E2E=1 pytest -m claude_e2e tests/e2e -v

Everything runs against an isolated gateway (its own home folder and port)
and a scratch project whose ``.claude/settings.local.json`` points Claude
Code at it, so the machine's own gateway and settings are never touched.
``AAR_GATEWAY_CLAUDE_SETTINGS`` makes every hook -- the project's and the
user's own -- edit that project file, not ``~/.claude/settings.json``.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.claude_e2e,
    pytest.mark.skipif(os.environ.get("AAR_E2E") != "1", reason="set AAR_E2E=1: uses the plan and a logged-in claude"),
    pytest.mark.skipif(shutil.which("claude") is None, reason="claude is not installed"),
]

SRC = str(Path(__file__).resolve().parents[2] / "src")
CLI = [sys.executable, "-m", "router.gateway.cli"]


class Rig:
    """An isolated gateway plus a project that sends Claude Code to it."""

    def __init__(self, root: Path):
        self.root = root
        self.home = root / "home"
        self.data = self.home / ".aar-gateway"
        self.project = root / "project"
        self.settings = self.project / ".claude" / "settings.local.json"
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.config = self.data / "config.yaml"
        self.broken = root / "broken.yaml"
        self.broken.write_text("interceptor: [not, a, mapping\n")
        # this gateway's processes: an isolated home (config, router file, logs)
        self.env = {**os.environ, "HOME": str(self.home), "AAR_GATEWAY_PORT": str(self.port),
                    "PYTHONPATH": SRC}
        for key in ("AAR_GATEWAY_CONFIG", "AAR_GATEWAY_LOG", "ANTHROPIC_BASE_URL"):
            self.env.pop(key, None)

    def gateway(self, *args: str, **kw) -> subprocess.CompletedProcess:
        return subprocess.run([*CLI, *args], env=self.env, capture_output=True, text=True, timeout=120, **kw)

    def write_settings(self, *, config: Path | None = None, base_url: bool = True) -> None:
        """What `aar-gateway enable` writes, scoped to the scratch project."""
        hook = " ".join(CLI)
        env = {"AAR_GATEWAY_PORT": str(self.port), "AAR_GATEWAY_LOG": str(self.data / "requests.jsonl"),
               "AAR_GATEWAY_CONFIG": str(config or self.config),
               "AAR_GATEWAY_CLAUDE_SETTINGS": str(self.settings), "PYTHONPATH": SRC}
        if base_url:
            env["ANTHROPIC_BASE_URL"] = self.url
        self.settings.parent.mkdir(parents=True, exist_ok=True)
        self.settings.write_text(json.dumps({"env": env, "hooks": {
            "SessionStart": [{"hooks": [{"type": "command", "command": f"{hook} check --session", "timeout": 90}]}],
            "UserPromptSubmit": [{"hooks": [{"type": "command", "command": f"{hook} check", "timeout": 30}]}],
            "PostToolUse": [{"hooks": [{"type": "command", "command": f"{hook} check", "timeout": 30}]}],
            "PreToolUse": [{"matcher": "Agent|Task",
                            "hooks": [{"type": "command", "command": f"{hook} hook", "timeout": 5}]}],
        }}, indent=2))

    def project_base_url(self) -> str | None:
        return json.loads(self.settings.read_text())["env"].get("ANTHROPIC_BASE_URL")

    def claude(self, prompt: str, *args: str, timeout: float = 300) -> str:
        env = {k: v for k, v in os.environ.items() if not k.startswith("AAR_") and k != "ANTHROPIC_BASE_URL"}
        done = subprocess.run(["claude", "-p", prompt, *args], cwd=self.project, env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
        return (done.stdout + done.stderr).strip()

    def records(self) -> list[dict]:
        path = self.data / "requests.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=2) as resp:
                return resp.status == 200
        except OSError:
            return False

    def ensure_up(self) -> None:
        if not self.healthy():
            assert self.gateway("start").returncode == 0
        assert self.healthy()


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    rig = Rig(tmp_path_factory.mktemp("e2e"))
    rig.project.mkdir()
    setup = rig.gateway("setup", "--no-enable", stdin=subprocess.DEVNULL)
    assert rig.config.exists() and (rig.data / "router.npz").exists(), setup.stdout + setup.stderr
    rig.write_settings()
    rig.ensure_up()
    yield rig
    rig.gateway("stop")


def new_records(rig: Rig, before: int) -> list[dict]:
    return rig.records()[before:]


def test_setup_loads_the_shipped_router(rig):
    health = json.loads(urllib.request.urlopen(f"{rig.url}/health", timeout=5).read())
    assert health["router"]["enabled"] and health["router"]["kind"] == "mlp" and not health["router"]["error"]


def test_a_request_goes_through_the_gateway_and_is_logged(rig):
    before = len(rig.records())
    assert "51" in rig.claude("What is 17 times 3? Reply with just the number.")
    recs = new_records(rig, before)
    main = [r for r in recs if r["reason"] != "side request (no tools), not routed"]
    assert main and all(r["status"] == 200 for r in recs)
    assert all(r["routed_model"].startswith("claude-") for r in main)
    assert all(r.get("cost_usd") is not None for r in main)


def test_a_tool_loop_stays_on_the_model_its_turn_started_on(rig):
    before = len(rig.records())
    out = rig.claude("Use the Bash tool to run exactly: echo one && echo two. Then reply with just: done",
                     "--allowedTools", "Bash(echo one && echo two)")
    assert "done" in out.lower()
    turn = [r for r in new_records(rig, before) if r["reason"] != "side request (no tools), not routed"]
    assert len(turn) >= 2, "the tool call and the answer after it"
    assert len({r["routed_model"] for r in turn if r["conversation"] == turn[0]["conversation"]}) == 1


def test_a_lookup_subagent_runs_on_haiku(rig):
    agents = rig.gateway("agents").stdout
    lib = rig.project / "lib"
    lib.mkdir(exist_ok=True)
    (lib / "shapes.py").write_text("def area(w, h):\n    return w * h\n")
    (lib / "checksums.py").write_text("def widget_checksum(data):\n    return sum(data) % 255\n")
    before = len(rig.records())
    out = rig.claude("Use the lookup subagent to find which file under lib/ defines the function "
                     "widget_checksum. Reply with just the file name.", "--agents", agents)
    assert "checksums" in out
    sub = [r for r in new_records(rig, before) if r["subagent"]]
    assert sub and all(r["routed_model"] == "claude-haiku-5-5" for r in sub)
    assert all(r["agent_role"] == "lookup" for r in sub)


def test_the_status_line_reports_the_session(rig):
    session = next(r["claude_session"] for r in reversed(rig.records()) if r.get("claude_session"))
    line = subprocess.run([*CLI, "statusline"], env=rig.env, input=json.dumps({"session_id": session}),
                          capture_output=True, text=True, timeout=30).stdout
    assert line.strip() and "not running" not in line


def test_the_dashboard_and_stats_answer(rig):
    assert urllib.request.urlopen(f"{rig.url}/dashboard", timeout=5).status == 200
    totals = json.loads(urllib.request.urlopen(f"{rig.url}/stats", timeout=5).read())["totals"]
    assert totals["requests"] >= 3 and totals["errors"] == 0


def test_a_web_page_cannot_change_the_settings(rig):
    req = urllib.request.Request(f"{rig.url}/api/preferences", data=b'{"mode": "off"}', method="POST",
                                 headers={"content-type": "application/json", "origin": "https://evil.example"})
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req, timeout=5)
    assert err.value.code == 403


def test_stop_and_restart_leave_one_gateway_on_the_port(rig):
    first = json.loads(urllib.request.urlopen(f"{rig.url}/health", timeout=5).read())["pid"]
    assert rig.gateway("restart").returncode == 0 and rig.healthy()
    second = json.loads(urllib.request.urlopen(f"{rig.url}/health", timeout=5).read())["pid"]
    assert second != first
    with pytest.raises(ProcessLookupError):
        os.kill(first, 0)


def test_a_gateway_that_cannot_start_hands_the_session_to_anthropic_and_back(rig):
    rig.gateway("stop")
    rig.write_settings(config=rig.broken)  # the hook's start attempt fails at once
    assert "pong" in rig.claude("Reply with just the word: pong").lower()
    assert (rig.data / "bypassed").exists() and rig.project_base_url() is None
    rig.write_settings(config=rig.config, base_url=False)  # still bypassed, the gateway can start again
    # a bypassed session starts the gateway in the background without holding
    # the message, so routing comes back with a later message
    for _ in range(3):
        rig.claude("Reply with just the word: pong")
        if not (rig.data / "bypassed").exists():
            break
    assert rig.healthy() and not (rig.data / "bypassed").exists() and rig.project_base_url() == rig.url
    before = len(rig.records())
    assert "pong" in rig.claude("Reply with just the word: pong").lower()
    assert any(r["status"] == 200 for r in new_records(rig, before)), "routed through the gateway again"


def test_the_gateway_dying_mid_turn_does_not_end_the_turn(rig):
    rig.ensure_up()
    rig.write_settings()
    proc = subprocess.Popen(
        ["claude", "-p", "Use the Bash tool to run exactly: sleep 15 ; echo done. Then reply with just: finished",
         "--allowedTools", "Bash(sleep 15 ; echo done)"],
        cwd=rig.project, env={k: v for k, v in os.environ.items()
                              if not k.startswith("AAR_") and k != "ANTHROPIC_BASE_URL"},
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(8)  # inside the tool call
    rig.write_settings(config=rig.broken)
    rig.gateway("stop")
    out, _ = proc.communicate(timeout=300)
    assert "finished" in out.lower()
    assert (rig.data / "bypassed").exists()
    rig.write_settings()
    rig.ensure_up()
