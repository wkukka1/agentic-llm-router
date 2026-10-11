"""FastAPI process for :mod:`router.gateway` -- see that package's docstring
for what this exposes and why, and ``docs/gateway.md`` for the walkthrough.

``aar-gateway code`` starts the gateway if needed and launches Claude Code
through it; ``aar-gateway dashboard`` opens the dashboard; ``aar-gateway
serve`` runs it in the foreground; ``aar-gateway statusline`` is the Claude
Code status-line command. ``AAR_GATEWAY_HOST``/``AAR_GATEWAY_PORT`` override
the default ``127.0.0.1:8787``; ``AAR_GATEWAY_CONFIG`` points at a gateway
config (``configs/gateway.example.yaml``).
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import interceptor

_router_status: dict = {"enabled": False}


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Load eagerly rather than on the first request -- a bad config (missing
    API key, unknown lane, router run that won't load) should fail the
    process at boot, in the logs, not surface on someone's first real call."""
    from .config import load_settings
    from .routing import build_picker

    settings = load_settings(os.environ.get("AAR_GATEWAY_CONFIG"))
    use_graded = settings["router"].get("use_graded", True)
    picker = build_picker(settings, graded=_graded_predictor(settings),
                          graded_file=interceptor.log_path().parent / "graded.jsonl" if use_graded else None)
    interceptor.configure(settings, picker)
    rcfg = settings["router"]
    _router_status.update(enabled=picker is not None, kind=rcfg.get("kind"),
                          source=rcfg.get("data_config") if rcfg.get("kind") == "knn" else rcfg.get("run"),
                          models=sorted(settings["models"]))
    pid_file = _pid_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    try:
        yield
    finally:
        if pid_file.exists() and pid_file.read_text().strip() == str(os.getpid()):
            pid_file.unlink()
        await interceptor.aclose()


def _pid_file() -> Path:
    return interceptor.log_path().parent / "gateway.pid"


def _graded_predictor(settings: dict):
    """The user's graded turns (``aar-gateway-grade``), when the router is on --
    built here, not in :mod:`.routing`, so tests can inject their own."""
    rcfg = settings["router"]
    if not rcfg.get("enabled") or not rcfg.get("use_graded", True):
        return None
    from ..embeddings.encoder import EncoderConfig, TextEncoder
    from .graded import graded_predictor

    encoder = TextEncoder(EncoderConfig(name="gateway-graded"))
    return graded_predictor(interceptor.log_path().parent / "graded.jsonl", encoder.encode,
                            min_prompts=int(rcfg.get("min_graded_prompts", 20)))


app = FastAPI(title="AAR gateway", lifespan=_lifespan)


@app.get("/api/events")
async def events(request: Request):
    """Server-sent events for the dashboard: ``update`` the moment a request
    is logged, a comment every 15 s to keep the connection open."""
    import asyncio

    from fastapi.responses import StreamingResponse

    queue = interceptor.subscribe()

    async def stream():
        try:
            yield "retry: 3000\n\n"
            while not await request.is_disconnected():
                try:
                    await asyncio.wait_for(queue.get(), timeout=15)
                    yield "event: update\ndata: {}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            interceptor.unsubscribe(queue)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"cache-control": "no-cache"})


@app.get("/health")
async def health():
    status = dict(_router_status)
    picker = getattr(interceptor, "_picker", None)
    if picker is not None and hasattr(picker, "calibration_status"):
        status["calibration"] = picker.calibration_status()
    return {"status": "ok", "router": status}


@app.get("/stats")
async def stats():
    return interceptor.summarize()


@app.get("/api/statusline")
async def statusline(session_id: str = ""):
    return interceptor.session_summary(session_id)


@app.post("/api/subagent-spawn")
async def subagent_spawn(request: Request):
    """Asked by the ``PreToolUse`` hook (``aar-gateway hook``) before Claude
    Code spawns a subagent: which model it should run on."""
    body = await request.json()
    return interceptor.record_spawn(str(body.get("session_id") or ""), body.get("subagent_type"),
                                    str(body.get("prompt") or ""), body.get("model"))


@app.get("/api/preferences")
async def get_preferences():
    return interceptor.preferences()


@app.post("/api/preferences")
async def update_preferences(request: Request):
    body = await request.json()
    try:
        if "mode" in body:
            return interceptor.set_routing(body["mode"], body.get("pinned_model"))
        if "allow_upgrade" in body:
            return interceptor.set_allow_upgrade(bool(body["allow_upgrade"]))
        if "role" in body:
            return interceptor.set_role_model(body["role"], body.get("model"))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    margin = body.get("quality_margin")
    if body.get("preset") is not None:
        if body["preset"] not in interceptor.PRESETS:
            raise HTTPException(400, f"unknown preset {body['preset']!r}")
        margin = interceptor.PRESETS[body["preset"]]
    if margin is None:
        raise HTTPException(400, "send quality_margin or preset")
    try:
        return interceptor.set_quality_margin(margin)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return (Path(__file__).parent / "static" / "dashboard.html").read_text(encoding="utf-8")


def _no_credentials() -> JSONResponse:
    return JSONResponse({"type": "error", "error": {
        "type": "authentication_error",
        "message": "no credentials: the gateway forwards the client's own Anthropic login "
                   "(run Claude Code through it), or name a model on an `openai` lane"}},
        status_code=401)


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    routed = await interceptor.handle_messages(request)
    return routed if routed is not None else _no_credentials()


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def anthropic_passthrough(path: str, request: Request):
    """The rest of the Anthropic API Claude Code calls besides
    ``/v1/messages`` (e.g. ``/v1/messages/count_tokens``) -- forwarded on the
    client's own credentials; registered last so the routes above win."""
    if not interceptor.has_client_credentials(request):
        return _no_credentials()
    return await interceptor.forward(request, f"v1/{path}")


def _base_url() -> str:
    return f"http://{os.environ.get('AAR_GATEWAY_HOST', '127.0.0.1')}:{os.environ.get('AAR_GATEWAY_PORT', '8787')}"


def _src_env() -> dict:
    """This package importable in a child process even when site-packages
    ``.pth`` files are skipped -- macOS marks them hidden in iCloud-synced
    folders, and Python 3.13+ ignores hidden ``.pth`` files."""
    src = str(Path(__file__).resolve().parents[2])
    return {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))}


def _healthy(url: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(url + "/health", timeout=1) as resp:
            return resp.status == 200
    except OSError:
        return False


def _ensure_running(url: str) -> None:
    """Start the gateway in the background if nothing answers at ``url``."""
    import subprocess
    import time

    if _healthy(url):
        return
    log = interceptor.log_path().parent / "gateway.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as out:
        subprocess.Popen([sys.executable, "-m", "router.gateway.server", "serve"],
                         stdout=out, stderr=out, start_new_session=True, env=_src_env())
    print(f"starting the gateway at {url} (log: {log}) ...", file=sys.stderr)
    for _ in range(180):
        if _healthy(url):
            return
        time.sleep(0.5)
    raise SystemExit(f"the gateway did not come up within 90s -- see {log}")


def _user_has_statusline() -> bool:
    for path in (Path.home() / ".claude" / "settings.json", Path(".claude") / "settings.json",
                 Path(".claude") / "settings.local.json"):
        try:
            if json.loads(path.read_text()).get("statusLine"):
                return True
        except (OSError, ValueError):
            continue
    return False


def _claude_command(url: str, claude_args: list[str]) -> tuple[list[str], dict]:
    """``claude`` pointed at the gateway. Two settings matter:

    * ``ENABLE_TOOL_SEARCH=true`` -- for any base URL that isn't Anthropic's,
      Claude Code stops loading MCP tools on demand and sends every tool
      schema with every request (~245k tokens instead of ~40k with many MCP
      servers: several times the cost, and compaction every turn). The
      gateway forwards tool-search requests unchanged, so turn it back on.
    * a status line showing which model actually answered -- the model
      itself can't say, it is told it is whichever model was asked for.
      Skipped when the user already has a status line, or with
      ``AAR_GATEWAY_STATUSLINE=0``.

    It also gives Claude Code the subagent roles (``--agents``,
    :data:`~router.gateway.subagents.ROLES`), each pinned to its tier and
    given an output budget -- unless the user passes ``--agents`` themselves,
    or with ``AAR_GATEWAY_AGENTS=0``.
    """
    env = {"ENABLE_TOOL_SEARCH": "true", **os.environ, "ANTHROPIC_BASE_URL": url}
    args = ["claude"]
    src = str(Path(__file__).resolve().parents[2])
    run = f"PYTHONPATH={shlex.quote(src)} {shlex.quote(sys.executable)} -m router.gateway.server"
    settings: dict = {}
    if os.environ.get("AAR_GATEWAY_STATUSLINE", "1") != "0" and not _user_has_statusline():
        settings["statusLine"] = {"type": "command", "command": f"{run} statusline"}
    if os.environ.get("AAR_GATEWAY_HOOK", "1") != "0":
        # sets each subagent's model as Claude Code spawns it (see router.gateway.subagents)
        settings["hooks"] = {"PreToolUse": [{"matcher": "Agent|Task",
                                             "hooks": [{"type": "command", "command": f"{run} hook",
                                                        "timeout": 5}]}]}
    if settings:
        args += ["--settings", json.dumps(settings)]
    if os.environ.get("AAR_GATEWAY_AGENTS", "1") != "0" and "--agents" not in claude_args:
        from .subagents import agent_definitions

        args += ["--agents", json.dumps(agent_definitions())]
    return args + claude_args, env


def _agents(argv: list[str]) -> None:
    """``aar-gateway agents``: print the roles as ``--agents`` JSON;
    ``--install [DIR]``: write them as agent files (default
    ``~/.claude/agents``) so plain ``claude`` has them too, without the
    gateway. Existing files are left alone."""
    from .subagents import ROLES, agent_definitions, agent_markdown

    if argv[:1] != ["--install"]:
        print(json.dumps(agent_definitions(), indent=2))
        return
    target = Path(argv[1]).expanduser() if len(argv) > 1 else Path.home() / ".claude" / "agents"
    target.mkdir(parents=True, exist_ok=True)
    for name in ROLES:
        path = target / f"{name}.md"
        if path.exists():
            print(f"kept   {path} (already exists)")
            continue
        path.write_text(agent_markdown(name))
        print(f"wrote  {path}")


def _stop() -> None:
    """Stops the gateway this machine is running, whichever way it was
    started (``aar-gateway serve``, ``aar-gateway code``, ``python -m ...``)."""
    import signal
    import time
    import urllib.request

    pid_file = _pid_file()
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        print(f"no gateway running (no {pid_file})")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        print("no gateway running (stale pid file removed)")
        return
    for _ in range(50):
        try:
            urllib.request.urlopen(f"{_base_url()}/health", timeout=0.5)
        except OSError:
            print(f"stopped the gateway (pid {pid})")
            return
        time.sleep(0.2)
    print(f"sent stop to pid {pid}; it is still answering on {_base_url()}")


def _hook() -> None:
    """Claude Code's ``PreToolUse`` hook for the Agent tool: asks the gateway
    which model the subagent should run on and returns the call with
    ``model`` set. Anything else -- another tool, the gateway down, no
    decision -- prints nothing, which leaves the call exactly as it was."""
    import urllib.request

    try:
        event = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return
    if event.get("tool_name") not in ("Agent", "Task"):
        return
    call = dict(event.get("tool_input") or {})
    payload = {"session_id": event.get("session_id"), "subagent_type": call.get("subagent_type"),
               "prompt": call.get("prompt"), "model": call.get("model")}
    try:
        req = urllib.request.Request(f"{_base_url()}/api/subagent-spawn", data=json.dumps(payload).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            alias = json.loads(resp.read()).get("alias")
    except (OSError, ValueError):
        return
    if alias:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                                 "updatedInput": {**call, "model": alias}}}))


def _statusline() -> None:
    """Reads Claude Code's status-line JSON on stdin, prints one line."""
    import urllib.parse
    import urllib.request

    try:
        session = json.loads(sys.stdin.read() or "{}").get("session_id", "")
    except ValueError:
        session = ""
    try:
        query = urllib.parse.urlencode({"session_id": session})
        with urllib.request.urlopen(f"{_base_url()}/api/statusline?{query}", timeout=1) as resp:
            info = json.loads(resp.read())
    except (OSError, ValueError):
        print("aar gateway: not running")
        return
    print(info.get("line", "aar gateway"))


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    argv = sys.argv[1:] if argv is None else argv
    url = _base_url()
    if argv[:1] == ["code"]:
        # everything after `code` belongs to `claude`, flags included
        _ensure_running(url)
        args, env = _claude_command(url, argv[1:])
        os.execvpe("claude", args, env)
    if argv[:1] == ["statusline"]:
        _statusline()
        return
    if argv[:1] == ["stop"]:
        _stop()
        return
    if argv[:1] == ["hook"]:
        _hook()
        return
    if argv[:1] == ["agents"]:
        _agents(argv[1:])
        return

    parser = argparse.ArgumentParser(prog="aar-gateway")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="run the gateway in the foreground (the default)")
    sub.add_parser("code", help="run Claude Code through the gateway, starting it if needed; "
                                "any further arguments go to `claude`")
    sub.add_parser("dashboard", help="open the dashboard in a browser")
    sub.add_parser("stop", help="stop the running gateway")
    sub.add_parser("statusline", help="Claude Code status-line command (reads its JSON on stdin)")
    sub.add_parser("agents", help="print the subagent roles as --agents JSON; "
                                  "--install [DIR] writes them to ~/.claude/agents")
    args = parser.parse_args(argv)

    if args.command == "dashboard":
        import webbrowser

        _ensure_running(url)
        print(url + "/dashboard")
        webbrowser.open(url + "/dashboard")
        return

    import uvicorn

    uvicorn.run(app, host=os.environ.get("AAR_GATEWAY_HOST", "127.0.0.1"),
                port=int(os.environ.get("AAR_GATEWAY_PORT", "8787")))


if __name__ == "__main__":
    main()
