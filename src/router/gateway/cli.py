"""``aar-gateway``: run Claude Code through the gateway, and manage the
gateway process.

    aar-gateway code [claude args...]   Claude Code through the gateway (starts it if needed)
    aar-gateway start | stop | restart  the gateway in the background
    aar-gateway status                  running? router loaded? where are its files?
    aar-gateway logs [-n N] [-f]        the gateway's own log
    aar-gateway setup [--router FILE|URL]   first-time setup: the router + a config
    aar-gateway doctor                  check the setup, with a fix for each problem
    aar-gateway local [add|remove] M    optional: a local model through Ollama
    aar-gateway config                  which config is in use
    aar-gateway dashboard               open the dashboard
    aar-gateway service install|uninstall|status   start at login (launchd / systemd)
    aar-gateway serve                   the gateway in the foreground
    aar-gateway agents [--install DIR]  the subagent kinds as --agents JSON / agent files

``hook``, ``guard`` and ``statusline`` are called by Claude Code itself. They import
nothing but the standard library: the hook runs on every subagent spawn and
must not wait for the web server's imports.

Environment: ``AAR_GATEWAY_HOST`` / ``AAR_GATEWAY_PORT`` (default
``127.0.0.1:8787``), ``AAR_GATEWAY_CONFIG`` (default
``~/.aar-gateway/config.yaml`` when it exists), ``AAR_GATEWAY_LOG``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

SERVICE_LABEL = "com.aar.gateway"
#: the router ``aar-gateway setup`` installs when not given one: a trained
#: network (weights only, 0.4 MB) shipped inside the package.
#: ``AAR_GATEWAY_ROUTER_URL`` points setup at another file or URL instead.
BUNDLED_ROUTER = Path(__file__).with_name("routers") / "llmrouterbench-mlp.npz"
WINDOWS = os.name == "nt"


# --------------------------------------------------------------------------- #
# where things are                                                            #
# --------------------------------------------------------------------------- #
def _home() -> Path:
    log = os.environ.get("AAR_GATEWAY_LOG")
    return Path(log).expanduser().parent if log else Path.home() / ".aar-gateway"


def _config_path() -> Path | None:
    from .config import config_path

    return config_path()


def _pid_file() -> Path:
    return _home() / "gateway.pid"


def _log_file() -> Path:
    return _home() / "gateway.log"


def _base_url() -> str:
    return f"http://{os.environ.get('AAR_GATEWAY_HOST', '127.0.0.1')}:{os.environ.get('AAR_GATEWAY_PORT', '8787')}"


def _src() -> str:
    return str(Path(__file__).resolve().parents[2])


def _src_env() -> dict:
    """This package importable in a child process even when site-packages
    ``.pth`` files are skipped -- macOS marks them hidden in iCloud-synced
    folders, and Python 3.13+ ignores hidden ``.pth`` files."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [_src(), os.environ.get("PYTHONPATH")]))}


def _get(url: str, timeout: float = 1.0) -> dict | None:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (OSError, ValueError):
        return None


def _healthy(url: str) -> bool:
    return _get(url + "/health") is not None


def _running_pid() -> int | None:
    """The gateway's pid, if its pid file names a live process. On Windows
    ``os.kill(pid, 0)`` would *terminate* the process (any signal but Ctrl-C/
    Ctrl-Break is TerminateProcess there), so liveness is the health check."""
    try:
        pid = int(_pid_file().read_text().strip())
    except (OSError, ValueError):
        return None
    if WINDOWS:
        return pid if _healthy(_base_url()) else None
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# process management                                                          #
# --------------------------------------------------------------------------- #
def _spawn() -> subprocess.Popen:
    """The gateway, started in the background; returns without waiting."""
    log = _log_file()
    log.parent.mkdir(parents=True, exist_ok=True)
    # detached, so it outlives this command and the terminal it ran in
    detach = ({"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
              if WINDOWS else {"start_new_session": True})
    with log.open("ab") as out:
        return subprocess.Popen([sys.executable, "-m", "router.gateway.cli", "serve"],
                                stdout=out, stderr=out, stdin=subprocess.DEVNULL, env=_src_env(), **detach)


def _ensure_running(url: str, *, quiet: bool = False, wait: float = 90) -> bool:
    """Start the gateway in the background if nothing answers at ``url``.
    ``True`` if this call started it. Gives up after ``wait`` seconds, or as
    soon as the new process exits (a crash on start, or another gateway
    already starting on the port) and nothing answers."""
    if _healthy(url):
        return False
    proc = _spawn()
    if not quiet:
        print(f"starting the gateway at {url} (log: {_log_file()}) ...", file=sys.stderr)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if _healthy(url):
            return True
        if proc.poll() is not None:
            time.sleep(1)  # the port's owner may have been mid-start
            if _healthy(url):
                return True
            raise SystemExit("the gateway exited while starting -- see `aar-gateway logs`")
        time.sleep(0.5)
    raise SystemExit(f"the gateway did not come up within {wait:g}s -- see `aar-gateway logs`")


def _stop() -> bool:
    """Stops the gateway this machine is running, however it was started.
    ``True`` if one was stopped."""
    import signal

    pid = _running_pid()
    if pid is None:
        _pid_file().unlink(missing_ok=True)
        print("the gateway is not running")
        return False
    os.kill(pid, signal.SIGTERM)  # on Windows: TerminateProcess
    for _ in range(50):
        if not _healthy(_base_url()):
            _pid_file().unlink(missing_ok=True)  # a terminated process can't remove its own
            print(f"stopped the gateway (pid {pid})")
            return True
        time.sleep(0.2)
    print(f"sent stop to pid {pid}; it is still answering on {_base_url()}")
    return False


def _status(as_json: bool = False) -> int:
    url = _base_url()
    health = _get(url + "/health")
    pid = _running_pid()
    cfg = _config_path()
    info = {"running": health is not None, "url": url, "pid": pid, "config": str(cfg) if cfg else None,
            "log": str(_log_file()), "requests_log": str(_home() / "requests.jsonl"),
            "dashboard": url + "/dashboard"}
    if health:
        router = health.get("router") or {}
        info["router"] = {"enabled": router.get("enabled"), "error": router.get("error"),
                          "kind": router.get("kind")}
        stats = _get(url + "/stats", timeout=3) or {}
        totals = stats.get("totals") or {}
        info["requests"] = totals.get("requests")
        base = totals.get("cost_if_requested_usd") or 0
        info["plan_usage_saved"] = round(1 - totals.get("cost_usd", 0) / base, 3) if base else None
        if pid:
            info["up_since"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(_pid_file().stat().st_mtime))
    if as_json:
        print(json.dumps(info, indent=2))
        return 0 if health else 1
    if not health:
        print(f"● stopped        nothing answers at {url}")
        print(f"  config         {info['config'] or 'none (defaults; router off)'}")
        print(f"  log            {info['log']}")
        print("  start it with  aar-gateway start   (or aar-gateway code)")
        return 1
    router = info["router"]
    state = ("on" if router["enabled"] else f"FAILED TO LOAD -- {router['error']}" if router["error"]
             else "off (enable it in the config)")
    saved = info.get("plan_usage_saved")
    print(f"● running        {url}" + (f"  (pid {pid}, since {info.get('up_since')})" if pid else ""))
    print(f"  router         {state}")
    print(f"  config         {info['config'] or 'none (defaults)'}")
    print(f"  requests       {info.get('requests', 0)}" +
          (f"  ·  {saved:.0%} less plan usage than without the router" if saved is not None else ""))
    print(f"  dashboard      {info['dashboard']}")
    print(f"  logs           {info['log']}")
    return 0


def _text(raw: bytes) -> str:
    """Log bytes as text; Windows writes \\r\\n, which print() would double up."""
    return raw.decode(errors="replace").replace("\r\n", "\n")


def _logs(lines: int, follow: bool) -> int:
    log = _log_file()
    if not log.exists():
        print(f"no log yet ({log})")
        return 1
    with log.open("rb") as f:
        tail = f.readlines()[-lines:] if lines > 0 else []
    sys.stdout.write(_text(b"".join(tail)))
    if not follow:
        return 0
    with log.open("rb") as f:
        f.seek(0, os.SEEK_END)
        try:
            while True:
                chunk = f.readline()
                if chunk:
                    sys.stdout.write(_text(chunk))
                    sys.stdout.flush()
                else:
                    time.sleep(0.3)
        except KeyboardInterrupt:
            return 0


# --------------------------------------------------------------------------- #
# local models (opt-in)                                                       #
# --------------------------------------------------------------------------- #
OLLAMA_URL = os.environ.get("AAR_GATEWAY_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_INSTALL = {
    "darwin": "brew install ollama && brew services start ollama\n"
              "  (or download the app from https://ollama.com/download and open it)",
    "linux": "curl -fsSL https://ollama.com/install.sh | sh",
    "win32": "download and run the installer from https://ollama.com/download",
}


def _ollama_models() -> list[str] | None:
    """Models Ollama has pulled, or ``None`` when Ollama isn't running."""
    tags = _get(f"{OLLAMA_URL}/api/tags", timeout=2)
    return None if tags is None else [m.get("name", "") for m in tags.get("models") or []]


def _ollama_name(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


def _write_local_models(models: dict) -> None:
    import yaml

    from .config import local_models_path

    path = local_models_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not models:
        path.unlink(missing_ok=True)
        return
    path.write_text("# Local models added with `aar-gateway local add` (Ollama). Remove one with\n"
                    "# `aar-gateway local remove <model>`, or delete this file to remove them all.\n"
                    + yaml.safe_dump({"models": models}, sort_keys=False))


def _local(argv: list[str]) -> int:
    """``aar-gateway local [add|remove] [model]``: opt-in local models via Ollama."""
    from .config import local_models

    action = argv[0] if argv else "status"
    configured = local_models()
    pulled = _ollama_models()
    installed = shutil.which("ollama") is not None or pulled is not None
    if action == "status":
        print(f"Ollama: {'running' if pulled is not None else 'installed, not running' if installed else 'not installed'}")
        if pulled:
            print("  models it has: " + ", ".join(pulled))
        print("local models in the gateway: " + (", ".join(configured) if configured else "none"))
        if not installed:
            print(f"\nto install Ollama: {OLLAMA_INSTALL.get(sys.platform, OLLAMA_INSTALL['linux'])}")
        print("\nadd one: aar-gateway local add llama3.1   (good at tool calls; 4.9 GB)")
        return 0
    if action not in ("add", "remove") or len(argv) < 2:
        raise SystemExit("usage: aar-gateway local [add|remove] <model>   (e.g. llama3.1)")
    name = _ollama_name(argv[1])
    if action == "remove":
        if name not in configured:
            print(f"{name} isn't a local model in the gateway")
            return 1
        del configured[name]
        _write_local_models(configured)
        print(f"removed {name} from the gateway (Ollama still has it: `ollama rm {name}` frees the disk)")
    else:
        if pulled is None:
            raise SystemExit("Ollama isn't running. " + (
                "Start it (open the Ollama app, or run `ollama serve`)." if installed else
                f"Install it first:\n  {OLLAMA_INSTALL.get(sys.platform, OLLAMA_INSTALL['linux'])}"))
        if name not in pulled:
            ollama = shutil.which("ollama")
            if ollama is None:
                raise SystemExit(f"Ollama doesn't have {name} yet and its command isn't on PATH: run `ollama pull {name}`")
            print(f"downloading {name} with Ollama (one-time) ...")
            if subprocess.run([ollama, "pull", name]).returncode != 0:
                raise SystemExit(f"`ollama pull {name}` failed")
        configured[name] = {"lane": "openai", "base_url": f"{OLLAMA_URL}/v1", "proxy": None,
                            "context_window": 32768, "max_tools": 64, "prices": {"input": 0, "output": 0}}
        _write_local_models(configured)
        print(f"added {name}. The router never picks it on its own; use it by\n"
              f"  - pinning it on the dashboard (Advanced settings -> Pin a model), or\n"
              f"  - aar-gateway code --model {name}")
    if _healthy(_base_url()):
        _stop()
        _ensure_running(_base_url(), quiet=True)
        print("restarted the gateway")
    return 0


# --------------------------------------------------------------------------- #
# setup                                                                       #
# --------------------------------------------------------------------------- #
CONFIG_TEMPLATE = """# AAR gateway config, written by `aar-gateway setup`.
# Every option is explained in configs/gateway.example.yaml in the repository,
# and in docs/gateway.md.
interceptor:
  # true: keep the messages you type (locally, never sent anywhere) so
  # `aar-gateway-grade` can learn which model suits your own work
  capture_prompts: false
  router:
    enabled: {enabled}
    kind: knn
    artifact: {artifact}      # the router file
    margin: 0.05              # cost preference; change it from the dashboard
"""


def _fetch_router(source: str, target: Path) -> dict:
    """Copy or download a router file to ``target``, checking it loads.
    Returns its header."""
    from ..models.artifacts import load_router_file

    tmp = target.with_suffix(".part")
    if source.startswith(("http://", "https://")):
        import urllib.request

        print(f"downloading the router from {source} ...")
        with urllib.request.urlopen(source, timeout=120) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
    else:
        path = Path(source).expanduser()
        if not path.exists():
            raise SystemExit(f"no router file at {path}")
        shutil.copyfile(path, tmp)
    try:
        payload, header = load_router_file(tmp)
    except Exception as exc:  # noqa: BLE001 -- anything unreadable is not a router file
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{source} is not a router file this gateway can load: {exc}") from exc
    tmp.replace(target)
    kind = "trained network" if header.get("format") == "mlp" else f"kNN, k={header.get('k')}"
    print(f"router installed: {target} ({kind}, {len(payload.model_ids)} models"
          + (f", {header['dataset']}" if header.get("dataset") else "") + ")")
    return header


def _setup(router: str | None, force: bool, enable: bool | None = False) -> int:
    """Everything a new user needs: the router file and a config that uses it."""
    home = _home()
    home.mkdir(parents=True, exist_ok=True)
    router_file = home / "router.npz"
    source = router or os.environ.get("AAR_GATEWAY_ROUTER_URL") or (
        str(BUNDLED_ROUTER) if BUNDLED_ROUTER.exists() else "")
    if source:
        _fetch_router(source, router_file)
    elif router_file.exists():
        print(f"router already installed: {router_file}")
    else:
        print("no router file to install yet: the gateway will forward every request to the model you\n"
              "asked for until you run `aar-gateway setup --router <file or URL>`.")
    config = home / "config.yaml"
    if config.exists() and not force:
        print(f"kept your config: {config}  (--force rewrites it)")
    else:
        config.write_text(CONFIG_TEMPLATE.format(enabled="true" if router_file.exists() else "false",
                                                 artifact=json.dumps(str(router_file))))
        print(f"wrote {config}")
    if _healthy(_base_url()):
        print("restarting the running gateway so it picks this up ...")
        _stop()
        _ensure_running(_base_url(), quiet=True)
    print()
    if auto_enabled():
        print("plain `claude` already goes through the gateway (`aar-gateway disable` to undo)")
    elif enable is None and sys.stdin.isatty():
        # opt in: only a yes changes Claude Code's settings
        answer = input("Also use the gateway in every Claude Code session, so plain `claude` saves too?\n"
                       "(otherwise start Claude Code with `aar-gateway code`) [y/N] ")
        enable = answer.strip().lower() in ("y", "yes")
    if enable and not auto_enabled():
        try:
            _enable()
        except SystemExit as exc:  # another base URL already set: say so, finish setup
            print(exc)
    elif not auto_enabled():
        print("plain `claude` is unchanged: start Claude Code with `aar-gateway code`\n"
              "(or run `aar-gateway enable` once to use the gateway in every session)")
    print()
    return _doctor()


# --------------------------------------------------------------------------- #
# doctor                                                                      #
# --------------------------------------------------------------------------- #
def _doctor() -> int:
    """Checks, each with a fix: the claude CLI, the config, the router, the
    port, the log folder."""
    problems = 0

    def report(ok: bool | None, what: str, detail: str = "", fix: str = "") -> None:
        nonlocal problems
        mark = {True: "✓", False: "✗", None: "!"}[ok]
        problems += ok is False
        print(f"{mark} {what}" + (f": {detail}" if detail else ""))
        if fix and ok is not True:
            print(f"    fix: {fix}")

    claude = shutil.which("claude")
    if claude:
        version = subprocess.run([claude, "--version"], capture_output=True, text=True, timeout=20).stdout.strip()
        report(True, "Claude Code installed", version)
    else:
        report(False, "Claude Code installed", "`claude` is not on PATH",
               "install Claude Code (https://claude.com/claude-code) and run `claude` once to log in")

    cfg = _config_path()
    settings = None
    try:
        from .config import load_settings

        settings = load_settings(cfg)
        report(True if cfg else None, "config", str(cfg) if cfg else "none found, using the defaults (router off)",
               f"put one at {_home() / 'config.yaml'} (see configs/gateway.example.yaml) or set AAR_GATEWAY_CONFIG")
    except Exception as exc:  # noqa: BLE001 -- doctor reports whatever is wrong
        report(False, "config", f"{cfg}: {exc}", "fix the error above, or move the file away to use the defaults")

    artifact = (settings or {}).get("router", {}).get("artifact")
    if artifact:
        path = Path(artifact).expanduser()
        report(path.exists(), "router file", str(path) if path.exists() else f"{path} is missing",
               "aar-gateway setup --router <file or URL>")

    url = _base_url()
    health = _get(url + "/health")
    if health:
        router = health.get("router") or {}
        if router.get("enabled"):
            report(True, "router", f"{router.get('kind')} loaded from {router.get('source')}")
        elif router.get("error"):
            report(False, "router", router["error"],
                   "check router.data_config in the config points at the benchmark data, then `aar-gateway restart`")
        else:
            report(None, "router", "off: requests go through unrouted", "set interceptor.router.enabled: true")
        report(True, "gateway running", url)
    else:
        if settings and settings["router"].get("enabled"):
            report(None, "router", "enabled in the config; it loads when the gateway starts",
                   "run `aar-gateway start`, then `aar-gateway doctor` again to check it loaded")
        import socket

        with socket.socket() as sock:
            sock.settimeout(0.5)
            taken = sock.connect_ex((os.environ.get("AAR_GATEWAY_HOST", "127.0.0.1"),
                                     int(os.environ.get("AAR_GATEWAY_PORT", "8787")))) == 0
        if taken:
            report(False, "port", f"something else is listening on {url}",
                   "stop it, or pick another port with AAR_GATEWAY_PORT")
        else:
            report(None, "gateway running", "not running", "aar-gateway start")

    home = _home()
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        report(True, "log folder writable", str(home))
    except OSError as exc:
        report(False, "log folder writable", f"{home}: {exc}", "set AAR_GATEWAY_LOG to a writable path")

    if not auto_enabled():
        report(None, "every Claude Code session", "off: only `aar-gateway code` uses the gateway",
               "aar-gateway enable")
    elif _suspended() and not _healthy(url):
        report(False, "every Claude Code session", "on, but the gateway is down, so `claude` talks to Anthropic "
               "directly (no routing) until it's back", "aar-gateway start  (and `aar-gateway logs` for why)")
    elif _healthy(url):
        report(True, "every Claude Code session", "plain `claude` goes through the gateway")
    else:
        report(False, "every Claude Code session", "on, but the gateway isn't running, so `claude` can't connect",
               "aar-gateway start  (or `aar-gateway disable` to talk to Anthropic directly)")

    print("\nall good" if problems == 0 else f"\n{problems} problem(s) to fix")
    return 0 if problems == 0 else 1


# --------------------------------------------------------------------------- #
# start at login                                                              #
# --------------------------------------------------------------------------- #
def _service_env() -> dict[str, str]:
    env = {"PYTHONPATH": _src()}
    cfg = _config_path()
    if cfg:
        env["AAR_GATEWAY_CONFIG"] = str(cfg.resolve())
    for key in ("AAR_GATEWAY_HOST", "AAR_GATEWAY_PORT", "AAR_GATEWAY_LOG", "AAR_GATEWAY_ALLOWED_HOSTS"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def launchd_plist() -> str:
    """macOS: a LaunchAgent that starts the gateway at login and restarts it
    if it crashes -- but not after ``aar-gateway stop`` (a clean exit)."""
    import plistlib

    log = str(_log_file())
    return plistlib.dumps({
        "Label": SERVICE_LABEL,
        "ProgramArguments": [sys.executable, "-m", "router.gateway.cli", "serve"],
        "EnvironmentVariables": _service_env(),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": log,
        "StandardErrorPath": log,
    }).decode()


def systemd_unit() -> str:
    """Linux: a user unit, restarted on failure but not after a clean stop."""
    env = "\n".join(f"Environment={shlex.quote(f'{k}={v}')}" for k, v in _service_env().items())
    return (f"[Unit]\nDescription=AAR gateway (routes Claude Code)\n\n[Service]\n"
            f"ExecStart={shlex.quote(sys.executable)} -m router.gateway.cli serve\n{env}\n"
            f"Restart=on-failure\n\n[Install]\nWantedBy=default.target\n")


def _service_paths() -> tuple[str, Path]:
    if sys.platform == "darwin":
        return "launchd", Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_LABEL}.plist"
    if sys.platform.startswith("linux"):
        return "systemd", Path.home() / ".config" / "systemd" / "user" / "aar-gateway.service"
    raise SystemExit(f"start-at-login isn't supported on {sys.platform}; run `aar-gateway start` instead")


def _service(action: str, run=subprocess.run) -> int:
    kind, path = _service_paths()
    if action == "status":
        print(f"{'installed' if path.exists() else 'not installed'} ({path})")
        return 0 if path.exists() else 1
    if action == "install":
        if _running_pid():
            _stop()  # the service starts its own
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(launchd_plist() if kind == "launchd" else systemd_unit())
        if kind == "launchd":
            run(["launchctl", "bootout", f"gui/{os.getuid()}/{SERVICE_LABEL}"], capture_output=True)
            run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True)
        else:
            run(["systemctl", "--user", "daemon-reload"], check=True)
            run(["systemctl", "--user", "enable", "--now", "aar-gateway.service"], check=True)
        print(f"installed: the gateway now starts at login ({path})")
        return 0
    if action == "uninstall":
        if not path.exists():
            print("not installed")
            return 0
        if kind == "launchd":
            run(["launchctl", "bootout", f"gui/{os.getuid()}/{SERVICE_LABEL}"], capture_output=True)
        else:
            run(["systemctl", "--user", "disable", "--now", "aar-gateway.service"], capture_output=True)
        path.unlink()
        print(f"uninstalled: removed {path}; the gateway no longer starts at login")
        return 0
    raise SystemExit(f"unknown service action {action!r}")


# --------------------------------------------------------------------------- #
# Claude Code                                                                 #
# --------------------------------------------------------------------------- #
def _self_command() -> str:
    """How Claude Code calls back into this package (status line, hook): this
    Python, ``-m router.gateway.cli``. Installed (uv, pip) that's enough; from
    a source checkout on macOS/Linux, PYTHONPATH points at it too -- Windows
    shells don't take the ``VAR=value cmd`` form, and there the package is
    expected to be installed."""
    if WINDOWS:
        return subprocess.list2cmdline([sys.executable, "-m", "router.gateway.cli"])
    command = f"{shlex.quote(sys.executable)} -m router.gateway.cli"
    if "site-packages" in _src():
        return command
    return f"PYTHONPATH={shlex.quote(_src())} {command}"


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

    It also installs the subagent hook (``AAR_GATEWAY_HOOK=0`` to skip) and
    gives Claude Code the subagent kinds (``--agents``,
    :data:`~router.gateway.subagents.ROLES`) -- unless the user passes
    ``--agents`` themselves, or with ``AAR_GATEWAY_AGENTS=0``.
    """
    env = {"ENABLE_TOOL_SEARCH": "true", **os.environ, "ANTHROPIC_BASE_URL": url}
    args = ["claude"]
    run = _self_command()
    settings: dict = {}
    enabled = auto_enabled()  # the user settings already have the hook and status line
    if not enabled and os.environ.get("AAR_GATEWAY_STATUSLINE", "1") != "0" and not _user_has_statusline():
        settings["statusLine"] = {"type": "command", "command": f"{run} statusline"}
    if not enabled and os.environ.get("AAR_GATEWAY_HOOK", "1") != "0":
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


# --------------------------------------------------------------------------- #
# plain `claude`, every session (enable / disable)                            #
# --------------------------------------------------------------------------- #
def _claude_settings_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def _ours(entry: dict) -> bool:
    """A hook entry this package wrote: its command runs ``router.gateway.cli``."""
    return any("router.gateway.cli" in h.get("command", "") for h in entry.get("hooks", []))


def _read_claude_settings(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except ValueError as exc:
        raise SystemExit(f"{path} isn't valid JSON ({exc}); fix it, then run this again") from None


def _write_claude_settings(path: Path, settings: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".aar-tmp")
    tmp.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def auto_enabled() -> bool:
    """Whether plain ``claude`` goes through this gateway (``aar-gateway enable``),
    counting a session that is on Anthropic directly while the gateway is down."""
    if _suspended():
        return True
    try:
        settings = _read_claude_settings(_claude_settings_path())
    except SystemExit:
        return False
    return (settings.get("env") or {}).get("ANTHROPIC_BASE_URL") == _base_url()


def _temporary_install() -> str | None:
    """The temporary folder this copy of the package runs from, if it does:
    ``enable`` would make every Claude Code session depend on a folder the
    system can delete (seen live: a test install under /private/tmp left
    every session failing with "connection refused")."""
    import tempfile

    roots = set()
    for root in (tempfile.gettempdir(), "/tmp", "/private/tmp", "/var/tmp"):
        roots |= {Path(root), Path(root).resolve()}
    exe = Path(sys.executable).absolute()  # a venv's python is a symlink to the real one: not resolved
    for place in (exe.parent, Path(_src())):
        for root in roots:
            if place.is_relative_to(root):
                return str(place)
    return None


# --------------------------------------------------------------------------- #
# when the gateway is down: talk to Anthropic directly until it's back        #
# --------------------------------------------------------------------------- #
#: Without this, plain `claude` sends every request to a gateway that isn't
#: there and fails with "connection refused" -- and can't even run a command
#: to fix it. Claude Code's hooks check the gateway as a session starts and
#: before each message: down, its address comes out of the user settings (so
#: Claude Code talks to Anthropic directly) and a restart is tried in the
#: background; back up, the address goes back in.
RETRY_START_SECONDS = 120


def _suspend_file() -> Path:
    return _home() / "suspended.json"


def _suspended() -> dict | None:
    try:
        return json.loads(_suspend_file().read_text())
    except (OSError, ValueError):
        return None


def _mark_suspended(**fields) -> None:
    path = _suspend_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**(_suspended() or {}), **fields}))


def _suspend() -> bool:
    """Takes the gateway's address out of Claude Code's user settings, if it
    is there; ``True`` if it was."""
    path = _claude_settings_path()
    try:
        settings = _read_claude_settings(path)
    except SystemExit:
        return False
    env = settings.get("env") or {}
    if env.get("ANTHROPIC_BASE_URL") != _base_url():
        return False
    _mark_suspended(url=_base_url(), since=time.time())
    del env["ANTHROPIC_BASE_URL"]
    _write_claude_settings(path, settings)
    return True


def _resume() -> bool:
    """Puts the gateway's address back after :func:`_suspend`; ``True`` if it
    did. If the user has pointed Claude Code somewhere else since, that stays."""
    if not _suspended():
        return False
    path = _claude_settings_path()
    try:
        settings = _read_claude_settings(path)
    except SystemExit:
        return False
    env = settings.setdefault("env", {})
    _suspend_file().unlink(missing_ok=True)
    if env.get("ANTHROPIC_BASE_URL"):
        return False
    env["ANTHROPIC_BASE_URL"] = _base_url()
    _write_claude_settings(path, settings)
    return True


def _say(message: str) -> None:
    """A hook's note to the user: shown in Claude Code, not added to the conversation."""
    print(json.dumps({"systemMessage": message}))


def _guard(wait: float = 0) -> None:
    """Claude Code's SessionStart (``wait`` > 0: start the gateway and wait for
    it) and UserPromptSubmit (no waiting) hook. Prints nothing unless it
    switched the session one way or the other."""
    url = _base_url()
    up = _healthy(url)
    if not up and wait:
        try:
            _ensure_running(url, quiet=True, wait=wait)
            up = True
        except SystemExit:
            pass
    if up:
        if _resume():
            _say(f"aar gateway is back at {url}: routing is on again.")
        return
    switched = _suspend()
    state = _suspended()
    if state is None:  # Claude Code isn't pointed at this gateway by `enable`: nothing to do
        return
    if wait:  # the start above was this attempt
        _mark_suspended(tried=time.time())
    elif _running_pid() is None and time.time() - state.get("tried", 0) > RETRY_START_SECONDS:
        try:
            _spawn()  # in the background; a later message switches back once it answers
            _mark_suspended(tried=time.time())
        except OSError:
            pass
    if switched:
        _say(f"aar gateway isn't answering at {url}, so Claude Code is talking to Anthropic directly "
             "(no routing) until it's back. `aar-gateway logs` shows why it stopped.")


def _enable(force: bool = False) -> int:
    """Every ``claude`` session goes through the gateway, no command needed:
    Claude Code's user settings get the gateway's address, a SessionStart
    hook that starts the gateway when it isn't running, the subagent hook and
    the status line; the subagent kinds go to ``~/.claude/agents``; on macOS
    and Linux the gateway also starts at login. Only what's missing is added,
    and ``aar-gateway disable`` takes exactly that out again."""
    temp = _temporary_install()
    if temp and not force:
        raise SystemExit(f"This aar-gateway runs from a temporary folder ({temp}).\n"
                         "Every Claude Code session would depend on it, and the system can delete that folder.\n"
                         "Install it for good first (`uv tool install \"agentic-llm-router[gateway] @ "
                         "git+https://github.com/wkukka1/agentic-llm-router\"`) and run `aar-gateway enable` "
                         "from there, or pass --force.")
    path = _claude_settings_path()
    settings = _read_claude_settings(path)
    url, run = _base_url(), _self_command()
    env = settings.setdefault("env", {})
    current = env.get("ANTHROPIC_BASE_URL")
    if current and current != url and not force:
        raise SystemExit(f"Claude Code already sends requests to {current} (ANTHROPIC_BASE_URL in {path}).\n"
                         f"Run `aar-gateway enable --force` to point it at the gateway instead.")
    env["ANTHROPIC_BASE_URL"] = url
    env.setdefault("ENABLE_TOOL_SEARCH", "true")  # see _claude_command
    hooks = settings.setdefault("hooks", {})
    wanted = {
        # both print nothing unless the gateway is down (_guard): a hook's plain
        # output would become part of the conversation
        "SessionStart": {"hooks": [{"type": "command", "command": f"{run} start --quiet", "timeout": 90}]},
        "UserPromptSubmit": {"hooks": [{"type": "command", "command": f"{run} guard", "timeout": 5}]},
        "PreToolUse": {"matcher": "Agent|Task",
                       "hooks": [{"type": "command", "command": f"{run} hook", "timeout": 5}]},
    }
    for event, entry in wanted.items():
        hooks[event] = [e for e in hooks.get(event, []) if not _ours(e)] + [entry]
    if not settings.get("statusLine"):
        settings["statusLine"] = {"type": "command", "command": f"{run} statusline"}
    if path.exists() and not path.with_name("settings.json.aar-backup").exists():
        shutil.copy2(path, path.with_name("settings.json.aar-backup"))
    _write_claude_settings(path, settings)
    _suspend_file().unlink(missing_ok=True)
    print(f"plain `claude` now goes through the gateway in every session ({path})")
    _agents(["--install"])
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        try:
            _service("install")
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"couldn't register the login service ({exc}); each session starts the gateway instead")
    else:
        _ensure_running(url, quiet=True)
    print("undo with `aar-gateway disable`")
    return 0


def _disable() -> int:
    """Takes out what :func:`_enable` added; the rest of the settings stay."""
    path = _claude_settings_path()
    settings = _read_claude_settings(path)
    env = settings.get("env") or {}
    ours = env.get("ANTHROPIC_BASE_URL") == _base_url()
    if ours or (_suspended() and not env.get("ANTHROPIC_BASE_URL")):  # suspended: the address is out already
        env.pop("ANTHROPIC_BASE_URL", None)
        if env.get("ENABLE_TOOL_SEARCH") == "true":
            del env["ENABLE_TOOL_SEARCH"]
    if not env:
        settings.pop("env", None)
    hooks = settings.get("hooks") or {}
    for event in list(hooks):
        hooks[event] = [e for e in hooks[event] if not _ours(e)]
        if not hooks[event]:
            del hooks[event]
    if not hooks:
        settings.pop("hooks", None)
    if "router.gateway.cli" in (settings.get("statusLine") or {}).get("command", ""):
        del settings["statusLine"]
    if path.exists():
        _write_claude_settings(path, settings)
    _suspend_file().unlink(missing_ok=True)
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        _service("uninstall")
    print("plain `claude` talks to Anthropic directly again; `aar-gateway code` still uses the gateway")
    return 0


def _agents(argv: list[str]) -> None:
    """``aar-gateway agents``: print the kinds as ``--agents`` JSON;
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

    try:
        session = json.loads(sys.stdin.read() or "{}").get("session_id", "")
    except ValueError:
        session = ""
    info = _get(f"{_base_url()}/api/statusline?{urllib.parse.urlencode({'session_id': session})}")
    if info:
        print(info.get("line", "aar gateway"))
    else:
        print("aar gateway: down, talking to Anthropic directly" if _suspended() else "aar gateway: not running")


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    url = _base_url()
    # Claude Code's own calls, and `code` (whose arguments belong to `claude`):
    # handled before argparse, with no heavy imports
    if argv[:1] == ["code"]:
        _ensure_running(url)
        args, env = _claude_command(url, argv[1:])
        claude = shutil.which("claude", path=env.get("PATH"))
        if claude is None:
            raise SystemExit("`claude` is not on PATH -- install Claude Code first (see `aar-gateway doctor`)")
        if WINDOWS:  # no exec() there: run it and pass its exit code on
            return subprocess.call([claude, *args[1:]], env=env)
        os.execve(claude, [claude, *args[1:]], env)
    if argv[:1] == ["hook"]:
        _hook()
        return 0
    if argv[:1] == ["statusline"]:
        _statusline()
        return 0
    if argv[:1] == ["guard"]:
        _guard()
        return 0
    if argv[:1] == ["agents"]:
        _agents(argv[1:])
        return 0
    if argv[:1] == ["local"]:
        return _local(argv[1:])

    parser = argparse.ArgumentParser(prog="aar-gateway", description="Route Claude Code through the AAR gateway.")
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.add_parser("code", help="run Claude Code through the gateway, starting it if needed "
                                "(further arguments go to `claude`)")
    start = sub.add_parser("start", help="start the gateway in the background")
    start.add_argument("--quiet", action="store_true", help="print nothing (Claude Code's SessionStart hook)")
    sub.add_parser("stop", help="stop the gateway")
    sub.add_parser("restart", help="stop, then start the gateway")
    status = sub.add_parser("status", help="is it running, is the router loaded, where are its files")
    status.add_argument("--json", action="store_true")
    logs = sub.add_parser("logs", help="the gateway's log")
    logs.add_argument("-n", type=int, default=50, help="lines to show (default 50)")
    logs.add_argument("-f", "--follow", action="store_true", help="keep printing new lines")
    setup = sub.add_parser("setup", help="install the router and write a config (first-time setup)")
    setup.add_argument("--router", help="a router file or URL (default: the one shipped with the package)")
    setup.add_argument("--force", action="store_true", help="rewrite an existing config")
    every = setup.add_mutually_exclusive_group()
    every.add_argument("--yes", action="store_true", help="use the gateway in every `claude` session without asking")
    every.add_argument("--no-enable", action="store_true",
                       help="leave plain `claude` alone (use `aar-gateway code` instead)")
    enable = sub.add_parser("enable", help="every `claude` session goes through the gateway, no command needed")
    enable.add_argument("--force", action="store_true", help="replace another ANTHROPIC_BASE_URL in your settings, "
                                                             "or enable from a temporary install")
    sub.add_parser("disable", help="undo `enable`: plain `claude` talks to Anthropic directly again")
    sub.add_parser("doctor", help="check the setup, with a fix for each problem")
    sub.add_parser("local", help="optional local models via Ollama: local [add|remove] <model>")
    sub.add_parser("config", help="which config file is in use")
    sub.add_parser("dashboard", help="open the dashboard in a browser")
    service = sub.add_parser("service", help="start the gateway at login (launchd on macOS, systemd on Linux)")
    service.add_argument("action", choices=("install", "uninstall", "status"))
    sub.add_parser("serve", help="run the gateway in the foreground")
    sub.add_parser("agents", help="the subagent kinds as --agents JSON; --install [DIR] writes agent files")
    sub.add_parser("statusline", help="Claude Code's status-line command (reads JSON on stdin)")
    sub.add_parser("hook", help="Claude Code's PreToolUse hook for subagents (reads JSON on stdin)")
    sub.add_parser("guard", help="Claude Code's UserPromptSubmit hook: talk to Anthropic directly while the "
                                 "gateway is down")
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "start":
        if args.quiet:  # Claude Code's SessionStart hook
            _guard(wait=45)
            return 0
        started = _ensure_running(url)
        print(("started" if started else "already running") + f": {url}  (dashboard {url}/dashboard)")
        return 0
    if args.command == "stop":
        _stop()
        return 0
    if args.command == "restart":
        _stop()
        _ensure_running(url, quiet=True)
        print(f"restarted: {url}")
        return 0
    if args.command == "status":
        return _status(as_json=args.json)
    if args.command == "logs":
        return _logs(args.n, args.follow)
    if args.command == "doctor":
        return _doctor()
    if args.command == "setup":
        return _setup(args.router, args.force, enable=True if args.yes else False if args.no_enable else None)
    if args.command == "enable":
        return _enable(force=args.force)
    if args.command == "disable":
        return _disable()
    if args.command == "config":
        cfg = _config_path()
        print(cfg if cfg else f"none -- using the defaults (router off). Put a config at "
                              f"{_home() / 'config.yaml'} or set AAR_GATEWAY_CONFIG.")
        return 0 if cfg else 1
    if args.command == "dashboard":
        import webbrowser

        _ensure_running(url)
        print(url + "/dashboard")
        webbrowser.open(url + "/dashboard")
        return 0
    if args.command == "service":
        return _service(args.action)

    import uvicorn  # serve

    from .server import app

    uvicorn.run(app, host=os.environ.get("AAR_GATEWAY_HOST", "127.0.0.1"),
                port=int(os.environ.get("AAR_GATEWAY_PORT", "8787")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
