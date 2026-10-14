"""The aar-gateway command line: process management, setup checks,
start-at-login files, and the Windows-specific paths."""

from __future__ import annotations

import plistlib
import sys

import pytest

from router.gateway import cli


@pytest.fixture
def nothing_running(monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_PORT", "59999")  # nothing listens here
    monkeypatch.setattr(cli, "_get", lambda url, timeout=1.0: None)


def test_status_when_stopped_says_how_to_start(nothing_running, capsys):
    assert cli.main(["status"]) == 1
    out = capsys.readouterr().out
    assert "stopped" in out and "aar-gateway start" in out


def test_status_when_running_shows_router_and_savings(monkeypatch, capsys):
    replies = {"/health": {"router": {"enabled": True, "kind": "knn"}},
               "/stats": {"totals": {"requests": 12, "cost_usd": 1.0, "cost_if_requested_usd": 4.0}}}
    monkeypatch.setattr(cli, "_get", lambda url, timeout=1.0: next(
        (v for k, v in replies.items() if k in url), None))
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "running" in out and "router         on" in out and "75% less plan usage" in out


def test_config_defaults_to_the_home_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AAR_GATEWAY_CONFIG", raising=False)
    assert cli.main(["config"]) == 1 and "none" in capsys.readouterr().out
    (tmp_path / "config.yaml").write_text("interceptor: {}\n")  # AAR_GATEWAY_LOG is under tmp_path
    assert cli.main(["config"]) == 0 and capsys.readouterr().out.strip() == str(tmp_path / "config.yaml")
    monkeypatch.setenv("AAR_GATEWAY_CONFIG", str(tmp_path / "other.yaml"))
    cli.main(["config"])
    assert capsys.readouterr().out.strip() == str(tmp_path / "other.yaml")


def test_logs_shows_the_last_lines(tmp_path, capsys):
    (tmp_path / "gateway.log").write_text("".join(f"line {i}\n" for i in range(10)))
    assert cli.main(["logs", "-n", "2"]) == 0
    assert capsys.readouterr().out == "line 8\nline 9\n"


def test_logs_from_windows_print_one_line_ending(tmp_path, capsys):
    (tmp_path / "gateway.log").write_bytes(b"a\r\nb\r\n")
    assert cli.main(["logs", "-n", "1"]) == 0
    assert capsys.readouterr().out == "b\n"


def test_doctor_flags_a_missing_claude_and_a_bad_config(nothing_running, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, "which", lambda name, path=None: None)
    bad = tmp_path / "config.yaml"
    bad.write_text("interceptor:\n  models:\n    m: {lane: carrier-pigeon}\n")
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "✗ Claude Code installed" in out and "✗ config" in out and "carrier-pigeon" in out
    assert "fix:" in out


def test_launchd_and_systemd_restart_on_crash_but_not_after_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_CONFIG", str(tmp_path / "config.yaml"))
    plist = plistlib.loads(cli.launchd_plist().encode())
    assert plist["ProgramArguments"][1:] == ["-m", "router.gateway.cli", "serve"]
    assert plist["KeepAlive"] == {"SuccessfulExit": False} and plist["RunAtLoad"] is True
    assert plist["EnvironmentVariables"]["AAR_GATEWAY_CONFIG"].endswith("config.yaml")
    unit = cli.systemd_unit()
    assert "Restart=on-failure" in unit and "-m router.gateway.cli serve" in unit


def test_service_install_and_uninstall_on_macos(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(cli.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(cli, "_running_pid", lambda: None)
    monkeypatch.setattr(cli.os, "getuid", lambda: 501, raising=False)  # absent on Windows
    calls = []
    run = lambda cmd, **kw: calls.append(cmd)  # noqa: E731 -- never touch the real launchctl
    assert cli._service("install", run=run) == 0
    plist = tmp_path / "Library" / "LaunchAgents" / "com.aar.gateway.plist"
    assert plist.exists() and calls[-1][:2] == ["launchctl", "bootstrap"]
    assert cli._service("uninstall", run=run) == 0
    assert not plist.exists() and calls[-1][:2] == ["launchctl", "bootout"]


def test_windows_never_probes_with_a_signal_and_needs_no_shell_env(tmp_path, monkeypatch):
    """os.kill(pid, 0) on Windows terminates the process; the hook command
    can't use the POSIX `VAR=value cmd` form there."""
    monkeypatch.setattr(cli, "WINDOWS", True)
    (tmp_path / "gateway.pid").write_text("4242")

    def boom(*a):
        raise AssertionError("os.kill must not be used to probe on Windows")
    monkeypatch.setattr(cli.os, "kill", boom)
    monkeypatch.setattr(cli, "_healthy", lambda url: True)
    assert cli._running_pid() == 4242
    command = cli._self_command()
    assert "PYTHONPATH=" not in command and command.endswith("-m router.gateway.cli")


def _fake_ollama(monkeypatch, pulled):
    """Ollama answering /api/tags with ``pulled`` (None: not running); the gateway itself down."""
    def get(url, timeout=1.0):
        if "/api/tags" in url:
            return None if pulled is None else {"models": [{"name": m} for m in pulled]}
        return None
    monkeypatch.setattr(cli, "_get", get)
    monkeypatch.setattr(cli, "_healthy", lambda url: False)


def test_local_status_says_how_to_install_ollama_when_missing(monkeypatch, capsys):
    _fake_ollama(monkeypatch, None)
    monkeypatch.setattr(cli.shutil, "which", lambda name, path=None: None)
    assert cli.main(["local"]) == 0
    out = capsys.readouterr().out
    assert "not installed" in out and "to install Ollama" in out


def test_local_add_needs_ollama_running(monkeypatch):
    _fake_ollama(monkeypatch, None)
    monkeypatch.setattr(cli.shutil, "which", lambda name, path=None: "/usr/bin/ollama")
    with pytest.raises(SystemExit, match="isn't running"):
        cli.main(["local", "add", "llama3.1"])


def test_local_add_pulls_once_adds_and_remove_takes_it_out(monkeypatch, capsys):
    from router.gateway.config import load_settings, local_models_path

    _fake_ollama(monkeypatch, ["qwen2.5-coder:14b"])
    monkeypatch.setattr(cli.shutil, "which", lambda name, path=None: "/usr/bin/ollama")
    pulls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **kw: pulls.append(cmd) or
                        cli.subprocess.CompletedProcess(cmd, 0))
    assert cli.main(["local", "add", "llama3.1"]) == 0
    assert pulls == [["/usr/bin/ollama", "pull", "llama3.1:latest"]]
    models = load_settings(None)["models"]
    assert models["llama3.1:latest"]["lane"] == "openai" and models["llama3.1:latest"]["proxy"] is None
    assert "claude-opus-5-5" in models  # added to the defaults, never replacing them
    assert cli.main(["local", "remove", "llama3.1"]) == 0
    assert "llama3.1:latest" not in load_settings(None)["models"] and not local_models_path().exists()
