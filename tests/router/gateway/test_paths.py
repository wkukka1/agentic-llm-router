"""Every gateway file lives under one folder, found one way, by the server, the CLI and the grader."""

from __future__ import annotations

from pathlib import Path

from router.gateway import cli, config, interceptor, server, share


def test_a_tilde_in_the_log_setting_means_the_same_folder_everywhere(monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", "~/gw/requests.jsonl")
    home = Path("~/gw").expanduser()
    assert interceptor.log_path() == home / "requests.jsonl"
    assert config.gateway_home() == cli._home() == home
    assert server._pid_file() == cli._pid_file() == home / "gateway.pid"
    assert share.share_dir() == home / "share"


def test_default_folder_when_no_log_is_set(monkeypatch):
    monkeypatch.delenv("AAR_GATEWAY_LOG")
    home = Path.home() / ".aar-gateway"
    assert interceptor.log_path() == home / "requests.jsonl"
    assert config.graded_path() == home / "graded.jsonl"
