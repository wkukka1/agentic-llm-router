"""Fixtures for the collection-server tests: a private gateway home (the end-to-end test drives the real
share module) and a fake `kaggle` command."""

from __future__ import annotations

import json
import os
import pathlib
import stat
import sys

import pytest


@pytest.fixture(autouse=True)
def _isolated_gateway_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "gw" / "requests.jsonl"))
    home = tmp_path / "home"
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: home))


@pytest.fixture
def kaggle(tmp_path, monkeypatch):
    """A fake `kaggle` executable that records its arguments and the folder it was pointed at."""
    log = tmp_path / "kaggle-calls.jsonl"
    exe = tmp_path / "bin" / ("kaggle.cmd" if os.name == "nt" else "kaggle")
    exe.parent.mkdir()
    script = tmp_path / "bin" / "fake_kaggle.py"
    script.write_text("\n".join([
        "import json, os, sys",
        "args = sys.argv[1:]",
        "folder = args[args.index('-p') + 1]",
        "files = sorted(os.listdir(folder))",
        f"open({str(log)!r}, 'a').write(json.dumps({{'args': args, 'files': files}}) + chr(10))",
        "sys.exit(int(os.environ.get('FAKE_KAGGLE_EXIT', '0')))",
    ]), encoding="utf-8")
    if os.name == "nt":
        exe.write_text("\r\n".join(["@echo off", f'"{sys.executable}" "{script}" %*', ""]))
    else:
        exe.write_text("\n".join(["#!/bin/sh", f'exec "{sys.executable}" "{script}" "$@"', ""]))
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{exe.parent}{os.pathsep}{os.environ['PATH']}")
    return lambda: [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
