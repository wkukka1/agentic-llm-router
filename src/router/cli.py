"""Shared CLI helpers for Phase 0 scripts."""

from __future__ import annotations

import argparse
import sys

from .config import Config, load_config


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to phase0.yaml")
    return p


def get_config(args: argparse.Namespace) -> Config:
    return load_config(args.config)


def info(msg: str) -> None:
    print(f"[phase0] {msg}", file=sys.stderr)
