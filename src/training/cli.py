"""Shared CLI helpers for the ``scripts/`` entry points.

Phase 0 scripts use :func:`base_parser` / :func:`get_config` / :func:`info`; the
Phase 1/2/routing scripts use :func:`raw_parser` plus :func:`write_json`,
:func:`zeroshot_only` and :func:`float_table` for the boilerplate that used to be
copy-pasted into every one of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from router.config import Config, load_config, resolve_path as _resolve_path


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to phase0.yaml")
    return p


def raw_parser(description: str | None) -> argparse.ArgumentParser:
    """``ArgumentParser`` that renders the module docstring verbatim in ``--help``."""
    return argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )


def get_config(args: argparse.Namespace) -> Config:
    return load_config(args.config)


def info(msg: str) -> None:
    print(f"[phase0] {msg}", file=sys.stderr)


def resolve(path: str | Path, root: str | Path | None = None) -> Path:
    """Absolute ``path``, taken relative to ``root`` (default: the repo root).

    A thin wrapper over :func:`router.config.resolve_path` (XD-11) -- kept as
    its own name/signature since every caller here resolves a config *file*
    path before any ``Config`` instance exists, so it can't become
    ``Config.resolve`` itself."""
    return _resolve_path(path, root=root)


def json_default(obj: Any) -> Any:
    """``json.dumps(default=...)`` hook that keeps numpy ints/bools as ints/bools.

    ``default=float`` turned ``np.int64`` counts into ``12.0`` and raised on
    arrays / paths; this maps each type to its natural JSON counterpart.
    """
    import numpy as np

    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):  # datetime / pd.Timestamp
        return obj.isoformat()
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def write_json(path: str | Path, payload: Any, *, root: str | Path | None = None,
               announce: bool = True) -> Path:
    """Write ``payload`` as pretty JSON (numpy-aware, see :func:`json_default`),
    making parent dirs.

    The ``mkdir`` + ``json.dumps(..., indent=2)`` + ``print("wrote ...")`` dance
    that ~11 scripts + the pool-expansion battery each carried.
    """
    out = resolve(path, root)
    text = json.dumps(payload, indent=2, default=json_default)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    if announce:
        print(f"wrote {out}")
    return out


def zeroshot_only(*frames, suffix: str = ":5shot"):
    """Drop rows whose ``query_id`` index ends with ``suffix`` from each frame.

    RouterBench ships 0-shot and 5-shot variants of the same item; the routing
    reports score only the 0-shot subset so every policy sees the same queries.
    Returns a single frame when called with one, else a tuple.
    """
    out = tuple(f.loc[[not str(q).endswith(suffix) for q in f.index]] for f in frames)
    return out[0] if len(out) == 1 else out


def float_table(width: int = 200, precision: int = 4, **extra):
    """``pd.option_context`` with a fixed float format -- for ``df.to_string()`` blocks."""
    import pandas as pd

    return pd.option_context(
        "display.width", width,
        "display.float_format", lambda x: f"{x:.{precision}f}",
        *(kv for item in extra.items() for kv in item),
    )
