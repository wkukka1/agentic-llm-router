"""Provenance helpers shared by every checkpoint / ledger writer.

``git_sha`` / ``git_dirty`` describe the code; ``file_digest`` / ``artifact_digests``
hash the input artifacts a run consumed. Keeping one copy here stops the three
former inline implementations (``nirt.train``, ``nirt.baseline.checkpoint``,
``pool_expansion.battery``) from drifting apart.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Optional

from .config import REPO_ROOT


def git_sha(cwd: Optional[str | Path] = None) -> Optional[str]:
    """``HEAD`` commit sha, or ``None`` outside a git checkout.

    ``cwd`` defaults to this repo's root, not the process working directory,
    so a run launched from elsewhere still records the right commit."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(cwd if cwd is not None else REPO_ROOT),
        )
        return out.decode().strip()
    except Exception:  # pragma: no cover - defensive
        return None


def git_dirty(cwd: Optional[str | Path] = None) -> Optional[bool]:
    """``True`` if the working tree has uncommitted changes, ``None`` if unknown.
    ``cwd`` defaults to the repo root (see :func:`git_sha`)."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            cwd=str(cwd if cwd is not None else REPO_ROOT),
        )
        return bool(out.decode().strip())
    except Exception:  # pragma: no cover - defensive
        return None


def file_digest(path: str | Path, *, algo: str = "sha256", n: int = 16) -> Optional[str]:
    """First ``n`` hex chars of ``algo(path bytes)``, or ``None`` if the file is absent."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.new(algo)
    with open(p, "rb") as fh:           # streamed: artifacts can be GBs
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def artifact_digests(
    spec: Mapping[str, str | Path] | Iterable[tuple[str, str | Path]],
    *,
    algo: str = "sha256",
    n: int = 16,
) -> dict[str, Optional[str]]:
    """``{label: file_digest(path)}`` for a mapping (or pairs) of label -> path."""
    items = spec.items() if isinstance(spec, Mapping) else spec
    return {label: file_digest(path, algo=algo, n=n) for label, path in items}
