"""One HTTP download helper for the raw-data fetchers (pool builders and public
benchmark loaders). Lives under ``training`` because ``evaluation`` may import
``training`` but not the other way round. Stdlib only: ``httpx`` is not a core
dependency and the training pipeline must install without the gateway extras."""
from __future__ import annotations

import urllib.request
from pathlib import Path

#: benchlm.ai answers Python's default User-Agent with 403 even for its free
#: CC-BY-NC-4.0 file; a normal browser-like UA is all it needs, not a bypass of
#: any access control. Sent with every download.
USER_AGENT = "Mozilla/5.0"


def fetch(url: str, dest: Path, *, timeout: float = 600) -> Path:
    """Download ``url`` to ``dest`` unless it already exists. Streams into
    ``<dest>.part`` and renames on success, so an interrupted run never leaves a
    truncated file that looks complete."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- fixed, trusted URLs only
        with tmp.open("wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    return dest
