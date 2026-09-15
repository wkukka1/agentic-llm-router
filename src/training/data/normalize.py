"""Text / query-id normalization shared by every loader.

* ``normalize_query_text`` -- light canonicalization used *only* for dedup and
  content hashing. The human-readable ``query`` column keeps the original text.
* ``content_hash`` -- stable hash of the normalized text; the basis for
  cross-dataset duplicate detection and leakage-free splitting.
* ``make_query_id`` -- deterministic per-(source, dataset, content) id.
"""

from __future__ import annotations

import ast
import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")


def sanitize_text(s: str) -> str:
    """Drop lone surrogates / invalid code points so text is safe to hash,
    write to parquet, and print on any console."""
    return str(s).encode("utf-8", "replace").decode("utf-8", "replace")


def render_prompt(raw: object) -> str:
    """Several sources store the prompt as a ``str(list[...])`` of turns.

    Render such values to a single readable string (turns joined by blank
    lines). Plain strings pass through unchanged.
    """
    if isinstance(raw, (list, tuple)):
        return sanitize_text("\n\n".join(str(t) for t in raw))
    s = str(raw)
    if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
        try:
            parsed = ast.literal_eval(s)
        except Exception:  # SyntaxError, ValueError, RecursionError, MemoryError, ...
            parsed = None
        # only a list of turns (strings / role-content dicts) is unpacked; a genuine
        # prompt such as "[1, 2, 3]" or a JSON-array task is kept verbatim
        if (isinstance(parsed, (list, tuple)) and parsed
                and all(isinstance(t, (str, dict)) for t in parsed)):
            return sanitize_text("\n\n".join(str(t) for t in parsed))
    return sanitize_text(s)


def normalize_query_text(text: str) -> str:
    """Aggressive normalization for dedup/hashing only (not for display)."""
    t = unicodedata.normalize("NFKC", sanitize_text(text))
    t = t.lower().strip()
    t = _WS.sub(" ", t)
    return t


def content_hash(text: str) -> str:
    norm = normalize_query_text(text)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def make_query_id(source: str, dataset: str, text: str) -> str:
    """Deterministic id, unique per (source, dataset, normalized content)."""
    h = content_hash(text)[:16]
    ds = re.sub(r"[^a-z0-9]+", "_", dataset.lower()).strip("_")
    return f"{source}:{ds}:{h}"
