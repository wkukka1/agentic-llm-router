"""Opt-in usage sharing: anonymous routing records, to improve the router.

Off unless the user turns it on (``aar-gateway setup`` asks; ``aar-gateway
share on|off``). Then each message the user types adds one record to
``~/.aar-gateway/share/queue.jsonl``:

* the models asked for and used, and why (a category, not free text);
* tokens, cost, the cost on the model asked for, latency, status and the
  error *type*; whether the message corrected a cheaper model's answer; the
  thread mode; the prompt's length, rounded to a power of two;
* the prompt's embedding, compressed: the 384-number MiniLM vector projected
  onto 64 fixed random directions (:data:`PROJECTION_SEED`) and rounded to
  small integers. Coarse on purpose: a full embedding of a known model can
  be turned back into much of a short text.

Never: the prompt, error messages, file paths, session or conversation ids,
times finer than the day. Nothing is sent anywhere until a destination is
configured; until then the queue stays on this computer, capped at
:data:`MAX_RECORDS`, and ``aar-gateway share show`` prints what it holds.
"""

from __future__ import annotations

import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from .config import atomic_write_text

#: the compressed embedding: this many dimensions, from this fixed projection
DIMS = 64
PROJECTION_SEED = 20261017
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#: the queue keeps the newest this many records
MAX_RECORDS = 20_000
#: why a model was used, as categories (the gateway's reasons are free text)
_REASONS = (("continues a server-side thread", "thread"), ("same turn", "same_turn"),
            ("pinned", "pinned"), ("routing off", "routing_off"), ("after a correction", "after_correction"),
            ("side request", "side_request"), ("fell back", "fallback"), ("router error", "router_error"),
            ("client's choice", "client_choice"), ("subagent", "subagent"))

_lock = threading.Lock()
_worker: ThreadPoolExecutor | None = None
_encoder = None
_projection = None
_consent: dict = {"mtime": None, "on": False}


def share_dir() -> Path:
    from .interceptor import log_path

    return log_path().parent / "share"


def consent_path() -> Path:
    return share_dir() / "consent.json"


def queue_path() -> Path:
    return share_dir() / "queue.jsonl"


def enabled() -> bool:
    """The user's choice, re-read when the consent file changes (the CLI
    writes it; the gateway runs in another process)."""
    path = consent_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    if mtime != _consent["mtime"]:
        try:
            on = bool(json.loads(path.read_text(encoding="utf-8")).get("share_usage"))
        except (OSError, ValueError):
            on = False
        _consent.update(mtime=mtime, on=on)
    return _consent["on"]


def set_enabled(on: bool) -> None:
    atomic_write_text(consent_path(), json.dumps({"share_usage": on,
                                                  "changed": datetime.now(UTC).date().isoformat()}))


def reason_category(reason: str | None) -> str:
    text = (reason or "").lower()
    return next((cat for key, cat in _REASONS if key in text), "router")


def _compress(vector) -> list[int]:
    import numpy as np

    global _projection
    if _projection is None:
        rng = np.random.default_rng(PROJECTION_SEED)
        _projection = rng.normal(0, 1 / math.sqrt(DIMS), (len(vector), DIMS))
    z = np.asarray(vector, dtype=float) @ _projection
    z = z / (np.linalg.norm(z) or 1.0)
    return [int(v) for v in np.clip(np.round(z * 127), -127, 127)]


def _embed(text: str) -> list[int] | None:
    global _encoder
    try:
        if _encoder is None:
            from ..embeddings.encoder import EncoderConfig, TextEncoder, light_backend

            _encoder = TextEncoder(EncoderConfig(name="share", model_name=EMBEDDING_MODEL, backend=light_backend()))
        return _compress(_encoder.encode([text])[0])
    except Exception:  # noqa: BLE001 -- a missing encoder means no embedding, never a failed record
        return None


def build_record(record: dict, human: str | None) -> dict:
    """The shareable part of a request-log record (see the module docstring)."""
    error = record.get("error") or ""
    return {
        "v": 1,
        "day": (record.get("ts") or datetime.now(UTC).isoformat())[:10],
        "gateway": _version(),
        "requested_model": record.get("requested_model"),
        "routed_model": record.get("routed_model"),
        "decision": reason_category(record.get("reason")),
        "subagent_role": record.get("agent_role"),
        "correction": bool(record.get("correction")),
        "thread": record.get("thread"),
        "status": record.get("status"),
        "error_type": error.split(":", 1)[0].strip() if error else None,
        "latency_s": record.get("latency_s"),
        **{k: record.get(k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                      "cache_creation_input_tokens", "cost_usd", "cost_if_requested_usd")},
        "prompt_chars": 2 ** round(math.log2(len(human))) if human else None,
        "embedding_model": f"{EMBEDDING_MODEL}/rp{DIMS}-int8",
        "embedding": _embed(human) if human else None,
    }


def record(log_record: dict, human: str | None) -> None:
    """Queue one record when sharing is on: a message the user typed only.
    Runs on a background thread: never slows the request."""
    if not human or log_record.get("subagent") or not enabled():
        return
    global _worker
    if _worker is None:
        _worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aar-share")
    _worker.submit(_append, dict(log_record), human)


def _append(log_record: dict, human: str) -> None:
    try:
        line = json.dumps(build_record(log_record, human)) + "\n"
        with _lock:
            path = queue_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
            _cap(path)
    except Exception:  # noqa: BLE001 -- sharing must never affect routing
        pass


def _cap(path: Path) -> None:
    if path.stat().st_size < MAX_RECORDS * 1500:  # a record is ~700 bytes: cheap check first
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) > MAX_RECORDS:
        atomic_write_text(path, "\n".join(lines[-MAX_RECORDS:]) + "\n")


def queued() -> list[dict]:
    path = queue_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def flush(timeout: float | None = None) -> None:
    """Wait for queued records to be written (tests, shutdown)."""
    if _worker is not None:
        _worker.submit(lambda: None).result(timeout=timeout)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("agentic-llm-router")
    except Exception:  # noqa: BLE001
        return os.environ.get("AAR_GATEWAY_VERSION", "dev")
