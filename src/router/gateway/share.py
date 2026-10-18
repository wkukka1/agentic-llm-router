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
times finer than the day. The queue stays on this computer, capped at
:data:`MAX_RECORDS` (``aar-gateway share show`` prints it), until an upload
token is set (``aar-gateway share token``); then it goes to the private
Hugging Face dataset :data:`DEFAULT_DATASET` once a day, or after
:data:`UPLOAD_AFTER` records, as files under ``records/<install id>/`` (a
random id, not tied to the person). A failed upload keeps the records for
next time and never affects routing.
"""

from __future__ import annotations

import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from .config import atomic_write_text, locked_file, read_jsonl

#: the compressed embedding: this many dimensions, from this fixed projection
DIMS = 64
PROJECTION_SEED = 20261017
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#: the queue keeps the newest this many records
MAX_RECORDS = 20_000
#: where records go (a private dataset), and when
DEFAULT_DATASET = "Vivic04/aar-gateway-usage"
UPLOAD_AFTER = 100
UPLOAD_EVERY_S = 24 * 3600
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


def token_path() -> Path:
    return share_dir() / "token"


def state_path() -> Path:
    return share_dir() / "state.json"


def dataset() -> str:
    return os.environ.get("AAR_GATEWAY_SHARE_DATASET") or DEFAULT_DATASET


def token() -> str | None:
    """The upload token: ``AAR_GATEWAY_SHARE_TOKEN``, else the one saved by
    ``aar-gateway share token``. Never printed or logged."""
    if os.environ.get("AAR_GATEWAY_SHARE_TOKEN"):
        return os.environ["AAR_GATEWAY_SHARE_TOKEN"]
    try:
        return token_path().read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def set_token(value: str) -> None:
    atomic_write_text(token_path(), value.strip())
    try:
        os.chmod(token_path(), 0o600)  # readable by this user only (no-op on Windows)
    except OSError:
        pass


def state() -> dict:
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _install_id() -> str:
    path = share_dir() / "install-id"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        import uuid

        value = uuid.uuid4().hex[:16]
        atomic_write_text(path, value)
        return value


def upload(*, api=None) -> dict:
    """Send the queued records to the dataset as one file; on success they
    leave the queue. ``{"sent": n}``, or ``{"error": ...}`` / ``{"skipped": why}``."""
    tok = token()
    if not tok:
        return {"skipped": "no upload token (aar-gateway share token <token>)"}
    with _lock:
        path = queue_path()
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []
    lines = [line for line in lines if line.strip()]
    if not lines:
        return {"skipped": "nothing queued"}
    import uuid

    now = datetime.now(UTC)
    name = f"records/{_install_id()}/{now.date().isoformat()}-{uuid.uuid4().hex[:8]}.jsonl"
    st = state()
    try:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=tok)
        api.upload_file(path_or_fileobj="".join(lines).encode("utf-8"), path_in_repo=name,
                        repo_id=dataset(), repo_type="dataset", commit_message=f"usage records ({len(lines)})")
    except Exception as exc:  # noqa: BLE001 -- kept for next time; the type only, never the message (it can echo the token)
        st.update(last_error=type(exc).__name__, last_attempt=now.isoformat(timespec="seconds"))
        atomic_write_text(state_path(), json.dumps(st))
        return {"error": type(exc).__name__}
    with _lock, locked_file(path):  # drop exactly what was sent; anything written meanwhile stays
        current = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []
        remaining = current[len(lines):] if current[:len(lines)] == lines else [x for x in current if x not in set(lines)]
        atomic_write_text(path, "".join(remaining))
    st.update(last_upload=now.isoformat(timespec="seconds"), sent_total=st.get("sent_total", 0) + len(lines),
              last_error=None, last_attempt=now.isoformat(timespec="seconds"))
    atomic_write_text(state_path(), json.dumps(st))
    return {"sent": len(lines), "file": name}


def _upload_due(n_queued: int) -> bool:
    if not token() or n_queued == 0:
        return False
    if n_queued >= UPLOAD_AFTER:
        return True
    last = state().get("last_attempt")
    if not last:
        return True
    return (datetime.now(UTC) - datetime.fromisoformat(last)).total_seconds() >= UPLOAD_EVERY_S


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
    if reason_category(log_record.get("reason")) == "side_request":
        return  # Claude Code's own background requests (titles, summaries): not the user's messages
    global _worker
    if _worker is None:
        _worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aar-share")
    _worker.submit(_append, dict(log_record), human)


def _append(log_record: dict, human: str) -> None:
    try:
        n = _queue_line(json.dumps(build_record(log_record, human)) + "\n")
        if _upload_due(n):
            upload()
    except Exception:  # noqa: BLE001 -- sharing must never affect routing
        pass


def _queue_line(line: str) -> int:
    """Append one record to the queue and trim it to ``MAX_RECORDS``; returns
    its length. Locked across processes: the CLI's upload rewrites the same file."""
    path = queue_path()
    with _lock, locked_file(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
        _cap(path)
        with path.open(encoding="utf-8") as f:
            return sum(1 for _ in f)


def _cap(path: Path) -> None:
    if path.stat().st_size < MAX_RECORDS * 1500:  # a record is ~700 bytes: cheap check first
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) > MAX_RECORDS:
        atomic_write_text(path, "\n".join(lines[-MAX_RECORDS:]) + "\n")


def queued() -> list[dict]:
    return read_jsonl(queue_path())


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
