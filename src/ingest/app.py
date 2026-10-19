"""The collection server: ``POST /v1/records`` takes the anonymous queues gateways upload.

Run it anywhere that has a disk (``uvicorn ingest.app:app``); records are stored as
``<shard dir>/<UTC date>/<install id>.jsonl`` and :mod:`ingest.publish_kaggle` turns them into
dataset versions. There are no user credentials: abuse is bounded by validation, size limits
and a per-install rate limit. Nothing stored can be read back through this app.

Environment: ``INGEST_SHARD_DIR`` (default ``./shards``), ``INGEST_MAX_BODY_BYTES``,
``INGEST_MAX_RECORDS`` per upload, ``INGEST_RECORDS_PER_HOUR`` per install.
"""

from __future__ import annotations

import collections
import json
import os
import re
import threading
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .schema import SCHEMA, validate_record

_INSTALL = re.compile(r"^[0-9a-f]{16,32}$")
#: a gzip body may expand at most this many times its allowed wire size (a zip bomb stops here)
_EXPANSION = 20


def create_app(shard_dir: Path | str, *, max_body_bytes: int = 1_000_000, max_records: int = 5_000,
               records_per_hour: int = 20_000, clock=time.monotonic) -> FastAPI:
    shards = Path(shard_dir)
    app = FastAPI(title="AAR usage collection", docs_url=None, redoc_url=None, openapi_url=None)
    lock = threading.Lock()
    recent: dict[str, collections.deque] = {}

    def refuse(status: int, message: str) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status)

    def over_rate_limit(install: str, n: int) -> bool:
        now = clock()
        window = recent.setdefault(install, collections.deque())
        while window and now - window[0][0] > 3600:
            window.popleft()
        if sum(count for _, count in window) + n > records_per_hour:
            return True
        window.append((now, n))
        return False

    @app.post("/v1/records")
    async def records(request: Request):
        install = request.headers.get("x-aar-install", "")
        if not _INSTALL.match(install):
            return refuse(400, "bad install id")
        if request.headers.get("x-aar-schema") != str(SCHEMA):
            return refuse(400, f"unsupported schema; this server takes {SCHEMA}")
        if request.headers.get("content-encoding", "").lower() != "gzip":
            return refuse(400, "send a gzip body")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_body_bytes:
            return refuse(413, "body too large")
        body = await request.body()
        if len(body) > max_body_bytes:
            return refuse(413, "body too large")
        limit = max_body_bytes * _EXPANSION
        try:
            text = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(body, limit + 1)
        except zlib.error:
            return refuse(400, "not a gzip body")
        if len(text) > limit:
            return refuse(413, "body too large once unzipped")
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return refuse(400, "no records")
        if len(lines) > max_records:
            return refuse(413, f"at most {max_records} records per upload")

        good, rejected = [], []
        for number, line in enumerate(lines, 1):
            try:
                good.append(validate_record(json.loads(line)))
            except ValueError:  # the reason names a field, never a value: nothing from the record is echoed
                rejected.append(number)
        if not good:
            return refuse(422, f"no valid record (first bad: record {rejected[0]})")
        with lock:
            if over_rate_limit(install, len(good)):
                return refuse(429, "too many records from this install; try again later")
            folder = shards / datetime.now(UTC).strftime("%Y-%m-%d")
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / f"{install}.jsonl").open("a", encoding="utf-8") as f:
                for row in good:
                    f.write(json.dumps(row) + "\n")
        return {"accepted": len(good), **({"rejected": len(rejected)} if rejected else {})}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def _from_environment() -> FastAPI:
    return create_app(
        os.environ.get("INGEST_SHARD_DIR", "shards"),
        max_body_bytes=int(os.environ.get("INGEST_MAX_BODY_BYTES", 1_000_000)),
        max_records=int(os.environ.get("INGEST_MAX_RECORDS", 5_000)),
        records_per_hour=int(os.environ.get("INGEST_RECORDS_PER_HOUR", 20_000)),
    )


app = _from_environment()
