"""POST /v1/records: validates everything, stores shards, bounds abuse."""

from __future__ import annotations

import gzip
import json

import pytest
from fastapi.testclient import TestClient

from ingest.app import create_app
from router.gateway import share

LOGGED = {"ts": "2026-10-18T10:00:00+00:00", "requested_model": "claude-opus-5-5", "routed_model": "claude-haiku-5-5",
          "reason": "cheapest", "status": 200, "subagent": False, "input_tokens": 1, "output_tokens": 1}
INSTALL = "0123456789abcdef"


@pytest.fixture(autouse=True)
def _no_encoder(monkeypatch):
    monkeypatch.setattr(share, "_embed", lambda text: [i % 100 for i in range(share.DIMS)])


@pytest.fixture
def server(tmp_path):
    app = create_app(tmp_path / "shards", max_body_bytes=200_000, max_records=50, records_per_hour=100)
    with TestClient(app) as client:
        client.shards = tmp_path / "shards"
        yield client


def _body(rows):
    return gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode())


def _post(client, rows=None, *, body=None, install=INSTALL, schema="1", encoding="gzip"):
    rows = rows if rows is not None else [share.build_record(LOGGED, "typed message")]
    headers = {"content-type": "application/x-ndjson", "content-encoding": encoding,
               "x-aar-schema": schema, "x-aar-install": install}
    return client.post("/v1/records", content=body if body is not None else _body(rows), headers=headers)


def _stored(client):
    return [json.loads(line) for f in sorted(client.shards.rglob("*.jsonl")) for line in f.read_text().splitlines()]


def test_a_real_upload_is_accepted_and_stored_under_the_install_and_the_servers_date(server):
    rows = [share.build_record(LOGGED, f"message {i}") for i in range(3)]
    r = _post(server, rows)
    assert r.status_code == 200 and r.json() == {"accepted": 3}
    (shard,) = server.shards.rglob("*.jsonl")
    assert shard.name == f"{INSTALL}.jsonl" and shard.parent.parent == server.shards
    assert _stored(server) == rows


def test_a_bad_record_is_refused_without_blocking_the_good_ones(server):
    """A poisoned record cannot jam an install's queue for ever: the valid ones are stored, the bad one is
    counted, and nothing from it is stored or echoed."""
    rows = [share.build_record(LOGGED, "ok"), {**share.build_record(LOGGED, "bad"), "prompt": "leaked text"},
            share.build_record(LOGGED, "also ok")]
    r = _post(server, rows)
    assert r.status_code == 200 and r.json() == {"accepted": 2, "rejected": 1} and "leaked" not in r.text
    assert len(_stored(server)) == 2 and "leaked" not in json.dumps(_stored(server))


def test_an_upload_with_no_valid_record_is_an_error_so_a_broken_client_notices(server):
    r = _post(server, [{"prompt": "only text"}])
    assert r.status_code == 422 and "record 1" in r.text and "only text" not in r.text
    assert _stored(server) == []


@pytest.mark.parametrize("install", ["../../etc/passwd", "a/b", "XYZ", "", "1" * 100, "short"])
def test_the_install_id_cannot_escape_the_shard_folder(server, install):
    assert _post(server, install=install).status_code == 400
    assert not list(server.shards.rglob("*.jsonl"))


def test_wrong_schema_encoding_and_garbage_are_refused(server):
    assert _post(server, schema="2").status_code == 400
    assert _post(server, encoding="identity", body=b"{}").status_code == 400
    assert _post(server, body=b"not gzip at all").status_code == 400
    assert _post(server, body=gzip.compress(b"{not json}\n")).status_code == 422
    assert _post(server, body=gzip.compress(b"[1, 2]\n")).status_code == 422
    assert _post(server, rows=[]).status_code == 400  # nothing to store


def test_size_limits_cover_the_wire_and_the_unzipped_body(server):
    assert _post(server, body=b"x" * 300_000).status_code == 413          # too large as sent
    bomb = gzip.compress(b"0" * 50_000_000)                               # tiny on the wire, huge unzipped
    assert len(bomb) < 200_000
    assert _post(server, body=bomb).status_code == 413
    too_many = [share.build_record(LOGGED, "m")] * 51
    assert _post(server, too_many).status_code == 413


def test_an_install_is_rate_limited_but_others_are_not(server):
    row = share.build_record(LOGGED, "m")
    assert _post(server, [row] * 50).status_code == 200
    assert _post(server, [row] * 50).status_code == 200
    assert _post(server, [row]).status_code == 429          # 101st record this hour
    assert _post(server, [row], install="fedcba9876543210").status_code == 200
    assert len(_stored(server)) == 101


def test_health_and_nothing_else_is_readable(server):
    assert server.get("/health").json() == {"status": "ok"}
    assert server.get("/v1/records").status_code == 405
    assert server.get("/shards").status_code == 404
