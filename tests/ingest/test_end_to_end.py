"""A gateway's opted-in queue travels to the collection server and on to a dataset version."""

from __future__ import annotations

import gzip
import json
import socket
import threading
import time

import pytest
import uvicorn
from aar_gateway import share

from ingest.app import create_app
from ingest.publish_kaggle import publish

LOGGED = {"ts": "2026-10-18T10:00:00+00:00", "requested_model": "claude-opus-5-5", "routed_model": "claude-haiku-5-5",
          "reason": "cheapest within 0.05", "status": 200, "subagent": False, "input_tokens": 9, "output_tokens": 3,
          "error": "overloaded_error: Overloaded", "correction": False}


@pytest.fixture
def collector(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(tmp_path / "shards"), host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/v1/records", tmp_path / "shards"
    server.should_exit = True
    thread.join(5)


def test_queue_to_collector_to_kaggle_dataset(collector, kaggle, tmp_path, monkeypatch):
    url, shards = collector
    monkeypatch.setattr(share, "_embed", lambda text: [i % 90 - 45 for i in range(share.DIMS)])
    share.state_path().parent.mkdir(parents=True, exist_ok=True)
    share.state_path().write_text('{"last_attempt": "2999-01-01T00:00:00+00:00"}')  # only the explicit upload sends
    share.set_endpoint(url)
    share.set_enabled(True)
    for i in range(4):
        share.record(LOGGED, f"a private message about project {i}")
    share.flush(10)

    assert share.upload() == {"sent": 4} and share.queued() == []
    (shard,) = shards.rglob("*.jsonl")
    stored = [json.loads(line) for line in shard.read_text().splitlines()]
    assert len(stored) == 4 and all(len(r["embedding"]) == share.DIMS for r in stored)
    assert "private message" not in shard.read_text() and all(r["error_type"] == "overloaded_error" for r in stored)

    assert publish(shards, tmp_path / "work", "someone/aar-usage", create=True)["published"] == 4
    (call,) = kaggle()
    (dataset_file,) = [f for f in call["files"] if f.endswith(".jsonl.gz")]
    with gzip.open(tmp_path / "work" / "dataset" / dataset_file, "rt") as f:
        assert [json.loads(line) for line in f] == stored


def test_a_collector_that_is_down_costs_nothing(tmp_path):
    share.set_endpoint("http://127.0.0.1:9/v1/records")
    share.set_enabled(True)
    share._queue_line(json.dumps({"n": 1}) + "\n")
    assert "error" in share.upload() and len(share.queued()) == 1
