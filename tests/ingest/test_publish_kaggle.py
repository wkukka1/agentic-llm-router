"""publish_kaggle bundles the shards into a dataset folder and calls the kaggle CLI."""

from __future__ import annotations

import gzip
import json

import pytest

from ingest.publish_kaggle import publish

ROW = {"v": 1, "day": "2026-10-18", "status": 200}


def _shard(root, day, install, n):
    path = root / day / f"{install}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({**ROW, "n": i}) + "\n" for i in range(n)))


def test_a_first_publish_creates_the_dataset_with_one_gzip_per_day(tmp_path, kaggle):
    shards = tmp_path / "shards"
    _shard(shards, "2026-10-17", "aaaa1111aaaa1111", 2)
    _shard(shards, "2026-10-17", "bbbb2222bbbb2222", 3)
    _shard(shards, "2026-10-18", "aaaa1111aaaa1111", 1)
    result = publish(shards, tmp_path / "work", "someone/aar-usage", create=True, license_name="CC-BY-4.0")
    assert result == {"published": 6, "days": 2}
    (call,) = kaggle()
    assert call["args"][:2] == ["datasets", "create"]
    assert call["files"] == ["dataset-metadata.json", "records-2026-10-17.jsonl.gz", "records-2026-10-18.jsonl.gz"]
    work = tmp_path / "work" / "dataset"
    meta = json.loads((work / "dataset-metadata.json").read_text())
    assert meta["id"] == "someone/aar-usage" and meta["licenses"] == [{"name": "CC-BY-4.0"}]
    with gzip.open(work / "records-2026-10-17.jsonl.gz", "rt") as f:
        assert len([json.loads(line) for line in f]) == 5


def test_a_later_publish_is_a_new_version_that_still_holds_every_day(tmp_path, kaggle):
    shards = tmp_path / "shards"
    _shard(shards, "2026-10-17", "aaaa1111aaaa1111", 2)
    publish(shards, tmp_path / "work", "someone/aar-usage", create=True)
    _shard(shards, "2026-10-18", "aaaa1111aaaa1111", 4)
    assert publish(shards, tmp_path / "work", "someone/aar-usage")["published"] == 4
    first, second = kaggle()
    assert second["args"][:2] == ["datasets", "version"] and "-m" in second["args"]
    assert second["files"] == ["dataset-metadata.json", "records-2026-10-17.jsonl.gz", "records-2026-10-18.jsonl.gz"]


def test_nothing_new_means_no_kaggle_call(tmp_path, kaggle):
    shards = tmp_path / "shards"
    _shard(shards, "2026-10-17", "aaaa1111aaaa1111", 2)
    publish(shards, tmp_path / "work", "someone/aar-usage", create=True)
    assert publish(shards, tmp_path / "work", "someone/aar-usage") == {"published": 0, "days": 0, "skipped": "nothing new"}
    assert len(kaggle()) == 1


def test_a_failed_kaggle_call_publishes_nothing_and_the_records_go_out_next_time(tmp_path, kaggle, monkeypatch):
    shards = tmp_path / "shards"
    _shard(shards, "2026-10-17", "aaaa1111aaaa1111", 2)
    monkeypatch.setenv("FAKE_KAGGLE_EXIT", "1")
    with pytest.raises(RuntimeError):
        publish(shards, tmp_path / "work", "someone/aar-usage", create=True)
    monkeypatch.setenv("FAKE_KAGGLE_EXIT", "0")
    assert publish(shards, tmp_path / "work", "someone/aar-usage", create=True)["published"] == 2


def test_credentials_are_never_written_into_the_dataset_folder(tmp_path, kaggle, monkeypatch):
    monkeypatch.setenv("KAGGLE_KEY", "supersecretkey")
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    shards = tmp_path / "shards"
    _shard(shards, "2026-10-17", "aaaa1111aaaa1111", 1)
    publish(shards, tmp_path / "work", "someone/aar-usage", create=True)
    for f in (tmp_path / "work").rglob("*"):
        if f.is_file() and not f.name.endswith(".gz"):
            assert "supersecretkey" not in f.read_text(errors="ignore")
