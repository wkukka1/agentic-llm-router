"""/stats for a log that fits one file must not change when summaries become streaming and mergeable."""

from __future__ import annotations

import json
from pathlib import Path

from summary_fixture import make_records

from router.gateway import interceptor

GOLDEN = Path(__file__).with_name("golden_summary.json")


def _normalised(summary: dict) -> dict:
    return json.loads(json.dumps({**summary, "log": "LOG"}, sort_keys=True))


def test_stats_for_a_single_log_file_are_unchanged():
    interceptor.append_jsonl(interceptor.log_path(), *make_records(400))
    assert _normalised(interceptor.summarize()) == json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_the_same_records_summarised_directly_give_the_same_answer():
    records = make_records(400)
    interceptor.append_jsonl(interceptor.log_path(), *records)
    assert _normalised(interceptor.summarize(records)) == _normalised(interceptor.summarize())
