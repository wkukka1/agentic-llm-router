"""Public (non-project) benchmark score lookups for LLM profiles.

Pure record-building / merge logic is tested directly against small inline
fixtures -- no live network calls, per the plan's leakage-safety requirement
that these sources never touch this project's own response data.
"""

from __future__ import annotations

import json

from training.data.public_benchmarks import (
    arena_elo_record,
    benchlm_record,
    hf_leaderboard_record,
    lookup_arena_elo,
    lookup_benchlm,
    lookup_hf_leaderboard,
    merge_benchmarks,
)


# --------------------------------------------------------------------------- #
# pure record builders                                                       #
# --------------------------------------------------------------------------- #
def test_hf_leaderboard_record_keeps_only_present_benchmark_columns():
    """Column names verified against a live download of
    open-llm-leaderboard/contents (2026-10-07): the dataset's current
    benchmark suite is v3 (IFEval/BBH/MATH Lvl 5/GPQA/MUSR/MMLU-PRO) -- it
    has no MMLU/GSM8K/ARC/HellaSwag/TruthfulQA/Winogrande columns at all
    (that was a v1-era assumption that never matched the live schema)."""
    row = {"fullname": "mistralai/Mixtral-8x7B-Instruct-v0.1", "IFEval": 55.99, "BBH": 29.74,
           "GPQA": None}
    rec = hf_leaderboard_record(row)
    assert rec["ifeval"] == {"value": 55.99, "unit": "percent", "source": "hf_open_llm_leaderboard",
                              "version": "v3", "snapshot_date": None}
    assert rec["bbh"]["value"] == 29.74
    assert "gpqa" not in rec  # null column dropped, not fabricated as 0


def test_hf_leaderboard_record_empty_when_no_known_columns():
    assert hf_leaderboard_record({"fullname": "whatever"}) == {}


def test_benchlm_record_reads_category_breakdown():
    """Real schema (verified against a live download 2026-10-07): the item's
    OWN top-level ``displayScore`` is the authoritative overall score
    (``scoreFields.displayScore.role == "published-overall-score"`` in the
    file) -- NOT the nested ``scores.displayScore``/``scores.overallScore``,
    which the file itself labels ``"legacy-compatibility-score"`` and can
    disagree with the real one substantially (e.g. 38.49 vs. 54 for the
    same model). Category scores live under ``scores.displayCategoryScores``,
    camelCase (``multimodalGrounded``, ``instructionFollowing``)."""
    entry = {
        "slug": "claude-4-sonnet", "displayScore": 38.49, "publicRankingMode": "bench-align-v5",
        "scores": {
            "displayScore": 54, "overallScore": 54,  # legacy-compatibility -- must NOT be used
            "displayCategoryScores": {
                "agentic": None, "coding": 52.3, "reasoning": 52.5,
                "multimodalGrounded": 53, "knowledge": 47.4, "multilingual": None,
                "instructionFollowing": 52, "math": None,
            },
        },
    }
    rec = benchlm_record(entry, snapshot_date="2026-10-07")
    assert rec["overall"] == {"value": 38.49, "unit": "benchlm_score", "source": "benchlm",
                               "version": "bench-align-v5", "snapshot_date": "2026-10-07"}
    assert rec["coding"]["value"] == 52.3
    assert rec["multimodal_grounded"]["value"] == 53
    assert rec["instruction_following"]["value"] == 52
    assert "agentic" not in rec and "multilingual" not in rec and "math" not in rec  # null -> omitted
    assert rec["coding"]["source"] == "benchlm"


def test_benchlm_record_missing_category_omitted():
    entry = {"slug": "x", "displayScore": 10.0, "publicRankingMode": "v1",
             "scores": {"displayCategoryScores": {}}}
    rec = benchlm_record(entry, snapshot_date="2026-10-07")
    assert set(rec) == {"overall"}


def test_arena_elo_record_shape():
    rec = arena_elo_record(1114.0, snapshot_date="2023-12-07")
    assert rec == {"arena_elo": {"value": 1114.0, "unit": "elo", "source": "lmarena_hf",
                                  "version": None, "snapshot_date": "2023-12-07"}}


# --------------------------------------------------------------------------- #
# merge: union of disjoint keys, primary source wins on an actual collision   #
# --------------------------------------------------------------------------- #
def test_merge_benchmarks_unions_disjoint_sources():
    primary = {"mmlu": {"value": 70.6, "unit": "percent", "source": "hf_open_llm_leaderboard",
                         "version": "v1", "snapshot_date": None}}
    secondary = {"arena_elo": {"value": 1114.0, "unit": "elo", "source": "lmarena_hf",
                                "version": None, "snapshot_date": "2023-12-07"}}
    merged = merge_benchmarks(primary, secondary)
    assert set(merged) == {"mmlu", "arena_elo"}


def test_merge_benchmarks_primary_wins_on_key_collision():
    primary = {"overall": {"value": 1.0, "unit": "u", "source": "primary",
                            "version": None, "snapshot_date": None}}
    secondary = {"overall": {"value": 2.0, "unit": "u", "source": "secondary",
                              "version": None, "snapshot_date": None}}
    merged = merge_benchmarks(primary, secondary)
    assert merged["overall"]["source"] == "primary"


def test_merge_benchmarks_skips_none_sources():
    primary = None
    secondary = {"mmlu": {"value": 1.0, "unit": "u", "source": "s",
                           "version": None, "snapshot_date": None}}
    assert merge_benchmarks(primary, secondary) == secondary


def test_merge_benchmarks_all_none_is_empty_dict():
    assert merge_benchmarks(None, None) == {}


# --------------------------------------------------------------------------- #
# lookups: fixture-backed, no network                                        #
# --------------------------------------------------------------------------- #
def test_lookup_hf_leaderboard_finds_known_repo_id(tmp_path):
    table = [
        {"fullname": "mistralai/Mixtral-8x7B-Instruct-v0.1", "IFEval": 55.99, "BBH": 29.74},
        {"fullname": "some/other-model", "IFEval": 50.0},
    ]
    path = tmp_path / "contents.json"
    path.write_text(json.dumps(table), encoding="utf-8")

    rec = lookup_hf_leaderboard("mistralai/Mixtral-8x7B-Instruct-v0.1", table_path=path)
    assert rec["ifeval"]["value"] == 55.99
    assert rec["bbh"]["value"] == 29.74


def test_lookup_hf_leaderboard_returns_none_for_unknown_repo_id(tmp_path):
    path = tmp_path / "contents.json"
    path.write_text(json.dumps([{"fullname": "some/other-model", "IFEval": 50.0}]), encoding="utf-8")

    assert lookup_hf_leaderboard("anthropic/claude-2.0", table_path=path) is None


def test_lookup_benchlm_finds_known_slug(tmp_path):
    """Real top-level shape (verified against a live download): {"items":
    [...], "generatedAt": "...", ...} -- not {"models": [...]}."""
    payload = {
        "generatedAt": "2026-10-07T20:35:47.239Z",
        "items": [
            {"slug": "claude-4-sonnet", "displayScore": 38.49, "publicRankingMode": "bench-align-v5",
             "scores": {"displayCategoryScores": {"coding": 72.7}}},
        ],
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rec = lookup_benchlm("claude-4-sonnet", models_path=path)
    assert rec["overall"]["value"] == 38.49
    assert rec["overall"]["snapshot_date"] == "2026-10-07"
    assert rec["coding"]["value"] == 72.7


def test_lookup_benchlm_returns_none_for_unknown_slug(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"generatedAt": "2026-10-07T00:00:00Z", "items": []}),
                    encoding="utf-8")

    assert lookup_benchlm("not-a-real-slug", models_path=path) is None


def test_lookup_arena_elo_finds_known_model(tmp_path):
    csv_text = (
        "Model,MT-bench (score),Arena Elo rating,MMLU,License,Link\n"
        "mixtral-8x7b-instruct,8.3,1114,70.6,Apache-2.0,http://example\n"
        "claude-2.0,8.1,1131,78.5,Proprietary,http://example\n"
    )
    path = tmp_path / "leaderboard_table_20231207.csv"
    path.write_text(csv_text, encoding="utf-8")

    rec = lookup_arena_elo("mixtral-8x7b-instruct", table_path=path, snapshot_date="2023-12-07")
    assert rec["arena_elo"] == {"value": 1114.0, "unit": "elo", "source": "lmarena_hf",
                                 "version": None, "snapshot_date": "2023-12-07"}


def test_lookup_arena_elo_returns_none_for_unknown_model(tmp_path):
    csv_text = "Model,Arena Elo rating\nclaude-2.0,1131\n"
    path = tmp_path / "leaderboard.csv"
    path.write_text(csv_text, encoding="utf-8")

    assert lookup_arena_elo("gpt-4-1106-preview", table_path=path, snapshot_date="2023-12-07") is None


def test_restricted_unpickle_loads_pandas_and_blocks_other_globals(tmp_path):
    import os
    import pickle

    import pytest

    import pandas as pd

    from training.data.public_benchmarks import _restricted_unpickle, arena_category_elo_rows

    # object-dtype labels: the real Arena pickles predate pyarrow-backed strings
    table = pd.DataFrame([[1200.0, 1190.0, 1210.0, 500]],
                         index=pd.Index(["model-x"], dtype=object),
                         columns=pd.Index(["rating", "rating_q025", "rating_q975", "num_battles"],
                                          dtype=object))
    ok = tmp_path / "ok.pkl"
    ok.write_bytes(pickle.dumps({"text": {"full": {"leaderboard_table_df": table}}}))
    rows = arena_category_elo_rows(_restricted_unpickle(ok))
    assert rows == [{"category": "full", "arena_key": "model-x", "rating": 1200.0,
                     "rating_q025": 1190.0, "rating_q975": 1210.0, "num_battles": 500}]

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    bad = tmp_path / "bad.pkl"
    bad.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(pickle.UnpicklingError, match="blocked"):
        _restricted_unpickle(bad)
