from __future__ import annotations

import pandas as pd
import pytest

from router.config import Config
from training.data import schemas
from evaluation.data.judge_responses import (
    _render_response,
    duplicated_routerbench_query_ids,
    load_routerbench_response_text,
    response_text_lookup,
)
from training.data.model_registry import canonical_model_id
from training.data.normalize import make_query_id, render_prompt


@pytest.fixture
def cfg(tmp_path):
    local_dir = tmp_path / "routerbench"
    local_dir.mkdir()
    wide = pd.DataFrame({
        "sample_id": ["s0", "s1"],
        "prompt": ["what is 2+2?", "name a color"],
        "eval_name": ["gsm8k", "gsm8k"],
        "oracle_model_to_route_to": ["gpt-4-1106-preview", "gpt-4-1106-preview"],
        "gpt-4-1106-preview": [1.0, 0.0],
        "gpt-4-1106-preview|total_cost": [0.01, 0.01],
        "gpt-4-1106-preview|model_response": ["['four']", "['blue']"],
        "mixtral-8x7b-instruct": [0.0, 1.0],
        "mixtral-8x7b-instruct|total_cost": [0.001, 0.001],
        "mixtral-8x7b-instruct|model_response": ["['five']", "['red']"],
    })
    wide.to_pickle(local_dir / "routerbench_0shot.pkl")
    data = {"sources": {"routerbench": {"local_dir": "routerbench", "hf_repo": "x/y", "shots": ["0shot"]}}}
    return Config(data, root=tmp_path)


def test_ids_match_make_query_id_and_canonical_model_id(cfg):
    df = load_routerbench_response_text(cfg, shot="0shot")
    assert set(df.columns) == {"query_id", "model_id", "response_text"}
    assert len(df) == 4  # 2 queries x 2 models

    expected_qid_s0 = make_query_id(schemas.Source.ROUTERBENCH, "gsm8k", render_prompt("what is 2+2?"))
    expected_qid_s1 = make_query_id(schemas.Source.ROUTERBENCH, "gsm8k", render_prompt("name a color"))
    assert set(df["query_id"]) == {expected_qid_s0, expected_qid_s1}
    assert set(df["model_id"]) == {
        canonical_model_id("gpt-4-1106-preview"), canonical_model_id("mixtral-8x7b-instruct"),
    }

    row = df[(df["query_id"] == expected_qid_s0) &
             (df["model_id"] == canonical_model_id("gpt-4-1106-preview"))]
    assert row["response_text"].iloc[0] == "four"


def test_response_text_lookup_filters_by_query_ids(cfg):
    df = load_routerbench_response_text(cfg, shot="0shot")
    one_qid = df["query_id"].iloc[0]
    lookup = response_text_lookup(cfg, query_ids=[one_qid], shot="0shot")
    assert set(qid for qid, _ in lookup) == {one_qid}
    assert len(lookup) == 2  # both models for that one query


def test_non_zero_shot_not_supported(cfg):
    with pytest.raises(NotImplementedError):
        load_routerbench_response_text(cfg, shot="5shot")


def test_duplicate_prompts_are_dropped(tmp_path):
    local_dir = tmp_path / "routerbench"
    local_dir.mkdir()
    wide = pd.DataFrame({
        "sample_id": ["s0", "s1", "s2"],
        "prompt": ["what is 2+2?", "name a color", "what is 2+2?"],
        "eval_name": ["gsm8k", "gsm8k", "gsm8k"],
        "gpt-4-1106-preview": [1.0, 0.0, 0.0],
        "gpt-4-1106-preview|total_cost": [0.01, 0.01, 0.01],
        "gpt-4-1106-preview|model_response": ["['four']", "['blue']", "['five']"],
    })
    wide.to_pickle(local_dir / "routerbench_0shot.pkl")
    cfg = Config({"sources": {"routerbench": {"local_dir": "routerbench"}}}, root=tmp_path)
    dup_qid = make_query_id(schemas.Source.ROUTERBENCH, "gsm8k", "what is 2+2?")

    with pytest.warns(UserWarning, match="dropped 1"):
        df = load_routerbench_response_text(cfg)
    assert dup_qid not in set(df["query_id"])
    assert len(df) == 1
    assert duplicated_routerbench_query_ids(cfg) == {dup_qid}


@pytest.mark.parametrize("raw, expected", [
    ("['four']", "four"),
    ("['[1, 2, 3]']", "[1, 2, 3]"),        # list-literal answer inside the wrapper
    ("[1, 2, 3]", "[1, 2, 3]"),            # bare list literal is not rewritten
    ("['a', 'b']", "a\n\nb"),
    ("plain text", "plain text"),
])
def test_render_response(raw, expected):
    assert _render_response(raw) == expected


def test_render_response_survives_deep_nesting():
    raw = "[" * 10_000 + "]" * 10_000
    assert _render_response(raw) == raw
