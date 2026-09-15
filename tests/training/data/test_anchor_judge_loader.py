from __future__ import annotations

import pandas as pd
import pytest

from router.config import Config
from training.data import schemas
from training.data.loaders import load_anchor_judge
from training.data.facade import TrainingData


@pytest.fixture
def cfg(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    out_dir = processed / "anchor_judge"
    out_dir.mkdir()

    pd.DataFrame({
        "query_id": ["routerbench:gsm8k:aaaa"],
        "query": ["what is 2+2?"],
    }).to_parquet(processed / "queries.parquet")

    judgments = pd.DataFrame({
        "query_id": ["routerbench:gsm8k:aaaa"] * 3,
        "anchor_model_id": ["mixtral-8x7b-instruct"] * 3,
        "candidate_model_id": ["gpt-4-1106-preview", "gpt-4-1106-preview", "claude-v2"],
        "gain": [2, 2, 0],
        "anchor_is_a": [True, True, True],
        "reason": ["clearer", "SWAPPED-should-be-dropped", "UNPARSED-should-be-dropped"],
        "raw": ["{}", "{}", "garbage"],
        "parsed_ok": [True, True, False],
        "judge_provider": ["dummy"] * 3,
        "judge_model": ["dummy"] * 3,
        "swapped": [False, True, False],
    })
    judgments.to_parquet(out_dir / "judgments.parquet")

    data = {
        "paths": {"processed": "processed"},
        "anchor_judge": {"output_dir": "processed/anchor_judge"},
    }
    return Config(data, root=tmp_path)


def test_load_anchor_judge_shape_and_drops_swapped_rows(cfg):
    with pytest.warns(UserWarning, match="dropped 1 judgments"):
        df = load_anchor_judge(cfg)
    # one battle -> two participant rows; swapped and unparsed rows excluded
    assert len(df) == 2
    assert set(df["source"]) == {schemas.Source.ANCHOR_JUDGE}
    assert set(df["metric_type"]) == {schemas.MetricType.JUDGE_PREFERENCE}
    assert set(df["model_id"]) == {"mixtral-8x7b-instruct", "gpt-4-1106-preview"}
    assert (df["query"] == "what is 2+2?").all()

    anchor_row = df[df["model_id"] == "mixtral-8x7b-instruct"].iloc[0]
    candidate_row = df[df["model_id"] == "gpt-4-1106-preview"].iloc[0]
    # gain=+2 favors the candidate -> anchor loses (score 0.0), candidate wins (1.0)
    assert anchor_row["score"] == 0.0
    assert candidate_row["score"] == 1.0
    assert anchor_row["metadata"]["margin"] == 2
    assert anchor_row["metadata"]["opponent_model_id"] == "gpt-4-1106-preview"


def test_missing_judgments_file_raises_file_not_found(tmp_path):
    cfg = Config({"paths": {"processed": "processed"},
                   "anchor_judge": {"output_dir": "processed/anchor_judge"}}, root=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_anchor_judge(cfg)


def test_round_trips_through_phase1data_pairwise(cfg):
    responses = load_anchor_judge(cfg)
    d = TrainingData(cfg=cfg, responses=responses, queries=pd.DataFrame(),
                    models=pd.DataFrame(), splits={})
    pairs = d.pairwise(source=schemas.Source.ANCHOR_JUDGE, models="all")
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["model_a"] == "mixtral-8x7b-instruct"
    assert row["model_b"] == "gpt-4-1106-preview"
    assert row["winner"] == "b"
    assert row["score_a"] == 0.0
