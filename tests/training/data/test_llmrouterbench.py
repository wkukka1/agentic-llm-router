"""LLMRouterBench loader (arXiv 2601.07206, performance-cost setting).

Fixture JSON shape confirmed against the vendored framework's own reference
parser: evaluations/LLMRouterBench/baselines/data_loader.py:236-269
(`BaselineDataLoader.load_records_iter`) and the file-path convention
documented in evaluations/LLMRouterBench/README.md's "Result File Structure".
"""

from __future__ import annotations

import json

import pytest

from router.config import Config
from training.data import schemas
from training.data.loaders import load_llmrouterbench
from training.data.normalize import make_query_id


def _write_result(root, dataset, split, model, records, *, ts="20260101_000000"):
    out = root / dataset / split / model
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": dataset,
        "split": split,
        "model_name": model,
        "records": records,
    }
    (out / f"{ts}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def bench_config(tmp_path):
    root = tmp_path / "bench"
    # gpqa: MC (4-choice); aime: free-form; one out-of-pool model; one missing score.
    _write_result(root, "gpqa", "test", "gpt-5", [
        {"index": 0, "origin_query": "Q gpqa 1", "prompt": "Answer the following...Q gpqa 1",
         "prediction": "A", "ground_truth": "A", "score": 1.0,
         "prompt_tokens": 100, "completion_tokens": 20, "cost": 0.01},
        {"index": 1, "origin_query": "Q gpqa 2", "prompt": "Answer the following...Q gpqa 2",
         "prediction": "B", "ground_truth": "A", "score": 0.0,
         "prompt_tokens": 90, "completion_tokens": 15, "cost": 0.009},
    ])
    _write_result(root, "gpqa", "test", "claude-sonnet-4", [
        {"index": 0, "origin_query": "Q gpqa 1", "prompt": "Answer the following...Q gpqa 1",
         "prediction": "A", "ground_truth": "A", "score": 1.0,
         "prompt_tokens": 100, "completion_tokens": 25, "cost": 0.02},
    ])
    _write_result(root, "aime", "test", "gpt-5", [
        {"index": 0, "origin_query": "Q aime 1", "prompt": "Q aime 1",
         "prediction": "42", "ground_truth": "42", "score": 1.0,
         "prompt_tokens": 50, "completion_tokens": 500, "cost": 0.05},
        # a failed / unscored generation -- must be dropped, never treated as 0.0
        {"index": 1, "origin_query": "Q aime 2", "prompt": "Q aime 2",
         "prediction": "", "ground_truth": "7", "score": None,
         "prompt_tokens": 50, "completion_tokens": 0, "cost": 0.001},
    ])
    # out-of-pool model (like the framework's own excluded "openrouter" reference rows)
    _write_result(root, "aime", "test", "not-in-pool-model", [
        {"index": 0, "origin_query": "Q aime 1", "prompt": "Q aime 1",
         "prediction": "42", "ground_truth": "42", "score": 1.0,
         "prompt_tokens": 50, "completion_tokens": 500, "cost": 0.05},
    ])
    # duplicate re-run of gpt-5/gpqa under a LATER timestamp -- only this one must survive
    _write_result(root, "gpqa", "test", "gpt-5", [
        {"index": 0, "origin_query": "Q gpqa 1", "prompt": "Answer the following...Q gpqa 1",
         "prediction": "A", "ground_truth": "A", "score": 1.0,
         "prompt_tokens": 100, "completion_tokens": 20, "cost": 0.011},
    ], ts="20260201_000000")

    return Config({
        "seed": 42,
        "paths": {"processed": str(tmp_path / "proc"), "splits": str(tmp_path / "splits")},
        "sources": {"llmrouterbench": {
            "local_dir": str(root),
            "pool": ["gpt-5", "claude-sonnet-4"],
        }},
        "multiple_choice": {"by_task": {"gpqa": 4, "mmlupro": 10}, "by_prefix": {}},
        "chance_correction": {"method": "normalized", "clip": True},
    }, root=tmp_path)


def test_loader_emits_canonical_schema(bench_config):
    df = load_llmrouterbench(bench_config)
    for col in schemas.RESPONSE_COLUMNS:
        assert col in df.columns
    assert set(df["source"].unique()) == {schemas.Source.LLMROUTERBENCH}


def test_out_of_pool_model_is_dropped(bench_config):
    df = load_llmrouterbench(bench_config)
    assert set(df["model_id"].unique()) == {"gpt-5", "claude-sonnet-4"}


def test_gpqa_is_mc_with_chance_correctable_choice_count(bench_config):
    df = load_llmrouterbench(bench_config)
    gpqa = df[df["dataset"] == "gpqa"]
    assert (gpqa["metric_type"] == schemas.MetricType.MC_ACCURACY).all()
    assert (gpqa["n_choices"] == 4).all()
    assert (gpqa["is_multiple_choice"]).all()


def test_mmlupro_would_not_inherit_the_routerbench_mmlu_prefix_rule(bench_config):
    """Review Focus item 1: this only proves the config's own by_task entry wins;
    the real protection is that llmrouterbench.yaml never shares a Config load
    with phase0.yaml's `mmlu*` -> 4 by_prefix rule at all."""
    from training.data.loaders import _mc_choices
    assert _mc_choices("mmlupro", bench_config) == 10


def test_aime_is_free_form_accuracy(bench_config):
    df = load_llmrouterbench(bench_config)
    aime = df[df["dataset"] == "aime"]
    assert (aime["metric_type"] == schemas.MetricType.ACCURACY).all()
    assert aime["n_choices"].isna().all()


def test_missing_score_is_dropped_not_defaulted_to_zero(bench_config):
    df = load_llmrouterbench(bench_config)
    aime_gpt5 = df[(df["dataset"] == "aime") & (df["model_id"] == "gpt-5")]
    assert len(aime_gpt5) == 1  # the score=None record never made it in
    assert aime_gpt5["score"].iloc[0] == 1.0


def test_duplicate_result_files_keep_only_the_latest_by_timestamp(bench_config):
    df = load_llmrouterbench(bench_config)
    gpt5_gpqa1 = df[(df["dataset"] == "gpqa") & (df["model_id"] == "gpt-5")
                     & (df["query"].str.contains("Q gpqa 1"))]
    assert len(gpt5_gpqa1) == 1
    assert float(gpt5_gpqa1["cost"].iloc[0]) == 0.011  # the later (20260201) file's value


def test_cost_and_tokens_pass_through_verbatim(bench_config):
    df = load_llmrouterbench(bench_config)
    row = df[(df["dataset"] == "aime") & (df["model_id"] == "gpt-5")].iloc[0]
    assert float(row["cost"]) == 0.05
    assert int(row["input_tokens"]) == 50
    assert int(row["output_tokens"]) == 500


def test_query_id_is_content_addressed_on_origin_query(bench_config):
    qid = make_query_id(schemas.Source.LLMROUTERBENCH, "aime", "Q aime 1")
    df = load_llmrouterbench(bench_config)
    assert qid in set(df["query_id"])


def test_missing_results_dir_raises_with_the_fix_in_the_message(tmp_path):
    cfg = Config({
        "seed": 42,
        "paths": {"processed": str(tmp_path / "proc"), "splits": str(tmp_path / "splits")},
        "sources": {"llmrouterbench": {"local_dir": str(tmp_path / "does-not-exist")}},
        "multiple_choice": {"by_task": {}, "by_prefix": {}},
    }, root=tmp_path)
    with pytest.raises(FileNotFoundError, match="download_llmrouterbench"):
        load_llmrouterbench(cfg)
