"""LLMRouterBench loader (arXiv 2601.07206, performance-cost setting).

Fixture JSON shape confirmed against the vendored framework's own reference
parser: evaluations/LLMRouterBench/baselines/data_loader.py:236-269
(`BaselineDataLoader.load_records_iter`) and the file-path convention
documented in evaluations/LLMRouterBench/README.md's "Result File Structure".
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from router.config import Config, load_config
from training.data import schemas
from training.data.loaders import _mc_choices, load_llmrouterbench
from training.data.normalize import make_query_id
from training.data.splits import make_splits


def _write_result(root, dataset, split, model, records, *, ts="20260101_000000", demo=False):
    out = root / dataset / split / model
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": dataset,
        "split": split,
        "model_name": model,
        "demo": demo,
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


def test_shipped_config_gives_mmlupro_ten_choices_and_never_shares_phase0s_prefix_rule():
    """Review Focus item 1, against the REAL configs/llmrouterbench.yaml (not a
    hand-built fixture dict) -- a prior version of this test asserted against its
    own fixture config and would have passed even if the shipped YAML were wrong
    or deleted. This loads the actual file and also proves phase0.yaml's `mmlu*`
    -> 4 rule genuinely lives in a different, never-loaded-together Config."""
    cfg = load_config("configs/llmrouterbench.yaml")
    assert _mc_choices("mmlupro", cfg) == 10
    assert _mc_choices("gpqa", cfg) == 4
    assert "mmlu" not in cfg.multiple_choice.by_prefix

    phase0 = load_config("configs/phase0.yaml")
    assert "llmrouterbench" not in phase0.get("source_order", [])
    assert "llmrouterbench" not in phase0.sources


def test_shipped_config_split_block_actually_works():
    """C1 regression: split.presplit was true with no source of metadata['origin'],
    which crashes build_splits.py's documented third pipeline step. Drives the real
    config's split block over a small synthetic frame the way build_splits.py does."""
    cfg = load_config("configs/llmrouterbench.yaml")
    assert cfg.split.presplit is False  # presplit needs metadata['origin'], which this
                                         # source's loader never sets (no train/test files)
    responses = pd.DataFrame({
        "query_id": ["q1", "q2", "q3", "q4"],
        "query": ["one", "two", "three", "four"],
        "model_id": ["gpt-5", "gpt-5", "claude-sonnet-4", "claude-sonnet-4"],
        "source": [schemas.Source.LLMROUTERBENCH] * 4,
    })
    splits = make_splits(responses, cfg)
    assert set(splits["train"]["query_ids"]) | set(splits["validation"]["query_ids"]) \
        | set(splits["test"]["query_ids"]) == {"q1", "q2", "q3", "q4"}


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


def test_empty_results_dir_raises_an_actionable_error_not_a_schema_error(tmp_path):
    """I4: an existing-but-empty results_dir previously fell through to
    coerce_response_frame's generic 'missing required columns' ValueError, which
    points at a broken loader instead of the real problem (never downloaded /
    nothing on disk yet)."""
    root = tmp_path / "bench"
    root.mkdir()
    cfg = Config({
        "seed": 42,
        "paths": {"processed": str(tmp_path / "proc"), "splits": str(tmp_path / "splits")},
        "sources": {"llmrouterbench": {"local_dir": str(root)}},
        "multiple_choice": {"by_task": {}, "by_prefix": {}},
    }, root=tmp_path)
    with pytest.raises(FileNotFoundError, match="download_llmrouterbench"):
        load_llmrouterbench(cfg)


def test_a_pool_that_matches_nothing_raises_an_actionable_error(tmp_path):
    """I4: every file present but filtered out by an over-narrow pool must not
    fall through to the generic 'missing required columns' schema error."""
    root = tmp_path / "bench"
    _write_result(root, "aime", "test", "gpt-5", [
        {"index": 0, "origin_query": "Q aime 1", "prompt": "Q aime 1",
         "prediction": "42", "ground_truth": "42", "score": 1.0,
         "prompt_tokens": 50, "completion_tokens": 500, "cost": 0.05},
    ])
    cfg = Config({
        "seed": 42,
        "paths": {"processed": str(tmp_path / "proc"), "splits": str(tmp_path / "splits")},
        "sources": {"llmrouterbench": {"local_dir": str(root), "pool": ["nobody-in-this-tree"]}},
        "multiple_choice": {"by_task": {}, "by_prefix": {}},
    }, root=tmp_path)
    with pytest.raises(ValueError, match="filtered out"):
        load_llmrouterbench(cfg)


def test_out_of_pool_drop_is_warned_about_by_name(bench_config):
    """I6: matches load_irt_router's behaviour -- a pool typo is otherwise invisible."""
    with pytest.warns(UserWarning, match="not-in-pool-model"):
        load_llmrouterbench(bench_config)


def test_demo_result_file_never_evicts_a_real_run(tmp_path):
    """C2: the reference framework marks smoke-test runs with `demo: true`
    (baseline_config_performance_cost.yaml's skip_demo) and a demo file can be
    written well after the real collection run. Dedup-by-latest-timestamp must
    not let a later demo stub replace real data."""
    root = tmp_path / "bench"
    _write_result(root, "aime", "test", "gpt-5", [
        {"index": 0, "origin_query": "Q real 1", "prompt": "Q real 1",
         "prediction": "42", "ground_truth": "42", "score": 1.0,
         "prompt_tokens": 50, "completion_tokens": 500, "cost": 0.05},
        {"index": 1, "origin_query": "Q real 2", "prompt": "Q real 2",
         "prediction": "7", "ground_truth": "7", "score": 1.0,
         "prompt_tokens": 50, "completion_tokens": 500, "cost": 0.05},
    ], ts="20260101_000000")
    _write_result(root, "aime", "test", "gpt-5", [
        {"index": 0, "origin_query": "Q demo 1", "prompt": "Q demo 1",
         "prediction": "1", "ground_truth": "1", "score": 1.0,
         "prompt_tokens": 5, "completion_tokens": 5, "cost": 0.001},
    ], ts="20260201_000000", demo=True)

    cfg = Config({
        "seed": 42,
        "paths": {"processed": str(tmp_path / "proc"), "splits": str(tmp_path / "splits")},
        "sources": {"llmrouterbench": {"local_dir": str(root), "pool": ["gpt-5"]}},
        "multiple_choice": {"by_task": {}, "by_prefix": {}},
    }, root=tmp_path)
    df = load_llmrouterbench(cfg)
    assert len(df) == 2
    assert set(df["query"]) == {"Q real 1", "Q real 2"}


def test_unknown_dataset_is_dropped_not_silently_labelled_accuracy(tmp_path):
    """I3: LLMRouterBench's raw results tree includes datasets outside the
    performance-cost setting (e.g. `arc-agi`, explicitly commented out of
    baseline_config_performance_cost.yaml's own whitelist). Admitting them as a
    generic `accuracy` row would violate the "only performance-cost" Global
    Constraint and silently widen the training item pool."""
    root = tmp_path / "bench"
    _write_result(root, "arc-agi", "test", "gpt-5", [
        {"index": 0, "origin_query": "Q arc 1", "prompt": "Q arc 1",
         "prediction": "x", "ground_truth": "x", "score": 1.0,
         "prompt_tokens": 10, "completion_tokens": 10, "cost": 0.01},
    ])
    cfg = Config({
        "seed": 42,
        "paths": {"processed": str(tmp_path / "proc"), "splits": str(tmp_path / "splits")},
        "sources": {"llmrouterbench": {"local_dir": str(root), "pool": ["gpt-5"]}},
        "multiple_choice": {"by_task": {}, "by_prefix": {}},
    }, root=tmp_path)
    with pytest.raises(ValueError, match="filtered out"):
        load_llmrouterbench(cfg)  # the only file present is the unknown dataset


def test_split_outside_the_configured_allowlist_is_dropped(tmp_path):
    """I8: query_id is content-addressed on (source, dataset, text) and excludes
    split, so an unwhitelisted split (a dev/scratch run) silently averages into
    a real one downstream instead of erroring -- must never reach the frame."""
    root = tmp_path / "bench"
    _write_result(root, "aime", "test", "gpt-5", [
        {"index": 0, "origin_query": "Q aime 1", "prompt": "Q aime 1",
         "prediction": "42", "ground_truth": "42", "score": 1.0,
         "prompt_tokens": 50, "completion_tokens": 500, "cost": 0.05},
    ])
    _write_result(root, "aime", "scratch", "gpt-5", [
        {"index": 0, "origin_query": "Q aime 1", "prompt": "Q aime 1",
         "prediction": "42", "ground_truth": "42", "score": 0.0,
         "prompt_tokens": 50, "completion_tokens": 500, "cost": 0.05},
    ])
    cfg = Config({
        "seed": 42,
        "paths": {"processed": str(tmp_path / "proc"), "splits": str(tmp_path / "splits")},
        "sources": {"llmrouterbench": {
            "local_dir": str(root), "pool": ["gpt-5"], "splits": ["test"],
        }},
        "multiple_choice": {"by_task": {}, "by_prefix": {}},
    }, root=tmp_path)
    df = load_llmrouterbench(cfg)
    assert len(df) == 1
    assert df["split"].iloc[0] == "test"
