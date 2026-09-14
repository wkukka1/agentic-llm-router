from __future__ import annotations

import pandas as pd
import pytest

from router.data import schemas
from router.data.loaders import load_routerbench
from router.data.model_registry import canonical_model_id, is_canonical
from router.data.normalize import content_hash, make_query_id, render_prompt
from router.data.quality import check_responses
from router.data.response_matrix import build_tables, collapse_duplicate_observations, pivot


# --------------------------------------------------------------------------- #
# schema                                                                      #
# --------------------------------------------------------------------------- #
def test_coerce_requires_required_columns():
    with pytest.raises(ValueError):
        schemas.coerce_response_frame(pd.DataFrame({"query_id": ["x"]}))


def test_coerce_adds_nullable_columns_as_na(toy_responses):
    for col in schemas.RESPONSE_COLUMNS:
        assert col in toy_responses.columns
    assert toy_responses["cost"].isna().all()          # never fabricated to 0
    assert toy_responses["input_tokens"].isna().all()


def test_missing_values_not_zero_filled(toy_responses):
    # cost / tokens / latency must be NA, not 0
    assert toy_responses[["cost", "latency", "input_tokens", "output_tokens"]].isna().all().all()


# --------------------------------------------------------------------------- #
# normalize / ids                                                             #
# --------------------------------------------------------------------------- #
def test_render_prompt_unpacks_stringified_list():
    assert render_prompt("['a', 'b']") == "a\n\nb"
    assert render_prompt("plain") == "plain"


def test_query_id_is_deterministic_and_content_addressed():
    a = make_query_id("routerbench", "gsm8k", "  Hello  World ")
    b = make_query_id("routerbench", "gsm8k", "hello world")
    assert a == b                      # normalization collapses ws/case
    assert content_hash("Hello") == content_hash("hello")


# --------------------------------------------------------------------------- #
# model registry                                                              #
# --------------------------------------------------------------------------- #
def test_model_aliases_canonicalize():
    assert canonical_model_id("mistralai/mixtral-8x7b-chat") == "mixtral-8x7b-instruct"
    assert canonical_model_id("meta/llama-2-70b-chat") == "llama-2-70b-chat"
    assert is_canonical("gpt-4-1106-preview")
    assert not is_canonical("some-unseen-model")


def test_low_confidence_aliases_are_not_merged():
    from router.data.model_registry import alias_confidence

    # distinct GPT-4 checkpoints must never collapse together
    assert canonical_model_id("gpt-4-0613") == "gpt-4-0613"
    assert canonical_model_id("gpt-4-0314") == "gpt-4-0314"
    # RouterBench's ambiguous claude-v2 is kept apart from Arena's claude-2.x
    assert canonical_model_id("claude-2.1") == "claude-2.1"
    assert canonical_model_id("claude-v2") == "claude-v2"
    # cross-source Mixtral IS merged (only one instruct checkpoint ever existed)
    assert canonical_model_id("mixtral-8x7b-instruct-v0.1") == "mixtral-8x7b-instruct"
    assert alias_confidence("mixtral-8x7b-instruct-v0.1") == "high"


# --------------------------------------------------------------------------- #
# quality checks                                                              #
# --------------------------------------------------------------------------- #
def test_quality_clean_frame_has_no_errors(toy_responses):
    report = check_responses(toy_responses)
    assert report.ok(), report.render()


def test_quality_detects_out_of_range_score(toy_responses):
    bad = toy_responses.copy()
    bad.loc[0, "score"] = 1.5
    report = check_responses(bad)
    assert any(i.code == "score_out_of_range" for i in report.errors)


def test_quality_detects_duplicate_observation(toy_responses):
    dup = pd.concat([toy_responses, toy_responses.iloc[[0]]], ignore_index=True)
    report = check_responses(dup)
    assert any(i.code == "duplicate_observation" for i in report.errors)


def test_quality_allows_multiple_arena_battles_same_cell(toy_responses):
    # a second battle on the same (query, model) with a different opponent
    extra = toy_responses.iloc[[5]].copy()
    extra["metadata"] = [{"pair_id": "p2", "role": "a", "opponent_model_id": "x"}]
    report = check_responses(pd.concat([toy_responses, extra], ignore_index=True))
    assert not any(i.code == "duplicate_observation" for i in report.errors)
    assert not any(i.code == "duplicate_pairwise_battle" for i in report.errors)


def test_quality_detects_impossible_tokens(toy_responses):
    bad = toy_responses.copy()
    bad["input_tokens"] = bad["input_tokens"].astype("Int64")
    bad.loc[0, "input_tokens"] = -5
    report = check_responses(bad)
    assert any(i.code == "impossible_input_tokens" for i in report.errors)


def test_quality_warns_on_missing_mc_metadata(toy_responses):
    report = check_responses(toy_responses)
    assert any(i.code == "missing_mc_choices" for i in report.warnings)


def test_summary_answers_acceptance_questions(toy_responses):
    report = check_responses(toy_responses)
    s = report.summary
    for key in ("Total observations", "Unique queries", "Unique models",
                "Unique datasets", "Observations by metric", "Observations by source",
                "Missing cost", "Multiple-choice observations"):
        assert key in s


# --------------------------------------------------------------------------- #
# response matrix                                                             #
# --------------------------------------------------------------------------- #
def test_collapse_duplicates_averages_and_records_fanin(toy_responses):
    d = pd.concat([toy_responses, toy_responses.iloc[[0]]], ignore_index=True)
    d.loc[d.index[-1], "score"] = 0.0        # 1.0 and 0.0 -> mean 0.5
    out, n = collapse_duplicate_observations(d)
    assert n == 2
    row = out[(out["query_id"] == "routerbench:gsm8k:aaaa")
              & (out["model_id"] == "gpt-4-1106-preview")]
    assert len(row) == 1
    assert row["score"].iloc[0] == pytest.approx(0.5)
    assert row["metadata"].iloc[0]["collapsed_from"] == 2


def test_pivot_keeps_metrics_separate(toy_responses):
    tables = build_tables(cfg=_DummyCfg(), responses=toy_responses)
    R = pivot(tables["responses"], schemas.MetricType.ACCURACY)
    assert "gpt-4-1106-preview" in R.columns
    assert R.loc["routerbench:gsm8k:aaaa", "gpt-4-1106-preview"] == 1.0
    with pytest.raises(ValueError):
        pivot(tables["responses"], "nonexistent_metric")


def test_metric_type_preserved_through_pipeline(toy_responses):
    tables = build_tables(cfg=_DummyCfg(), responses=toy_responses)
    metrics = set(tables["responses"]["metric_type"])
    assert schemas.MetricType.ACCURACY in metrics
    assert schemas.MetricType.MC_ACCURACY in metrics
    assert schemas.MetricType.ARENA_PREFERENCE in metrics


# --------------------------------------------------------------------------- #
# real RouterBench loader (skips if data absent)                              #
# --------------------------------------------------------------------------- #
def test_routerbench_loader_smoke(cfg):
    try:
        df = load_routerbench(cfg)
    except FileNotFoundError:
        pytest.skip("RouterBench data not present")
    df = schemas.coerce_response_frame(df)
    assert len(df) > 0
    assert df["score"].between(0, 1).all()
    assert set(df["source"]) == {schemas.Source.ROUTERBENCH}
    # metric identity retained
    assert df["metric_type"].isin(schemas.MetricType.ALL).all()


def test_routerbench_5shot_item_identity(cfg):
    """E2 policy: 0-/5-shot are separate items (``:5shot`` suffix) that share the
    underlying question text -> the same content hash -> the same split group."""
    if "5shot" not in list(cfg.sources.routerbench.shots):
        pytest.skip("5shot not enabled in config")
    try:
        df = load_routerbench(cfg)
    except FileNotFoundError:
        pytest.skip("RouterBench data not present")
    from router.data.normalize import content_hash

    shot = df["metadata"].map(lambda m: (m or {}).get("shots"))
    five = df[shot == "5shot"]
    zero = df[shot == "0shot"]
    assert len(five) and len(zero)
    # every 5-shot query_id carries the suffix; no 0-shot one does
    assert five["query_id"].str.endswith(":5shot").all()
    assert not zero["query_id"].str.endswith(":5shot").any()
    # 5-shot item text == its 0-shot sibling's text (question, not exemplars)
    zero_hashes = {content_hash(q) for q in zero["query"].unique()}
    five_hashes = {content_hash(q) for q in five["query"].unique()}
    assert five_hashes <= zero_hashes
    # stripping the suffix recovers the sibling id
    zero_ids = set(zero["query_id"])
    assert {q[: -len(":5shot")] for q in five["query_id"].unique()} <= zero_ids


class _DummyCfg:
    """Minimal config stub for build_tables when we pass responses directly."""

    def get(self, key, default=None):
        return {
            "chance_correction": {"method": "normalized", "clip": True,
                                  "warn_on_missing_choices": False},
        }.get(key, default)
