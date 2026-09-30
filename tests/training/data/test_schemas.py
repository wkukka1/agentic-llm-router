"""Controlled-vocabulary contracts in training.data.schemas."""

from __future__ import annotations

from training.data import schemas


def test_llmrouterbench_source_is_registered_and_absolute():
    assert schemas.Source.LLMROUTERBENCH in schemas.Source.ALL
    assert schemas.Source.LLMROUTERBENCH not in schemas.Source.PAIRWISE


def test_llm_judge_score_metric_is_registered_as_correctness_not_pairwise():
    assert schemas.MetricType.LLM_JUDGE_SCORE in schemas.MetricType.ALL
    assert schemas.MetricType.LLM_JUDGE_SCORE in schemas.MetricType.CORRECTNESS
    assert schemas.MetricType.LLM_JUDGE_SCORE not in schemas.MetricType.PAIRWISE
    assert schemas.METRIC_RANGES[schemas.MetricType.LLM_JUDGE_SCORE] == (0.0, 1.0)
