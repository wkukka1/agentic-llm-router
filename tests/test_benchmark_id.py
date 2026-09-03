"""Cross-source (task, sample_index) question identity (pool-expansion E3)."""

from __future__ import annotations

import pytest

from router.data.benchmark_id import (
    alignable_eval_names,
    lm_eval_sample_to_uid,
    rb_eval_to_task,
    rb_sample_to_uid,
)


@pytest.mark.parametrize("eval_name, task", [
    ("hellaswag", "hellaswag"),
    ("winogrande", "winogrande"),
    ("arc-challenge", "arc_challenge"),
    ("mmlu-abstract-algebra", "mmlu_abstract_algebra"),
    ("mmlu-high-school-us-history", "mmlu_high_school_us_history"),
    ("mmlu-professional-law", "mmlu_professional_law"),
])
def test_alignable_evals_map_to_lm_eval_tasks(eval_name, task):
    assert rb_eval_to_task(eval_name) == task


@pytest.mark.parametrize("eval_name", [
    "Chinese_character_riddles", "abstract2title", "consensus_summary",
    "grade-school-math", "mbpp", "mtbench", "mtbench-math", "bias_detection",
    "test-match", "accounting_audit",
])
def test_custom_evals_are_not_alignable(eval_name):
    assert rb_eval_to_task(eval_name) is None


def test_rb_and_lm_eval_produce_the_same_uid_for_the_same_question():
    rb = rb_sample_to_uid("mmlu-anatomy", "mmlu-anatomy.val.42")
    lm = lm_eval_sample_to_uid("mmlu_anatomy", 42)
    assert rb == lm == "benchmark:mmlu_anatomy:42"

    assert rb_sample_to_uid("hellaswag", "hellaswag.val.0") == "benchmark:hellaswag:0"
    assert rb_sample_to_uid("winogrande", "winogrande.dev.1266") == "benchmark:winogrande:1266"


def test_non_alignable_sample_returns_none():
    assert rb_sample_to_uid("consensus_summary", "consensus_summary.dev.3") is None
    # malformed tail
    assert rb_sample_to_uid("hellaswag", "hellaswag.val.x") is None


def test_alignable_eval_names_filters_a_mixed_list():
    names = ["hellaswag", "mmlu-anatomy", "consensus_summary", "chinese_poem", "winogrande"]
    got = alignable_eval_names(names)
    assert set(got) == {"hellaswag", "mmlu-anatomy", "winogrande"}
    assert got["mmlu-anatomy"] == "mmlu_anatomy"
