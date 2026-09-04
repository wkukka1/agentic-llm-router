"""The 10-domain label space and its four source mappings."""

import pytest

from router.taxonomy import (
    DOMAIN_DESCRIPTIONS,
    DOMAIN_LABELS,
    Domain,
    domain_from_arena_flags,
)


def test_label_space_is_ten_distinct_domains():
    assert len(DOMAIN_LABELS) == len(set(DOMAIN_LABELS)) == 10


def test_every_domain_has_a_description():
    """Descriptions are the zero-shot anchors; a missing one silently degrades."""
    assert set(DOMAIN_DESCRIPTIONS) == set(DOMAIN_LABELS)
    assert all(len(v) > 20 for v in DOMAIN_DESCRIPTIONS.values())


def test_catch_all_domains_exist():
    """v1 failed partly because real traffic had nowhere to go."""
    assert Domain.META_OTHER in set(Domain)
    assert Domain.PERSONAL_LIFE in set(Domain)


@pytest.mark.parametrize(
    "flags,expected",
    [
        ({"is_code": True, "is_math": False, "is_creative_writing": False}, Domain.SOFTWARE_TECH),
        ({"is_code": False, "is_math": True, "is_creative_writing": False}, Domain.SCIENCE_MATH),
        ({"is_code": False, "is_math": False, "is_creative_writing": True}, Domain.ARTS_ENTERTAINMENT),
    ],
)
def test_arena_flags_map(flags, expected):
    assert domain_from_arena_flags(**flags) is expected


def test_unflagged_arena_row_is_none_not_a_guess():
    """An unflagged prompt could be any of the ten domains. Guessing would
    poison training; these rows are what the hand labelling exists for."""
    assert domain_from_arena_flags(
        is_code=False, is_math=False, is_creative_writing=False
    ) is None


def test_arena_flag_precedence_is_code_then_math():
    assert domain_from_arena_flags(
        is_code=True, is_math=True, is_creative_writing=True) is Domain.SOFTWARE_TECH
    assert domain_from_arena_flags(
        is_code=False, is_math=True, is_creative_writing=True) is Domain.SCIENCE_MATH


class TestDomainMerges:
    """Optional coarser grouping, applied at inference not at training."""

    def test_merge_map_only_touches_the_two_intended_pairs(self):
        from router.taxonomy import DOMAIN_LABELS, MERGED_DOMAIN_LABELS, apply_domain_merges

        assert len(MERGED_DOMAIN_LABELS) == len(DOMAIN_LABELS) - 2
        for d in DOMAIN_LABELS:
            merged = apply_domain_merges(d)
            if d in ("business_finance", "law_politics"):
                assert merged == "business_law"
            elif d in ("humanities", "arts_entertainment"):
                assert merged == "culture"
            else:
                assert merged == d, f"{d} should be untouched"

    def test_science_and_software_stay_separate(self):
        """Merging them scores better (0.785 vs 0.773) and is deliberately not
        done: maths and code route to different models, so collapsing them buys
        a metric by destroying a distinction the router needs."""
        from router.taxonomy import apply_domain_merges

        assert apply_domain_merges("science_math") != apply_domain_merges("software_tech")

    def test_merge_is_idempotent(self):
        from router.taxonomy import MERGED_DOMAIN_LABELS, apply_domain_merges

        for d in MERGED_DOMAIN_LABELS:
            assert apply_domain_merges(d) == d

    def test_unknown_domain_passes_through(self):
        from router.taxonomy import apply_domain_merges

        assert apply_domain_merges("not_a_domain") == "not_a_domain"

    def test_head_merges_by_summing_probabilities(self):
        """The merged score for a group must be the sum of its members', not
        the max -- that is what makes post-hoc merging beat retraining."""
        import numpy as np

        from router.inference import DomainHead

        labels = ["business_finance", "law_politics", "software_tech"]
        proba = np.array([[0.3, 0.3, 0.4]])
        merged, names = DomainHead._merge(proba, labels)
        assert names == ["business_law", "software_tech"]
        assert merged[0, names.index("business_law")] == pytest.approx(0.6)
        assert merged.sum() == pytest.approx(1.0)


def test_software_tech_owns_questions_about_ai():
    """Agreed boundary: questions *about* AI/ML systems are software_tech;
    meta_other is for questions about the assistant itself.

    This was the single largest error source against externally-labelled sets
    (18 of 44 errors across 402 prompts) and it was a definition disagreement,
    not a model failure. Pinned so it does not drift back."""
    from router.taxonomy import DOMAIN_DESCRIPTIONS

    software = DOMAIN_DESCRIPTIONS["software_tech"].lower()
    assert "machine learning" in software or "ai" in software
    assert "model" in software
    meta = DOMAIN_DESCRIPTIONS["meta_other"].lower()
    assert "assistant" in meta
    # Hardware lives here too: a separate class was rejected on the data
    # (only 31 hardware-flavoured prompts in 2,441).
    assert "hardware" in software


class TestTaskTypes:
    """Task type: what the user wants *done*, orthogonal to domain."""

    def test_seven_distinct_tasks(self):
        from router.tasktype import TASK_DESCRIPTIONS, TASK_LABELS

        assert len(TASK_LABELS) == len(set(TASK_LABELS)) == 7
        assert set(TASK_DESCRIPTIONS) == set(TASK_LABELS)

    def test_media_is_its_own_task_not_a_kind_of_create(self):
        """A request for an image does not go to a cheaper or dearer language
        model, it goes to a different kind of model entirely. Splitting it out
        of `create` costs the rest of the taxonomy nothing measurable
        (-0.013 top-1, 95% CI [-0.026, 0.000]) and reaches F1 0.725 itself."""
        from router.tasktype import TASK_LABELS, TaskType

        assert TaskType.MEDIA.value in TASK_LABELS
        assert TaskType.MEDIA is not TaskType.CREATE

    def test_dolly_question_variants_collapse_to_one_task(self):
        """open/general/closed_qa differ only by whether a passage was
        attached (100% vs 0% context), which an instruction-only classifier
        cannot recover. They are one task."""
        from router.tasktype import TaskType, task_from_dolly

        for c in ("open_qa", "general_qa", "closed_qa"):
            assert task_from_dolly(c) is TaskType.ANSWER

    def test_every_dolly_category_maps(self):
        from router.tasktype import DOLLY_MAP, task_from_dolly

        for category in DOLLY_MAP:
            assert task_from_dolly(category) is not None
        assert task_from_dolly("not_a_category") is None
        assert task_from_dolly("") is None

    def test_task_axis_is_independent_of_domain(self):
        """The two label spaces must not overlap -- if they shared names, a
        consumer could not tell which axis a prediction came from."""
        from router.tasktype import TASK_LABELS
        from router.taxonomy import DOMAIN_LABELS

        assert not (set(TASK_LABELS) & set(DOMAIN_LABELS))
