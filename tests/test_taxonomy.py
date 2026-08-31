"""The 10-domain label space and its four source mappings."""

import pytest

from router.taxonomy import (
    DOMAIN_DESCRIPTIONS,
    DOMAIN_LABELS,
    Domain,
    domain_from_arena_flags,
    domain_from_bigbench_task,
    domain_from_mmlu_pro,
    domain_from_routerarena_category,
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
    "category,expected",
    [
        ("math", Domain.SCIENCE_MATH),
        ("physics", Domain.SCIENCE_MATH),
        ("biology", Domain.SCIENCE_MATH),
        ("computer science", Domain.SOFTWARE_TECH),
        ("engineering", Domain.SOFTWARE_TECH),
        ("health", Domain.MEDICINE_HEALTH),
        ("psychology", Domain.MEDICINE_HEALTH),
        ("economics", Domain.BUSINESS_FINANCE),
        ("law", Domain.LAW_POLITICS),
        ("history", Domain.HUMANITIES),
        ("philosophy", Domain.HUMANITIES),
    ],
)
def test_mmlu_pro_categories_map(category, expected):
    assert domain_from_mmlu_pro(category) is expected


def test_mmlu_pro_own_other_is_dropped_not_mapped():
    """MMLU-Pro's 'other' is miscellany, not our meta class -- mapping it would
    teach the model that grab-bag exam questions are conversational."""
    assert domain_from_mmlu_pro("other") is None


@pytest.mark.parametrize(
    "category,expected",
    [
        ("61 Medicine and health", Domain.MEDICINE_HEALTH),
        ("00 Computer science, knowledge, and systems", Domain.SOFTWARE_TECH),
        ("51 Mathematics", Domain.SCIENCE_MATH),
        ("57 Biology", Domain.SCIENCE_MATH),
        ("34 Law", Domain.LAW_POLITICS),
        ("90 History", Domain.HUMANITIES),
        ("78 Music", Domain.ARTS_ENTERTAINMENT),
        ("40 Language", Domain.LANGUAGE),
    ],
)
def test_routerarena_subclass_maps(category, expected):
    """v1 used the Dewey *parent* class and put medicine under technology; the
    subclass is unambiguous, which is most of why v3 works."""
    assert domain_from_routerarena_category(category) is expected


def test_routerarena_unmapped_subclass_returns_none():
    assert domain_from_routerarena_category("99 Nonexistent") is None
    assert domain_from_routerarena_category("") is None


@pytest.mark.parametrize(
    "task,expected",
    [
        ("cs_algorithms", Domain.SOFTWARE_TECH),
        ("arithmetic", Domain.SCIENCE_MATH),
        ("emoji_movie", Domain.ARTS_ENTERTAINMENT),
        ("conlang_translation", Domain.LANGUAGE),
        ("anachronisms", Domain.HUMANITIES),
    ],
)
def test_bigbench_task_names_map(task, expected):
    assert domain_from_bigbench_task(task) is expected


def test_bigbench_synthetic_tasks_are_unmapped():
    """Tasks whose prompts resemble nothing a user would send stay out."""
    assert domain_from_bigbench_task("dyck_languages") is None
    assert domain_from_bigbench_task("") is None


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
    # The old wording claimed hardware for software_tech; hardware is now
    # unowned, pending enough examples to place it (only 31 in 2,441 rows).
    assert "hardware" not in software
