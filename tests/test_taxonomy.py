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
