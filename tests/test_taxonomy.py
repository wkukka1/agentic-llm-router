import pytest

from router.taxonomy import (
    DOMAIN_LABELS,
    Domain,
    TaskType,
    domain_from_routerarena,
    task_type_from_arena_tags,
    task_type_from_dataset_name,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0 Computer science, information, and general works", Domain.CS_GENERAL),
        ("5 Science", Domain.SCIENCE),
        ("9 History", Domain.HISTORY),
    ],
)
def test_domain_from_routerarena_maps_dewey_prefix(raw, expected):
    assert domain_from_routerarena(raw) is expected


@pytest.mark.parametrize("raw", ["", "2 Religion", "not a domain", None])
def test_domain_from_routerarena_rejects_unmapped(raw):
    """Unmapped rows must return None so the loader drops rather than mislabels."""
    assert domain_from_routerarena(raw) is None


def test_domain_labels_are_unique_and_complete():
    assert len(DOMAIN_LABELS) == len(set(DOMAIN_LABELS)) == 9


@pytest.mark.parametrize(
    "name,expected",
    [
        ("LiveCodeBench", TaskType.CODE),
        ("MMLUPro_computer science", TaskType.CODE),
        ("MMLUPro_math", TaskType.MATH),
        ("AIME", TaskType.MATH),
        ("WMT19-de-en", TaskType.TRANSLATION),
        ("MMLU_formal_logic", TaskType.REASONING),
        ("MMLUPro_health", TaskType.FACTUAL_QA),
        ("NarrativeQA", TaskType.READING_COMPREHENSION),
        ("something unheard of", TaskType.OTHER),
    ],
)
def test_task_type_rule_precedence(name, expected):
    """Specific rules must win over the generic substrings that follow them."""
    assert task_type_from_dataset_name(name) is expected


def test_arena_tags_precedence_favours_code_then_math():
    assert task_type_from_arena_tags(is_code=True, is_math=True, is_creative_writing=False) is TaskType.CODE
    assert task_type_from_arena_tags(is_code=False, is_math=True, is_creative_writing=True) is TaskType.MATH
    assert task_type_from_arena_tags(is_code=False, is_math=False, is_creative_writing=True) is TaskType.CREATIVE_WRITING
    assert task_type_from_arena_tags(is_code=False, is_math=False, is_creative_writing=False) is TaskType.OTHER
