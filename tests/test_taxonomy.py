"""The capability label space, and the two ways it gets assigned."""

import pytest

from router.taxonomy import (
    CAPABILITY_LABELS,
    Capability,
    capability_from_arena_flags,
    capability_from_dataset_name,
    capability_hint,
)


def test_label_space_is_five_unique_capabilities():
    assert len(CAPABILITY_LABELS) == len(set(CAPABILITY_LABELS)) == 5
    assert "other" in CAPABILITY_LABELS


@pytest.mark.parametrize(
    "flags,expected",
    [
        ({"is_code": True, "is_math": False, "is_creative_writing": False}, Capability.CODE),
        ({"is_code": False, "is_math": True, "is_creative_writing": False}, Capability.MATH),
        ({"is_code": False, "is_math": False, "is_creative_writing": True}, Capability.CREATIVE_WRITING),
        ({"is_code": False, "is_math": False, "is_creative_writing": False}, Capability.OTHER),
    ],
)
def test_arena_flags_map_to_capability(flags, expected):
    assert capability_from_arena_flags(**flags) is expected


def test_arena_flag_precedence_puts_code_first():
    """Flags are not mutually exclusive; code constrains routing most."""
    assert capability_from_arena_flags(
        is_code=True, is_math=True, is_creative_writing=True
    ) is Capability.CODE
    assert capability_from_arena_flags(
        is_code=False, is_math=True, is_creative_writing=True
    ) is Capability.MATH


def test_absent_flags_mean_other_not_missing():
    """An unflagged prompt is genuinely OTHER -- roughly half of real traffic."""
    assert capability_from_arena_flags(
        is_code=False, is_math=False, is_creative_writing=False
    ) is Capability.OTHER


@pytest.mark.parametrize(
    "name,expected",
    [
        ("LiveCodeBench", Capability.CODE),
        ("MMLUPro_computer science", Capability.CODE),
        ("MATH", Capability.MATH),
        ("AIME", Capability.MATH),
        ("GSM8K", Capability.MATH),
        ("MMLUPro_math", Capability.MATH),
        ("WMT19-de-en", Capability.TRANSLATION),
        ("MedMCQA", Capability.OTHER),
        ("QANTA_History", Capability.OTHER),
        ("", Capability.OTHER),
    ],
)
def test_dataset_provenance_maps_to_capability(name, expected):
    assert capability_from_dataset_name(name) is expected


def test_specific_dataset_rules_beat_generic_substrings():
    """'MMLUPro_computer science' must be CODE despite matching nothing else."""
    assert capability_from_dataset_name("MMLUPro_computer science") is Capability.CODE
    # 'mathqa' must not be shadowed by the bare 'math' rule ordering.
    assert capability_from_dataset_name("MathQA") is Capability.MATH


def test_knowledge_recall_falls_through_to_other():
    """RouterArena is mostly MCQ recall; OTHER is the correct label, not a gap."""
    for name in ("PubMedQA", "OpenTDB_Geography", "MusicTheoryBench", "SocialiQA"):
        assert capability_from_dataset_name(name) is Capability.OTHER


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("```python\ndef f(): pass\n```", Capability.CODE),
        ("Prove that sqrt(2) is irrational", Capability.MATH),
        ("write me a poem about rain", Capability.CREATIVE_WRITING),
        ("translate this into French", Capability.TRANSLATION),
        ("hey how are you", Capability.OTHER),
    ],
)
def test_capability_hint_is_a_usable_zero_training_baseline(prompt, expected):
    assert capability_hint(prompt) is expected


def test_capability_hint_handles_empty_input():
    assert capability_hint("") is Capability.OTHER
