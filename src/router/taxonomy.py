"""Canonical label spaces for the router's upstream classifiers.

Two orthogonal axes are modelled, because they answer different routing
questions:

* ``Domain``   -- what the prompt is *about* (topic). Sourced from RouterArena's
                  Dewey-style ``Domain`` column.
* ``TaskType`` -- what the model has to *do* (capability). Sourced from dataset
                  provenance (RouterArena ``Dataset name``) and from the LMArena
                  ``category_tag`` binary flags.

Keeping them separate matters: "math" is a capability that shows up inside the
science, technology and CS domains, and a router that collapses the two loses
the ability to say "physics word problem -> reasoning model".
"""

from __future__ import annotations

import re
from enum import StrEnum


class Domain(StrEnum):
    """Topic of the prompt. Mirrors RouterArena's 9 populated Dewey classes."""

    CS_GENERAL = "cs_general"
    PHILOSOPHY_PSYCHOLOGY = "philosophy_psychology"
    SOCIAL_SCIENCE = "social_science"
    LANGUAGE = "language"
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    ARTS_RECREATION = "arts_recreation"
    LITERATURE = "literature"
    HISTORY = "history"


class TaskType(StrEnum):
    """Capability the prompt demands."""

    CODE = "code"
    MATH = "math"
    REASONING = "reasoning"
    FACTUAL_QA = "factual_qa"
    READING_COMPREHENSION = "reading_comprehension"
    TRANSLATION = "translation"
    CREATIVE_WRITING = "creative_writing"
    OTHER = "other"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


DOMAIN_LABELS: list[str] = [d.value for d in Domain]
TASK_TYPE_LABELS: list[str] = [t.value for t in TaskType]
DIFFICULTY_LABELS: list[str] = [d.value for d in Difficulty]

#: RouterArena encodes the Dewey top-level class as a leading digit, e.g.
#: ``"0 Computer science, information, and general works"``.
_ROUTERARENA_DOMAIN_BY_DEWEY: dict[str, Domain] = {
    "0": Domain.CS_GENERAL,
    "1": Domain.PHILOSOPHY_PSYCHOLOGY,
    "3": Domain.SOCIAL_SCIENCE,
    "4": Domain.LANGUAGE,
    "5": Domain.SCIENCE,
    "6": Domain.TECHNOLOGY,
    "7": Domain.ARTS_RECREATION,
    "8": Domain.LITERATURE,
    "9": Domain.HISTORY,
}


def domain_from_routerarena(raw: str) -> Domain | None:
    """Map a raw RouterArena ``Domain`` string onto :class:`Domain`.

    Returns ``None`` for Dewey class 2 (Religion) and anything unrecognised, so
    callers can drop rather than silently mislabel.
    """
    if not raw:
        return None
    match = re.match(r"\s*(\d)", raw)
    if match is None:
        return None
    return _ROUTERARENA_DOMAIN_BY_DEWEY.get(match.group(1))


#: Substring -> TaskType, applied to RouterArena's ``Dataset name`` provenance.
#: Order matters: the first matching rule wins, so put specific before generic.
_TASK_TYPE_RULES: tuple[tuple[str, TaskType], ...] = (
    ("livecodebench", TaskType.CODE),
    ("humaneval", TaskType.CODE),
    ("mbpp", TaskType.CODE),
    ("mmlupro_computer science", TaskType.CODE),
    ("wmt", TaskType.TRANSLATION),
    ("aime", TaskType.MATH),
    ("math", TaskType.MATH),
    ("gsm8k", TaskType.MATH),
    ("asdiv", TaskType.MATH),
    ("finqa", TaskType.MATH),
    ("narrativeqa", TaskType.READING_COMPREHENSION),
    ("superglue-rc", TaskType.READING_COMPREHENSION),
    ("superglue-clozetest", TaskType.READING_COMPREHENSION),
    ("superglue-wic", TaskType.READING_COMPREHENSION),
    ("pubmedqa", TaskType.READING_COMPREHENSION),
    ("ethics", TaskType.REASONING),
    ("formal_logic", TaskType.REASONING),
    ("chessinstruct", TaskType.REASONING),
    ("socialiqa", TaskType.REASONING),
    ("superglue-causalreasoning", TaskType.REASONING),
    ("superglue-entailment", TaskType.REASONING),
    ("superglue-wsc", TaskType.REASONING),
    ("qanta", TaskType.FACTUAL_QA),
    ("opentdb", TaskType.FACTUAL_QA),
    ("mmlu", TaskType.FACTUAL_QA),
    ("medmcqa", TaskType.FACTUAL_QA),
    ("geobench", TaskType.FACTUAL_QA),
    ("geographydata", TaskType.FACTUAL_QA),
    ("musictheorybench", TaskType.FACTUAL_QA),
)


def task_type_from_dataset_name(raw: str) -> TaskType:
    """Infer the capability axis from RouterArena dataset provenance."""
    name = (raw or "").strip().lower()
    for needle, task in _TASK_TYPE_RULES:
        if needle in name:
            return task
    return TaskType.OTHER


def task_type_from_arena_tags(
    *, is_code: bool, is_math: bool, is_creative_writing: bool
) -> TaskType:
    """Collapse LMArena's binary ``category_tag`` flags onto :class:`TaskType`.

    The flags are not mutually exclusive; precedence follows how strongly each
    flag constrains model choice (code and math dominate a routing decision).
    """
    if is_code:
        return TaskType.CODE
    if is_math:
        return TaskType.MATH
    if is_creative_writing:
        return TaskType.CREATIVE_WRITING
    return TaskType.OTHER
