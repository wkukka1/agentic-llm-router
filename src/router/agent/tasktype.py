"""Task-type inference: the capability axis of the signal triple.

**Placeholder.** This is keyword matching, not a model. The trained version is a
classifier on LMArena's ``is_code`` / ``math_v0.1`` / ``creative_writing_v0.1``
flags -- ~140k weakly-labelled rows already reachable through
:mod:`router.data.sources.arena`. Until that head is trained, the rules below
cover the high-precision cases and default to ``other``.

It is behind the same ``str -> str`` interface the trained head will expose, so
swapping it in is a one-line change in the pipeline construction.
"""

from __future__ import annotations

import re

from router.data.taxonomy import TaskType

_CODE = re.compile(
    r"```|\b(refactor|debug|compile|stack ?trace|regex|API|SDK|function|class|"
    r"variable|repository|typescript|javascript|python|rust|golang|sql|bash)\b",
    re.IGNORECASE,
)
_MATH = re.compile(
    r"\$\$|\\frac|\\int|\b(prove|proof|theorem|derivative|integral|equation|"
    r"solve for|factorial|probability|matrix|eigenvalue)\b",
    re.IGNORECASE,
)
_TRANSLATION = re.compile(r"\btranslate\b|\binto (english|french|spanish|german|chinese|japanese)\b", re.IGNORECASE)
_CREATIVE = re.compile(
    r"\b(write|compose|draft) (me )?(a |an )?(poem|story|song|lyric|novel|script|screenplay)\b",
    re.IGNORECASE,
)
_READING = re.compile(
    r"\b(summari[sz]e|according to the (passage|text|document)|based on the (passage|text|article))\b",
    re.IGNORECASE,
)
_REASONING = re.compile(
    r"\b(why|explain why|reason|implication|trade-?off|compare|critique|argue|justify)\b",
    re.IGNORECASE,
)
_FACTUAL = re.compile(r"^\s*(who|what|when|where|which)\b", re.IGNORECASE)


def infer_task_type(prompt: str) -> str:
    """First matching rule wins; order is by how strongly it constrains routing."""
    text = prompt or ""
    for pattern, task in (
        (_CODE, TaskType.CODE),
        (_MATH, TaskType.MATH),
        (_TRANSLATION, TaskType.TRANSLATION),
        (_CREATIVE, TaskType.CREATIVE_WRITING),
        (_READING, TaskType.READING_COMPREHENSION),
        (_FACTUAL, TaskType.FACTUAL_QA),
        (_REASONING, TaskType.REASONING),
    ):
        if pattern.search(text):
            return task.value
    return TaskType.OTHER.value
