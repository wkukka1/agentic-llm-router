"""Label space for the router's first stage.

**This replaces a Dewey-decimal topic taxonomy that failed.** That version
classified prompts into 9 library categories (medicine filed under
"technology", mathematics under "science"). It reached 91% on benchmark
questions and **47% on real user prompts** -- it had learned to recognise exam
formatting, and its best class (`language`, 0.99 F1) turned out to be the WMT
translation subset being trivially identifiable by non-English text. On real
traffic that class became a dumping ground, predicted 4x more often than it
occurred.

Three lessons are baked into the design here:

1. **Classify by capability, not subject.** What matters for routing is what the
   model must *do* -- run code, calculate, compose prose -- not which shelf a
   librarian would file it on. "Explain recursion" and "explain the French
   Revolution" are the same routing decision; "write a sorting function" and
   "write a sonnet" are not.
2. **Every label must be assignable from real traffic.** Each class below is
   grounded in an annotation that exists on real prompts, not only on
   benchmarks. A label no data source can supply is a label the model will
   invent.
3. **`OTHER` is a first-class citizen.** Roughly half of real prompts are
   greetings, meta-questions, personal advice or requests no benchmark
   taxonomy anticipated. Forcing them into a topic is what produced the
   dumping-ground failure.
"""

from __future__ import annotations

import re
from enum import StrEnum


class Capability(StrEnum):
    """What the answering model has to be good at."""

    #: Write, debug, explain or transform code; IT and devops questions.
    CODE = "code"
    #: Calculation, proof, symbolic manipulation, quantitative reasoning.
    MATH = "math"
    #: Produce prose as the artifact: stories, poems, lyrics, jokes, scripts.
    CREATIVE_WRITING = "creative_writing"
    #: Render text between languages.
    TRANSLATION = "translation"
    #: Everything else. Deliberately broad -- see the module docstring.
    OTHER = "other"


CAPABILITY_LABELS: list[str] = [c.value for c in Capability]


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


DIFFICULTY_LABELS: list[str] = [d.value for d in Difficulty]


def capability_from_arena_flags(
    *, is_code: bool, is_math: bool, is_creative_writing: bool
) -> Capability:
    """Map LMArena's ``category_tag`` flags onto :class:`Capability`.

    These are annotations on **real user prompts**, which is what makes them
    valuable: they are the only capability labels available on in-the-wild
    traffic. The flags are not mutually exclusive, so precedence follows how
    strongly each constrains model choice -- code and maths dominate a routing
    decision, and a prompt flagged both is overwhelmingly a coding task with
    arithmetic in it rather than the reverse.

    A prompt with no flag is genuinely ``OTHER``: the absence is a real
    negative, not missing data.
    """
    if is_code:
        return Capability.CODE
    if is_math:
        return Capability.MATH
    if is_creative_writing:
        return Capability.CREATIVE_WRITING
    return Capability.OTHER


#: Substring -> Capability, matched against RouterArena's ``Dataset name``.
#: Provenance is a far more reliable label than the ``Domain`` column: a row
#: from LiveCodeBench is a coding task by construction, whereas its Dewey class
#: is a cataloguing decision. Order matters; first match wins.
_ROUTERARENA_RULES: tuple[tuple[str, Capability], ...] = (
    ("livecodebench", Capability.CODE),
    ("humaneval", Capability.CODE),
    ("mbpp", Capability.CODE),
    ("mmlupro_computer science", Capability.CODE),
    ("wmt", Capability.TRANSLATION),
    ("aime", Capability.MATH),
    ("gsm8k", Capability.MATH),
    ("asdiv", Capability.MATH),
    ("mathqa", Capability.MATH),
    ("finqa", Capability.MATH),
    ("mmlupro_math", Capability.MATH),
    ("opentdb_science: mathematics", Capability.MATH),
    ("mmlu_formal_logic", Capability.MATH),
    ("math", Capability.MATH),
)


def capability_from_dataset_name(raw: str) -> Capability:
    """Infer capability from RouterArena dataset provenance.

    Anything unmatched is ``OTHER``, which is correct rather than lazy:
    RouterArena is overwhelmingly multiple-choice knowledge recall, and that is
    exactly what ``OTHER`` denotes here.
    """
    name = (raw or "").strip().lower()
    for needle, capability in _ROUTERARENA_RULES:
        if needle in name:
            return capability
    return Capability.OTHER


#: Cheap surface cues, used only to sanity-check the learned model and to label
#: ad-hoc examples -- never as training supervision.
_CODE_HINT = re.compile(
    r"```|\b(def|class|import|function|SELECT|npm|pip|git|docker|API|regex)\b", re.IGNORECASE
)
_MATH_HINT = re.compile(
    r"\$\$|\\frac|\\int|\b(prove|integral|derivative|equation|solve for|theorem)\b", re.IGNORECASE
)
_CREATIVE_HINT = re.compile(
    r"\b(write|compose|draft) (me )?(a |an )?(poem|story|song|lyric|haiku|novel|script|joke)\b",
    re.IGNORECASE,
)
_TRANSLATION_HINT = re.compile(r"\btranslate\b", re.IGNORECASE)


def capability_hint(prompt: str) -> Capability:
    """Rule-based guess, for diagnostics and as a zero-training baseline."""
    for pattern, capability in (
        (_TRANSLATION_HINT, Capability.TRANSLATION),
        (_CODE_HINT, Capability.CODE),
        (_MATH_HINT, Capability.MATH),
        (_CREATIVE_HINT, Capability.CREATIVE_WRITING),
    ):
        if pattern.search(prompt or ""):
            return capability
    return Capability.OTHER
