"""Surface signals: everything readable off a prompt with no model and no labels.

These are free. No encoder pass, no training data, no annotation -- regex and
arithmetic over the prompt string, microseconds each. That matters for a router
whose whole purpose is to spend less: a signal that costs nothing can be
computed on every prompt including the ones that turn out to be trivial.

They are also the signals most likely to survive distribution shift. A
classifier trained on one month of traffic can drift; ``does this contain a code
fence`` cannot.

What they are good for, and what they are not:

* **Good**: routing *preconditions*. Whether a document was attached, whether
  the user asked for JSON, whether there is code to run, whether this is
  non-English -- each of these gates a capability rather than predicting a
  quality, so a cheap detector is the right instrument.
* **Not good**: difficulty. Prompt length alone predicts "both models failed"
  at AUC 0.527, and the full 24-dimensional classifier vector reaches only
  0.577 -- against 0.567 for a 1024-dimensional sentence embedding. Prompt text
  does not carry difficulty; see EXPERIMENTS.md. Do not expect these to supply
  it either.

Every feature is documented in :data:`SURFACE_FEATURES` in the order
:func:`extract` returns them, so the vector and its column names cannot drift
apart.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np

#: Fenced blocks (```), indented blocks, and inline spans. Fenced is the strong
#: signal; inline backticks appear in ordinary prose about software too.
_CODE_FENCE = re.compile(r"```|~~~")
_INLINE_CODE = re.compile(r"`[^`\n]+`")
#: Identifiers that read as code even unfenced: call syntax, dotted paths,
#: snake_case, and the punctuation density of a shell command.
_CODE_SHAPE = re.compile(
    r"\b\w+\([^)]*\)|\b\w+\.\w+\(|\b(?:def|class|import|from|function|const|let|var|"
    r"SELECT|INSERT|UPDATE|DELETE|public|private|#include)\b"
)
_URL = re.compile(r"https?://\S+|www\.\S+")
#: LaTeX delimiters, and bare mathematical operators that survive plain text.
_MATH = re.compile(r"\$\$?|\\\[|\\\(|\\frac|\\sum|\\int|\\sqrt|\^\d|\b\d+\s*[+\-*/=]\s*\d+")
_QUESTION = re.compile(r"\?")
#: An enumerated request: "1." / "2)" at a line start, or bulleted items.
_ENUMERATION = re.compile(r"(?:^|\n)\s*(?:\d{1,2}[.)]|[-*•])\s+", re.MULTILINE)
#: Explicit output-shape requests. These are instructions about the *response*,
#: which is exactly what a cost model wants to know before generating one.
_FORMAT_REQUEST = re.compile(
    r"\b(?:json|yaml|csv|xml|markdown|table|bullet(?:s| points?)?|"
    r"numbered list|code block|one word|single word|only the|no explanation)\b",
    re.IGNORECASE,
)
#: A stated budget on the answer's length.
_LENGTH_CONSTRAINT = re.compile(
    r"\b(?:in|under|at most|no more than|max(?:imum)?|within|exactly)\s+"
    r"\d+\s*(?:words?|characters?|chars?|sentences?|lines?|bullets?|paragraphs?)"
    r"|\b(?:brief|briefly|concise|concisely|short|tl;?dr|in short|one sentence)\b",
    re.IGNORECASE,
)
#: Role-play or system-prompt framing, which tends to precede long outputs.
_PERSONA = re.compile(
    r"\b(?:you are (?:a|an|my)|act as|pretend (?:to be|you)|imagine you|"
    r"as an? experienced|roleplay)\b",
    re.IGNORECASE,
)
#: Requests that cannot be satisfied from parameters alone.
_RECENCY = re.compile(
    r"\b(?:today|yesterday|tomorrow|current(?:ly)?|latest|right now|this (?:week|month|year)|"
    r"as of|202[4-9]|recent(?:ly)?|up to date|news)\b",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")

#: Column names for :func:`extract`, in order. Keep these in lockstep.
SURFACE_FEATURES: tuple[str, ...] = (
    "log_chars",
    "log_words",
    "log_lines",
    "mean_word_len",
    "uppercase_ratio",
    "digit_ratio",
    "punct_ratio",
    "non_ascii_ratio",
    "has_code_fence",
    "has_inline_code",
    "has_code_shape",
    "has_url",
    "n_urls",
    "has_math",
    "n_questions",
    "has_enumeration",
    "n_enumerated_items",
    "asks_format",
    "asks_length_limit",
    "has_persona",
    "needs_recency",
    "ends_with_question",
    "starts_imperative",
)

#: Verbs that start a request rather than a question. Used only for the
#: `starts_imperative` flag, which separates "what is X" from "write me X".
_IMPERATIVE_OPENERS = frozenset("""
write create make generate draft compose build design produce give show list
summarise summarize explain describe translate rewrite revise edit fix correct
improve refine polish convert turn compare rank rate analyse analyze find
search look calculate compute solve implement add remove update generate
""".split())


@dataclass(slots=True)
class SurfaceSignals:
    """One prompt's surface features, named rather than positional."""

    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.values) != len(SURFACE_FEATURES):
            raise ValueError(
                f"expected {len(SURFACE_FEATURES)} features, got {len(self.values)}"
            )

    def as_dict(self) -> dict[str, float]:
        return dict(zip(SURFACE_FEATURES, (float(v) for v in self.values), strict=True))

    def __getitem__(self, name: str) -> float:
        return float(self.values[SURFACE_FEATURES.index(name)])


def _ratio(n: int, d: int) -> float:
    return n / d if d else 0.0


def extract(prompt: str) -> SurfaceSignals:
    """Surface features for one prompt, in :data:`SURFACE_FEATURES` order.

    Counts are log-compressed rather than raw. Prompt length spans three orders
    of magnitude in real traffic, and a linear feature over that range lets a
    handful of very long prompts dominate any model fitted on it.
    """
    text = prompt or ""
    stripped = text.strip()
    words = stripped.split()
    n_chars, n_words = len(stripped), len(words)
    lines = stripped.count("\n") + 1 if stripped else 0

    alpha = [c for c in stripped if c.isalpha()]
    first = words[0].lower().strip(",.!:;") if words else ""
    enumerated = _ENUMERATION.findall(text)

    values = np.array([
        np.log1p(n_chars),
        np.log1p(n_words),
        np.log1p(lines),
        _ratio(sum(len(w) for w in words), n_words),
        _ratio(sum(1 for c in alpha if c.isupper()), len(alpha)),
        _ratio(sum(1 for c in stripped if c.isdigit()), n_chars),
        _ratio(sum(1 for c in stripped if unicodedata.category(c).startswith("P")), n_chars),
        _ratio(sum(1 for c in stripped if ord(c) > 127), n_chars),
        float(bool(_CODE_FENCE.search(text))),
        float(bool(_INLINE_CODE.search(text))),
        float(bool(_CODE_SHAPE.search(text))),
        float(bool(_URL.search(text))),
        np.log1p(len(_URL.findall(text))),
        float(bool(_MATH.search(text))),
        np.log1p(len(_QUESTION.findall(text))),
        float(bool(enumerated)),
        np.log1p(len(enumerated)),
        float(bool(_FORMAT_REQUEST.search(text))),
        float(bool(_LENGTH_CONSTRAINT.search(text))),
        float(bool(_PERSONA.search(text))),
        float(bool(_RECENCY.search(text))),
        float(stripped.endswith("?")),
        float(first in _IMPERATIVE_OPENERS),
    ], dtype=float)
    return SurfaceSignals(values)


def extract_many(prompts: list[str]) -> tuple[np.ndarray, list[str]]:
    """``(n_prompts, n_features)`` matrix plus its column names.

    Names travel with the matrix for the same reason they do in
    :meth:`router.RouterHead.vectorise` -- a feature matrix whose columns
    are documented somewhere else is one refactor from being wrong.
    """
    if not prompts:
        return np.zeros((0, len(SURFACE_FEATURES))), list(SURFACE_FEATURES)
    rows = np.vstack([extract(p).values for p in prompts])
    return rows, list(SURFACE_FEATURES)
