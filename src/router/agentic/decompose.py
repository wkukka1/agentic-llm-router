"""Turn a compound request into sub-tasks, and stitch sub-answers back together.

The orchestrator needs two functions:

* a **decomposer** ``str -> list[str]`` -- split the request into independently
  answerable sub-prompts (return ``[prompt]`` to decline).
* a **synthesizer** ``(original, [(sub, answer)]) -> str`` -- compose the final
  answer.

:class:`NaiveDecomposer` / :func:`concat_synthesizer` are dependency-free
defaults (regex splitting, labelled concatenation). :class:`LLMDecomposer` /
:class:`LLMSynthesizer` use a LangChain chat model for real use.
"""

from __future__ import annotations

import re
import warnings
from typing import Callable, Sequence

from .llm_clients import message_text

__all__ = [
    "NaiveDecomposer",
    "LLMDecomposer",
    "concat_synthesizer",
    "LLMSynthesizer",
]

# common abbreviations whose trailing "." is not a sentence end
_NO_SPLIT_BEFORE = r"(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\betc)(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bU\.S)"

# split on: list markers, a "?"/"." boundary before a capital, or a linking phrase.
# The capital lookahead is scoped case-sensitive -- under IGNORECASE `[A-Z]`
# would match any letter and split mid-sentence.
_SPLIT_RE = re.compile(
    r"\s*(?:"
    r"\n[ \t]*[-*•]+[ \t]+"        # bullet list item
    r"|\n[ \t]*\d+[.)][ \t]+"            # numbered list item
    rf"|{_NO_SPLIT_BEFORE}[?.]\s+(?-i:(?=[A-Z]))"   # sentence boundary before a capital
    r"|\band then\b|\bafter that\b|\bthen,\b|\bfollowed by\b|\bas well as\b"
    r"|;\s+then\b"
    # ", then" -- the comma form. Present in neither this list nor the triage
    # marker list, so "Summarise X, then implement Y, then analyse Z" was
    # triaged as one task and, once triage was fixed, still came back as one
    # part with "nothing to decompose".
    r"|,\s*then\b"
    r")\s*",
    re.IGNORECASE,
)

# "1. ", "2) ", "- ", "* " line prefixes an LLM adds despite being told not to
_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)]|#+)\s*")


def _cap(parts: list[str], max_parts: int, who: str) -> list[str]:
    """Apply the fan-out cap, and say so when it bites.

    The cap is a real budget guard -- a 50-way decomposition is 50 paid calls --
    but dropping the tail silently means the caller asks for twelve things, is
    charged for eight, and is handed a synthesized answer that reads as though
    it covered everything. Whoever set `max_parts` should hear that it was the
    binding constraint, not discover it in a bill.
    """
    if len(parts) <= max_parts:
        return parts
    warnings.warn(
        f"{who}: decomposed into {len(parts)} sub-tasks but max_parts={max_parts}; "
        f"dropping the last {len(parts) - max_parts} "
        f"({[p[:40] for p in parts[max_parts:]]}). Raise max_parts to keep them.",
        stacklevel=3,
    )
    return parts[:max_parts]


class NaiveDecomposer:
    """Regex split on list markers / "and then" / question boundaries. No LLM."""

    def __init__(self, *, max_parts: int = 6, min_part_words: int = 2):
        self.max_parts = max_parts
        # Words, not characters. The guard exists to stop punctuation residue
        # becoming a paid model call, and a character count answers a different
        # question: "plot it" is 7 characters and an entirely real task, while
        # "accordingly" is 11 and is not one.
        self.min_part_words = min_part_words

    def _coalesce(self, raw: list[str]) -> list[str]:
        """Fold fragments into their neighbours rather than discarding them.

        Dropping anything here loses a sub-task silently: "Do these: 1. read
        the file 2. compute the mean 3. plot it" used to come back as three
        parts, one of which was the lead-in "Do these:" -- billed as its own
        model call -- while "plot it" vanished for being below a character
        threshold. The caller asked for three things, paid for three calls, and
        got two of them answered, with no error raised anywhere.

        So: a clause ending in ``:`` introduces the next one and is prepended
        to it, and anything with too few words to be a request of its own joins
        the part before it. Nothing is thrown away.
        """
        out: list[str] = []
        lead_in = ""
        for part in raw:
            if lead_in:
                part, lead_in = f"{lead_in} {part}".strip(), ""
            if part.endswith(":"):        # a lead-in, not a task
                lead_in = part
            elif len(part.split()) < self.min_part_words and out:
                out[-1] = f"{out[-1]} {part}"
            else:
                out.append(part)
        if lead_in:                       # trailing colon with nothing after it
            if out:
                out[-1] = f"{out[-1]} {lead_in}"
            else:
                out.append(lead_in)
        return out

    def __call__(self, prompt: str) -> list[str]:
        raw = [p.strip(" \t\n-*.") for p in _SPLIT_RE.split(prompt.strip()) if p and p.strip()]
        parts = self._coalesce(raw)
        # de-dupe while keeping order
        seen, out = set(), []
        for p in parts:
            if p.lower() not in seen:
                seen.add(p.lower())
                out.append(p if p.endswith(("?", ".", "!")) else p + ".")
        if len(out) <= 1:
            return [prompt]
        return _cap(out, self.max_parts, type(self).__name__)


class LLMDecomposer:
    """Ask a LangChain chat model for a newline-separated task list.

    A request longer than ``max_prompt_chars`` goes to ``fallback`` instead of
    being truncated (a cut-off request silently loses its later sub-tasks)."""

    _PROMPT = (
        "Break the request below into a minimal ordered list of self-contained "
        "sub-tasks, one per line, no numbering or commentary. If it is already a "
        "single task, return it unchanged on one line.\n\nRequest:\n{prompt}"
    )

    def __init__(self, chat_model, *, max_parts: int = 8, max_prompt_chars: int = 4000,
                 fallback: Callable[[str], list[str]] | None = None):
        self._chat = chat_model
        self.max_parts = max_parts
        self.max_prompt_chars = int(max_prompt_chars)
        self._fallback = fallback or NaiveDecomposer()

    def __call__(self, prompt: str) -> list[str]:
        if len(prompt) > self.max_prompt_chars:
            warnings.warn(
                f"LLMDecomposer: prompt is {len(prompt)} chars (> {self.max_prompt_chars}); "
                "using the fallback decomposer instead of truncating",
                stacklevel=2,
            )
            return self._fallback(prompt)
        try:
            msg = self._chat.invoke(self._PROMPT.format(prompt=prompt))
        except Exception as exc:  # pragma: no cover - provider errors
            warnings.warn(f"LLMDecomposer call failed ({exc!r}); using the fallback decomposer",
                          stacklevel=2)
            return self._fallback(prompt)
        return self._parse(message_text(msg)) or [prompt]

    def _parse(self, text: str) -> list[str]:
        parts = []
        for ln in text.splitlines():
            ln = _LINE_PREFIX_RE.sub("", ln).strip()
            # a preamble such as "Here are the sub-tasks:" is not a task
            if len(ln) < 4 or ln.endswith(":"):
                continue
            parts.append(ln)
        return _cap(parts, self.max_parts, type(self).__name__)


def concat_synthesizer(original: str, pairs: Sequence[tuple[str, str]]) -> str:
    """Labelled concatenation -- the dependency-free default."""
    if len(pairs) == 1:
        return pairs[0][1]
    lines = [f"Answer to: {original.strip()}", ""]
    for i, (sub, ans) in enumerate(pairs, 1):
        lines.append(f"{i}. {sub.strip()}")
        lines.append(f"   {ans.strip()}")
        lines.append("")
    return "\n".join(lines).rstrip()


class LLMSynthesizer:
    """Fold the sub-answers into one coherent response with a chat model.

    If the sub-answers exceed ``max_body_chars`` they go to ``fallback``
    (concatenation) rather than being truncated -- a cut body drops the later
    sub-answers from the final answer."""

    _PROMPT = (
        "Original request:\n{original}\n\n"
        "Sub-answers:\n{body}\n\n"
        "Write a single, coherent final answer to the original request using the "
        "sub-answers above. Do not mention that the work was split up."
    )

    def __init__(self, chat_model, *, max_body_chars: int = 8000,
                 fallback: Callable[..., str] = concat_synthesizer):
        self._chat = chat_model
        self.max_body_chars = int(max_body_chars)
        self._fallback = fallback

    def __call__(self, original: str, pairs: Sequence[tuple[str, str]]) -> str:
        body = "\n\n".join(f"[{i}] {s}\n{a}" for i, (s, a) in enumerate(pairs, 1))
        if len(body) > self.max_body_chars:
            warnings.warn(
                f"LLMSynthesizer: sub-answers are {len(body)} chars (> {self.max_body_chars}); "
                "concatenating instead of truncating",
                stacklevel=2,
            )
            return self._fallback(original, pairs)
        try:
            msg = self._chat.invoke(self._PROMPT.format(original=original[:2000], body=body))
        except Exception as exc:  # pragma: no cover - provider errors
            warnings.warn(f"LLMSynthesizer call failed ({exc!r}); concatenating", stacklevel=2)
            return self._fallback(original, pairs)
        return message_text(msg).strip()
