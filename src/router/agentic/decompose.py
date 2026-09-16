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
    r")\s*",
    re.IGNORECASE,
)

# "1. ", "2) ", "- ", "* " line prefixes an LLM adds despite being told not to
_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)]|#+)\s*")


class NaiveDecomposer:
    """Regex split on list markers / "and then" / question boundaries. No LLM."""

    def __init__(self, *, max_parts: int = 6, min_part_chars: int = 8):
        self.max_parts = max_parts
        self.min_part_chars = min_part_chars

    def __call__(self, prompt: str) -> list[str]:
        raw = _SPLIT_RE.split(prompt.strip())
        parts = [p.strip(" \t\n-*.") for p in raw if p and len(p.strip()) >= self.min_part_chars]
        # de-dupe while keeping order
        seen, out = set(), []
        for p in parts:
            if p.lower() not in seen:
                seen.add(p.lower())
                out.append(p if p.endswith(("?", ".", "!")) else p + ".")
        return out[: self.max_parts] if len(out) > 1 else [prompt]


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
        return parts[: self.max_parts]


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
