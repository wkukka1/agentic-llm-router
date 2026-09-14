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
from typing import Callable, Sequence

__all__ = [
    "NaiveDecomposer",
    "LLMDecomposer",
    "concat_synthesizer",
    "LLMSynthesizer",
]

# split on: list markers, a "?"/"." boundary before a capital, or a linking phrase.
_SPLIT_RE = re.compile(
    r"\s*(?:"
    r"\n[ \t]*[-*•]+[ \t]+"        # bullet list item
    r"|\n[ \t]*\d+[.)][ \t]+"            # numbered list item
    r"|[?.]\s+(?=[A-Z])"                 # sentence boundary before a capital
    r"|\band then\b|\bafter that\b|\bthen,\b|\bfollowed by\b|\bas well as\b"
    r"|;\s+then\b"
    r")\s*",
    re.IGNORECASE,
)


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
    """Ask a LangChain chat model for a newline-separated task list."""

    _PROMPT = (
        "Break the request below into a minimal ordered list of self-contained "
        "sub-tasks, one per line, no numbering or commentary. If it is already a "
        "single task, return it unchanged on one line.\n\nRequest:\n{prompt}"
    )

    def __init__(self, chat_model, *, max_parts: int = 8,
                 fallback: Callable[[str], list[str]] | None = None):
        self._chat = chat_model
        self.max_parts = max_parts
        self._fallback = fallback or NaiveDecomposer()

    def __call__(self, prompt: str) -> list[str]:
        try:
            msg = self._chat.invoke(self._PROMPT.format(prompt=prompt[:4000]))
            text = str(getattr(msg, "content", msg))
            parts = [ln.strip(" \t-*.").strip() for ln in text.splitlines() if ln.strip()]
            parts = [p for p in parts if len(p) >= 4][: self.max_parts]
            return parts or [prompt]
        except Exception:  # pragma: no cover - provider errors
            return self._fallback(prompt)


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
    """Fold the sub-answers into one coherent response with a chat model."""

    _PROMPT = (
        "Original request:\n{original}\n\n"
        "Sub-answers:\n{body}\n\n"
        "Write a single, coherent final answer to the original request using the "
        "sub-answers above. Do not mention that the work was split up."
    )

    def __init__(self, chat_model, *, fallback: Callable[..., str] = concat_synthesizer):
        self._chat = chat_model
        self._fallback = fallback

    def __call__(self, original: str, pairs: Sequence[tuple[str, str]]) -> str:
        try:
            body = "\n\n".join(f"[{i}] {s}\n{a}" for i, (s, a) in enumerate(pairs, 1))
            msg = self._chat.invoke(self._PROMPT.format(original=original[:2000], body=body[:8000]))
            return str(getattr(msg, "content", msg)).strip()
        except Exception:  # pragma: no cover - provider errors
            return self._fallback(original, pairs)
