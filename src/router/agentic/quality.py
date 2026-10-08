"""Final quality check for an orchestrated answer.

An LLM judge reviews the original request, every step's task and answer, and
the synthesized answer, and returns a structured :class:`QualityVerdict`:
pass/fail, a 0-10 score, the steps it found weak, and follow-up tasks for
anything missing or wrong. The plan orchestrator
(:mod:`router.execution.routed`) turns a failed verdict into a repair round.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .decompose import SubTask, parse_plan_lines
from .llm_clients import message_text

__all__ = ["QualityVerdict", "QualityChecker", "parse_verdict"]

_FLAGS = re.IGNORECASE | re.MULTILINE
_VERDICT_RE = re.compile(r"^[ \t]*VERDICT:[ \t]*(PASS|FAIL)\b", _FLAGS)
_SCORE_RE = re.compile(r"^[ \t]*SCORE:[ \t]*(\d+(?:\.\d+)?)", _FLAGS)
_WEAK_RE = re.compile(r"^[ \t]*WEAK_STEPS:[ \t]*(.*)$", _FLAGS)
_ISSUES_RE = re.compile(r"^[ \t]*ISSUES:[ \t]*(.*)$", _FLAGS)
_FOLLOW_RE = re.compile(r"^[ \t]*FOLLOW_UPS:[ \t]*(.*)$", _FLAGS)
_KEY_LINE_RE = re.compile(r"^[ \t]*(VERDICT|SCORE|WEAK_STEPS|FOLLOW_UPS|ISSUES):", re.IGNORECASE)


@dataclass
class QualityVerdict:
    """``status``: ``ok`` (parsed), ``unparsed`` (no VERDICT / SCORE found) or
    ``error`` (the check call failed). Only an ``ok`` verdict can pass, and
    only an ``ok`` failing verdict drives a repair round."""

    passed: bool
    score: float = float("nan")
    weak_steps: list[int] = field(default_factory=list)
    issues: str = ""
    follow_ups: list[SubTask] = field(default_factory=list)
    status: str = "ok"
    raw: str = ""

    def as_dict(self) -> dict:
        return {"passed": self.passed, "score": self.score, "weak_steps": list(self.weak_steps),
                "issues": self.issues, "follow_ups": [t.text for t in self.follow_ups],
                "status": self.status}


def parse_verdict(text: str, *, pass_score: float = 7.0,
                  n_steps: Optional[int] = None) -> QualityVerdict:
    """The checker's reply -> :class:`QualityVerdict`. ``n_steps`` bounds
    ``WEAK_STEPS`` and follow-up ``[after N]`` references to steps that exist.
    A reply without a VERDICT or SCORE line is ``unparsed`` -- never a pass."""
    verdict, score_m = _VERDICT_RE.search(text), _SCORE_RE.search(text)
    if verdict is None or score_m is None:
        return QualityVerdict(False, status="unparsed", raw=text)
    score = min(max(float(score_m.group(1)), 0.0), 10.0)
    weak_m = _WEAK_RE.search(text)
    weak = sorted({int(x) for x in re.findall(r"\d+", weak_m.group(1))}) if weak_m else []
    if n_steps is not None:
        weak = [k for k in weak if 1 <= k <= n_steps]
    else:
        weak = [k for k in weak if k >= 1]
    issues_m = _ISSUES_RE.search(text)
    return QualityVerdict(
        passed=verdict.group(1).upper() == "PASS" and score >= pass_score,
        score=score, weak_steps=weak,
        issues=issues_m.group(1).strip() if issues_m else "",
        follow_ups=_follow_ups(text, n_steps), raw=text,
    )


def _follow_ups(text: str, n_steps: Optional[int]) -> list[SubTask]:
    """The lines after ``FOLLOW_UPS:`` (plus any task on that line) up to the
    next ``KEY:`` line, parsed like planner lines; ``[after N]`` names a
    reviewed step."""
    m = _FOLLOW_RE.search(text)
    if m is None:
        return []
    lines = [m.group(1)]
    for ln in text[m.end():].splitlines()[1:]:
        if _KEY_LINE_RE.match(ln):
            break
        lines.append(ln)
    return parse_plan_lines("\n".join(lines), ref_limit=n_steps if n_steps is not None else 10**6)


def _cut(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


class QualityChecker:
    """LLM judge for a synthesized answer: ``chat_model`` (``.invoke``) or
    ``chat_for`` (original request -> ``text -> message``, e.g. a routed
    helper via :meth:`AgenticRouter.helper_call`).

    Step answers and the final answer are cut to ``max_step_chars`` /
    ``max_answer_chars`` with a visible ``[truncated]`` marker, so the request
    fits a small judge's context; the prompt tells the judge so."""

    def __init__(self, chat_model=None, *,
                 chat_for: Callable[[str], Callable[[str], object]] | None = None,
                 prompt: Optional[str] = None, pass_score: float = 7.0,
                 max_step_chars: int = 1500, max_answer_chars: int = 6000):
        from .schemas import default_prompts

        if (chat_model is None) == (chat_for is None):
            raise ValueError("QualityChecker needs exactly one of chat_model / chat_for")
        self._chat, self._chat_for = chat_model, chat_for
        self.prompt = prompt or default_prompts()["verify"]
        self.pass_score = float(pass_score)
        self.max_step_chars, self.max_answer_chars = int(max_step_chars), int(max_answer_chars)
        #: set by the plan orchestrator so the check counts against ``max_calls``
        self.on_call: Optional[Callable[[], None]] = None

    def __call__(self, original: str, steps: Sequence[tuple[str, str]], answer: str) -> QualityVerdict:
        body = "\n\n".join(f"Step {i}: {task}\nAnswer: {_cut(ans, self.max_step_chars)}"
                           for i, (task, ans) in enumerate(steps, 1))
        request = self.prompt.format(original=_cut(original, 2000), steps=body,
                                     answer=_cut(answer, self.max_answer_chars))
        if self.on_call is not None:
            # outside the try: an exhausted call budget must not read as a failed check
            self.on_call()
        try:
            call = self._chat_for(original) if self._chat_for is not None else self._chat.invoke
            text = message_text(call(request))
        except Exception as exc:
            warnings.warn(f"QualityChecker call failed ({exc!r}); answer left unverified",
                          stacklevel=2)
            return QualityVerdict(False, issues=f"check failed: {exc!r}", status="error")
        return parse_verdict(text, pass_score=self.pass_score, n_steps=len(steps))
