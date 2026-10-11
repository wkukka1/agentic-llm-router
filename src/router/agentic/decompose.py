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
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .llm_clients import message_text

__all__ = [
    "NaiveDecomposer",
    "LLMDecomposer",
    "concat_synthesizer",
    "LLMSynthesizer",
    "looks_atomic",
    "SubTask",
    "parse_plan_lines",
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

# "1. ", "2) ", "- ", "* " line prefixes an LLM adds despite being told not to. A numbered
# marker needs trailing whitespace, else it would eat the start of "3.14 times 2".
_LINE_PREFIX_RE = re.compile(r"^\s*(?:(?:[-*•]+|#+)\s*|\d+[.)]\s+)")


# a sentence end followed by more text: "Do X. Then Y" (not a trailing "?" / ".")
_INNER_SENTENCE_END_RE = re.compile(r"[.?!]\s+\S")


def looks_atomic(prompt: str) -> bool:
    """One sentence with no comma, semicolon or line break -- nothing an LLM
    decomposer could split. Deliberately broader than triage's lexical
    multi-part markers, which miss comma lists ("explain why, fix it, and
    test it")."""
    text = prompt.strip()
    return not (_INNER_SENTENCE_END_RE.search(text) or any(c in text for c in ",;\n"))


def _dedupe_key(task: str) -> str:
    return " ".join(task.lower().split()).rstrip(".?! ")


def _fold_tail(parts: list[str], max_parts: int) -> list[str]:
    """Cap ``parts`` at ``max_parts`` by folding the overflow into the last kept
    part -- truncating would silently drop the later sub-tasks."""
    if len(parts) <= max_parts:
        return parts
    keep = max(max_parts, 1)
    return parts[: keep - 1] + [" ".join(parts[keep - 1:])]


# "[after 2]" / "[after 1, 3]" / "[after 1 and 2]" at the start of a planner line
_AFTER_RE = re.compile(r"^\[after\s+([^\]]*)\]\s*", re.IGNORECASE)
# "[150-300 words, max 450]" at the start of a planner line
_BUDGET_RE = re.compile(
    r"^\[\s*(\d+)\s*[-–]\s*(\d+)\s*w(?:ords?)?\s*,\s*max\s*(\d+)\s*(?:w(?:ords?)?)?\s*\]\s*",
    re.IGNORECASE)
# a planner line's own number: "3. ..." / "3) ..."
_LINE_NUMBER_RE = re.compile(r"^\s*(\d+)[.)]\s+")
_NONE_LINES = {"none", "n/a", "na", "-"}

#: ``(low, high, limit)`` words: aim for ``low..high``, never exceed ``limit``
Budget = tuple[int, int, int]
#: the budget of a step the planner gave none
DEFAULT_BUDGET: Budget = (150, 400, 600)
MIN_WORDS, MAX_WORDS = 20, 2000
TOKENS_PER_WORD = 1.35
#: the hard cut-off sits this far above the word limit: it stops a runaway
#: model, it must never chop a normal answer
SAFETY_FACTOR = 2.0
#: the final answer's share of the steps' combined budget
FINAL_SHARE = 0.6


def _clamp_budget(values) -> Budget:
    low, high, limit = sorted(min(max(int(v), MIN_WORDS), MAX_WORDS) for v in values)
    return low, high, limit


def budget_instruction(budget: Budget) -> str:
    low, high, limit = budget
    return f"Length: aim for about {low}-{high} words; never exceed {limit} words."


def budget_max_tokens(budget: Budget) -> int:
    """The safety cut-off (``max_tokens``) for a call with this budget."""
    return int(round(budget[2] * TOKENS_PER_WORD * SAFETY_FACTOR))


def budget_mid_tokens(budget: Budget) -> int:
    """Expected output tokens: the middle of the safe range (for forecasting)."""
    return int(round((budget[0] + budget[1]) / 2 * TOKENS_PER_WORD))


def derive_final_budget(budgets) -> Optional[Budget]:
    """The final answer's budget: ``FINAL_SHARE`` of the steps' combined
    budget, capped at ``MAX_WORDS`` (``None`` without step budgets)."""
    budgets = [b for b in budgets if b]
    if not budgets:
        return None
    return _clamp_budget(round(FINAL_SHARE * sum(b[i] for b in budgets)) for i in range(3))


def _sum_budgets(budgets) -> Optional[Budget]:
    budgets = [b for b in budgets if b]
    return _clamp_budget(sum(b[i] for b in budgets) for i in range(3)) if budgets else None


@dataclass(frozen=True)
class SubTask:
    """One planned sub-task: self-contained ``text``, the 1-based positions of
    the earlier sub-tasks whose results it needs (``depends_on``), and the
    planner's length ``budget`` (``None`` when it gave none)."""

    text: str
    depends_on: tuple[int, ...] = ()
    budget: Optional[Budget] = None


def parse_plan_lines(text: str, *, max_parts: int = 8,
                     ref_limit: Optional[int] = None) -> list[SubTask]:
    """Planner lines -> :class:`SubTask`\\ s.

    A line may start with ``[after N]`` (or ``[after 1, 3]``) naming the
    earlier lines whose results it needs, and/or -- in either order -- a
    length budget ``[A-B words, max C]`` (sorted into order and clamped to
    ``MIN_WORDS..MAX_WORDS``). ``N`` is the number the model gave that line
    (``"2. ..."``), else its position among the kept lines. When any line is
    numbered, unnumbered lines are commentary and are dropped.
    Malformed references never invalidate the plan: a reference to an unknown,
    later or the same line is dropped, and one to a dropped repeat points at
    the line it repeated.

    ``ref_limit`` switches to *external* references: they name existing steps
    ``1..ref_limit`` (the quality checker's follow-ups refer to the steps it
    reviewed), so only that range is checked."""
    kept: list[tuple[str, list[int], Optional[Budget]]] = []
    label_to_pos: dict[int, int] = {}
    key_to_pos: dict[str, int] = {}
    lines = text.splitlines()
    # in a numbered plan an unnumbered line is commentary ("Sure, here is the
    # plan", "Note: ..."): as a task it would also take a number the model uses
    numbered_plan = ref_limit is None and any(_LINE_NUMBER_RE.match(ln) for ln in lines)
    for ln in lines:
        m = _LINE_NUMBER_RE.match(ln)
        if numbered_plan and m is None:
            continue
        label = int(m.group(1)) if m else None
        ln = _LINE_PREFIX_RE.sub("", ln).strip()
        refs: list[int] = []
        budget: Optional[Budget] = None
        for _ in range(2):                      # the two tags, in either order
            after, sized = _AFTER_RE.match(ln), _BUDGET_RE.match(ln)
            if after:
                refs = [int(x) for x in re.findall(r"\d+", after.group(1))]
                ln = ln[after.end():].strip()
            elif sized:
                budget = _clamp_budget(sized.groups())
                ln = ln[sized.end():].strip()
        key = _dedupe_key(ln)
        if len(ln) < 4 or ln.endswith(":") or key in _NONE_LINES:
            continue
        if key in key_to_pos:
            if label is not None:
                label_to_pos.setdefault(label, key_to_pos[key])
            continue
        pos = len(kept) + 1
        key_to_pos[key] = pos
        label_to_pos.setdefault(label if label is not None else pos, pos)
        kept.append((ln, refs, budget))

    out = []
    for pos, (ln, refs, budget) in enumerate(kept, 1):
        if ref_limit is not None:
            deps = {r for r in refs if 1 <= r <= ref_limit}
        else:
            deps = {label_to_pos[r] for r in refs if r in label_to_pos and label_to_pos[r] < pos}
        out.append(SubTask(ln, tuple(sorted(deps)), budget))
    return _fold_subtasks(out, max_parts, external=ref_limit is not None)


def _fold_subtasks(tasks: list[SubTask], max_parts: int, *, external: bool = False) -> list[SubTask]:
    """:func:`_fold_tail` for sub-tasks: the overflow folds into the last kept
    one, which keeps the folded tasks' dependencies on earlier lines (all of
    them, for ``external`` references) and the sum of their budgets."""
    if len(tasks) <= max_parts:
        return tasks
    keep = max(max_parts, 1)
    tail = tasks[keep - 1:]
    deps = {d for t in tail for d in t.depends_on if external or d < keep}
    return tasks[: keep - 1] + [SubTask(" ".join(t.text for t in tail), tuple(sorted(deps)),
                                        _sum_budgets(t.budget for t in tail))]


class NaiveDecomposer:
    """Regex split on list markers / "and then" / question boundaries. No LLM.

    No part of the request is dropped: a fragment shorter than ``min_part_chars``
    is merged into its neighbour, and parts beyond ``max_parts`` are folded into
    the last one."""

    def __init__(self, *, max_parts: int = 6, min_part_chars: int = 8):
        self.max_parts = max_parts
        self.min_part_chars = min_part_chars

    def __call__(self, prompt: str) -> list[str]:
        seen, out = set(), []
        lead = ""          # short fragments before the first full part, awaiting a neighbour
        for frag in _SPLIT_RE.split(prompt.strip()):
            p = frag.strip(" \t\n-*.")
            if not p or p.lower() in seen:      # empty, or a repeat of an earlier part
                continue
            seen.add(p.lower())
            text = p if p.endswith(("?", ".", "!")) else p + "."
            if len(p) < self.min_part_chars:
                if out:
                    out[-1] += " " + text
                else:
                    lead = f"{lead} {text}".strip()
            else:
                out.append(f"{lead} {text}".strip())
                lead = ""
        out = _fold_tail(out, self.max_parts)
        return out if len(out) > 1 else [prompt]


class LLMDecomposer:
    """Ask a LangChain chat model for a newline-separated task list.

    ``chat_for`` (instead of ``chat_model``) picks the model per request:
    ``request -> (text -> message)`` -- e.g. :meth:`AgenticRouter.helper_call`,
    which routes the decomposition onto a pool model.

    A request longer than ``max_prompt_chars`` goes to ``fallback`` instead of
    being truncated (a cut-off request silently loses its later sub-tasks)."""

    def __init__(self, chat_model=None, *, max_parts: int = 8, max_prompt_chars: int = 4000,
                 fallback: Callable[[str], list[str]] | None = None,
                 prompt: str | None = None,
                 chat_for: Callable[[str], Callable[[str], object]] | None = None,
                 plan_prompt: str | None = None):
        from .schemas import default_prompts

        if (chat_model is None) == (chat_for is None):
            raise ValueError("LLMDecomposer needs exactly one of chat_model / chat_for")
        self._chat = chat_model
        self._chat_for = chat_for
        #: ``str.format`` template; the ``decompose`` prompt in ``configs/agentic.yaml`` unless given
        self.prompt = prompt or default_prompts()["decompose"]
        #: ``str.format`` template for :meth:`plan` (numbered sub-tasks with ``[after N]``)
        self.plan_prompt = plan_prompt or default_prompts()["plan"]
        self.max_parts = max_parts
        self.max_prompt_chars = int(max_prompt_chars)
        self._fallback = fallback or NaiveDecomposer()
        # set by AgenticRouter.__init__ so this call counts against the run's
        # shared ``max_calls`` budget like a routed call does (AGENTICROUTER-003)
        self.on_call: Callable[[], None] | None = None

    def _ask(self, template: str, prompt: str) -> str | None:
        """The model's reply to ``template`` for ``prompt``, or ``None`` when
        the request is too long or the call failed (the caller falls back)."""
        if len(prompt) > self.max_prompt_chars:
            warnings.warn(
                f"LLMDecomposer: prompt is {len(prompt)} chars (> {self.max_prompt_chars}); "
                "using the fallback decomposer instead of truncating",
                stacklevel=3,
            )
            return None
        if self.on_call is not None:
            self.on_call()
        try:
            call = self._chat_for(prompt) if self._chat_for is not None else self._chat.invoke
            return message_text(call(template.format(prompt=prompt)))
        except Exception as exc:  # pragma: no cover - provider errors
            warnings.warn(f"LLMDecomposer call failed ({exc!r}); using the fallback decomposer",
                          stacklevel=3)
            return None

    def __call__(self, prompt: str) -> list[str]:
        text = self._ask(self.prompt, prompt)
        if text is None:
            return self._fallback(prompt)
        return self._parse(text) or [prompt]

    def plan(self, prompt: str) -> list[SubTask]:
        """:meth:`__call__` with dependencies: the ``plan`` prompt, parsed by
        :func:`parse_plan_lines`. Falls back like :meth:`__call__`, giving
        independent sub-tasks."""
        text = self._ask(self.plan_prompt, prompt)
        if text is None:
            return [SubTask(s) for s in self._fallback(prompt)] or [SubTask(prompt)]
        return parse_plan_lines(text, max_parts=self.max_parts) or [SubTask(prompt)]

    def _parse(self, text: str) -> list[str]:
        parts, seen = [], set()
        for ln in text.splitlines():
            ln = _LINE_PREFIX_RE.sub("", ln).strip()
            # a preamble such as "Here are the sub-tasks:" is not a task
            if len(ln) < 4 or ln.endswith(":"):
                continue
            # a repeated task (case / spacing / end punctuation aside) is dropped
            # before the max_parts fold, so repeats can't crowd out real tasks
            key = _dedupe_key(ln)
            if key in seen:
                continue
            seen.add(key)
            parts.append(ln)
        return _fold_tail(parts, self.max_parts)


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

    def __init__(self, chat_model=None, *, max_body_chars: int = 8000,
                 fallback: Callable[..., str] = concat_synthesizer,
                 prompt: str | None = None,
                 chat_for: Callable[[str], Callable[[str], object]] | None = None):
        from .schemas import default_prompts

        if (chat_model is None) == (chat_for is None):
            raise ValueError("LLMSynthesizer needs exactly one of chat_model / chat_for")
        self._chat = chat_model
        #: ``original request -> (text -> message)``, as on :class:`LLMDecomposer`
        self._chat_for = chat_for
        #: ``str.format`` template; the ``synthesize`` prompt in ``configs/agentic.yaml`` unless given
        self.prompt = prompt or default_prompts()["synthesize"]
        self.max_body_chars = int(max_body_chars)
        self._fallback = fallback
        # set by AgenticRouter.__init__ so this call counts against the run's
        # shared ``max_calls`` budget like a routed call does (AGENTICROUTER-003)
        self.on_call: Callable[[], None] | None = None

    def __call__(self, original: str, pairs: Sequence[tuple[str, str]], *,
                 budget: Optional[Budget] = None) -> str:
        """``budget`` (``(low, high, limit)`` words) adds a length line to the
        request and a generous ``max_tokens`` safety cut-off to the call."""
        body = "\n\n".join(f"[{i}] {s}\n{a}" for i, (s, a) in enumerate(pairs, 1))
        if len(body) > self.max_body_chars:
            warnings.warn(
                f"LLMSynthesizer: sub-answers are {len(body)} chars (> {self.max_body_chars}); "
                "concatenating instead of truncating",
                stacklevel=2,
            )
            return self._fallback(original, pairs)
        if self.on_call is not None:
            self.on_call()
        try:
            call = self._chat_for(original) if self._chat_for is not None else self._chat.invoke
            request = self.prompt.format(original=original[:2000], body=body)
            if budget is None:
                msg = call(request)
            else:
                msg = call(f"{request}\n\n{budget_instruction(budget)}",
                           max_tokens=budget_max_tokens(budget))
        except Exception as exc:  # pragma: no cover - provider errors
            warnings.warn(f"LLMSynthesizer call failed ({exc!r}); concatenating", stacklevel=2)
            return self._fallback(original, pairs)
        return message_text(msg).strip()
