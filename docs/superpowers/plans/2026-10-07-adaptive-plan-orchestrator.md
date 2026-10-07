# Adaptive Plan Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the plan orchestrator in four ways:
- the planner declares dependencies between steps, and dependent steps get
  the outputs they need;
- each step is routed just before it runs, using that context;
- a step judged weak is escalated to a stronger model;
- a final quality check drives up to N repair rounds, which can add
  follow-up steps.

The best answer is returned and flagged if it never passes.

**Architecture:** Everything builds on `RoutedOrchestrator`
(`src/router/execution/routed.py`) and its inherited loop:
plan → reserve budget → execute → replan while unsuccessful
(`src/router/execution/orchestrator.py:84`).
- The planner only *describes* dependencies (`SubTask.depends_on`).
  Execution owns context and model choice.
- A failed quality check is an ordinary unsuccessful `ExecutionResult`, and
  `replan()` is the repair round.
- New behaviour is opt-in: `route_late`, `checker`.

**Tech Stack:** Python 3.11+, pytest, pydantic v2 settings, existing `AgenticRouter` / `RouterModel` / `ClientRegistry` fakes.

**Spec:** `docs/superpowers/specs/2026-10-07-adaptive-plan-orchestrator-design.md`

## Global Constraints

- **No new dependencies.** Python >= 3.11 (`pyproject.toml`).
- **Defaults stay off.** Code defaults leave today's behaviour unchanged:
  `RoutedOrchestrator(route_late=False, checker=None)`,
  `PlanSettings.route_late = False`, `quality_check.enabled = False`.
  `orchestrator: recursive` stays the default in `configs/agentic.yaml`.
- **One decomposition path.** Every LLM decomposition goes through
  `AgenticRouter.decompose()` or the new `AgenticRouter.decompose_plan()`,
  both gated by `_skip_decomposition`.
- **Context rule.** A failed dependency's `[error: ...]` placeholder is never
  passed to another step as context.
- **No committing.** The user commits; leave changes in the working tree.
  The steps below have no commit commands.
- **New classes.** Before writing any new class under `src/` (`SubTask`,
  `QualityVerdict`, `QualityChecker`), invoke the repo's `gating-new-classes`
  skill.
- **Comments.** Follow the codebase's docstring style; no new comment longer
  than one line.
- **Running tests:**
  - Bash: `.venv/Scripts/python.exe -m pytest <args>`
  - PowerShell: `.\.venv\Scripts\python.exe -m pytest <args>`

## Review Focus

1. **A passing round must win.** It beats an earlier failing round even when
   the failing round scored higher (e.g. FAIL 8 then PASS 7.5 returns the
   PASS answer). Test in Task 7.
2. **An over-budget repair round returns the best answer so far,** never
   `"[error: plan over budget ...]"`. Test in Task 7.
3. **Out-of-range weak-step numbers are ignored** (`0`, `99`): no crash and
   no extra calls. Tests in Tasks 6 and 7.
4. **Malformed planner references still give a valid plan.** Self, forward
   and unknown `[after N]` references, and references to a dropped repeat,
   must never make `Plan.validate()` raise. Tests in Tasks 1 and 3.
5. **Long dependency outputs** must not drop the task from the routing text
   (task first) and must not overflow the checker (answers cut with
   `…[truncated]`). Tests in Tasks 4 and 6.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `src/router/agentic/decompose.py` | Split requests into sub-tasks | + `SubTask`, `parse_plan_lines`, `_fold_subtasks`, `LLMDecomposer.plan` / `_ask` / `plan_prompt` |
| `src/router/agentic/quality.py` | **New.** Final-answer judge | `QualityVerdict`, `parse_verdict`, `QualityChecker` |
| `src/router/agentic/router.py` | Agent | + `decompose_plan`, `_skip_decomposition`; `from_settings` wiring; `_RunState.quality_check` → result |
| `src/router/agentic/result.py` | Result types | + `AgenticResult.quality_check` |
| `src/router/agentic/schemas.py` | Settings | `PlanSettings.route_late`, `.planner_prompt`, `.quality_check` (`QualityCheckSettings`); prompt validation |
| `src/router/execution/routed.py` | Plan orchestrator | Dependencies from the planner, `_contextual_prompt`, late routing, escalation, checker rounds, best answer, repair with follow-ups |
| `configs/agentic.yaml` | Prompts + settings | + `plan`, `verify` prompts; `plan:` section keys |
| `tests/router/agentic/test_plan_parse.py` | **New.** Parser, `LLMDecomposer.plan`, `decompose_plan` | |
| `tests/router/agentic/test_quality.py` | **New.** Verdict parser + checker | |
| `tests/router/execution/test_routed_orchestrator.py` | Orchestrator behaviour | New tests appended |
| `docs/agentic_router.md`, `docs/agentic_router_runbook.md` | Docs | Plan-orchestrator options |

---

## Stage 1 — Planner dependencies and context

### Task 1: `SubTask` and `parse_plan_lines`

**Files:**
- Modify: `src/router/agentic/decompose.py` (imports, `__all__`, new code after `_fold_tail`)
- Test: `tests/router/agentic/test_plan_parse.py` (create)

**Interfaces:**
- Consumes: the existing `_LINE_PREFIX_RE`, `_dedupe_key(task) -> str` and `_fold_tail` in `decompose.py`.
- Produces:
  - `SubTask(text: str, depends_on: tuple[int, ...] = ())`, a frozen dataclass whose dependencies are 1-based positions;
  - `parse_plan_lines(text: str, *, max_parts: int = 8, ref_limit: Optional[int] = None) -> list[SubTask]`.

- [ ] **Step 1: Invoke the `gating-new-classes` skill for `SubTask`** (a frozen dataclass holding one planned sub-task). Proceed as it directs.

- [ ] **Step 2: Write the failing tests** in `tests/router/agentic/test_plan_parse.py`:

```python
"""Planner output with dependencies: parse_plan_lines, LLMDecomposer.plan and
AgenticRouter.decompose_plan (router.agentic.decompose / router)."""

from __future__ import annotations

import pandas as pd
import pytest

from router.agentic import AgenticRouter, ClientRegistry, LLMDecomposer
from router.agentic.decompose import SubTask, parse_plan_lines
from router.routing import RouterModel


def test_after_marker_becomes_a_dependency():
    text = ("1. Implement quicksort in Rust.\n"
            "2. [after 1] Analyse the time complexity of the Rust quicksort from step 1.")
    assert parse_plan_lines(text) == [
        SubTask("Implement quicksort in Rust."),
        SubTask("Analyse the time complexity of the Rust quicksort from step 1.", (1,)),
    ]


def test_unmarked_lines_are_independent():
    assert [t.depends_on for t in parse_plan_lines("Explain A in detail.\nExplain B in detail.")] == [(), ()]


def test_forward_self_and_unknown_references_are_dropped():
    text = ("1. [after 2] Do the first thing.\n"
            "2. [after 2, 9] Do the second thing.\n"
            "3. [after 1 and 2] Do the third thing.")
    assert [t.depends_on for t in parse_plan_lines(text)] == [(), (), (1, 2)]


def test_reference_to_a_dropped_repeat_points_at_the_original():
    text = ("1. Implement quicksort in Rust.\n"
            "2. implement quicksort in rust\n"
            "3. [after 2] Test the Rust quicksort.")
    tasks = parse_plan_lines(text)
    assert [t.text for t in tasks] == ["Implement quicksort in Rust.", "Test the Rust quicksort."]
    assert tasks[1].depends_on == (1,)


def test_none_and_preamble_lines_are_not_tasks():
    assert parse_plan_lines("Here is the plan:\nnone\n1. Do the thing.") == [SubTask("Do the thing.")]


def test_overflow_folds_into_the_last_kept_task_with_its_dependencies():
    text = "1. Task one here.\n2. Task two here.\n3. [after 1] Task three here."
    assert parse_plan_lines(text, max_parts=2) == [
        SubTask("Task one here."),
        SubTask("Task two here. Task three here.", (1,)),
    ]


def test_external_references_are_checked_against_ref_limit_only():
    text = "[after 2] Fix the Rust quicksort from step 2.\n[after 7] Something else to do."
    assert [t.depends_on for t in parse_plan_lines(text, ref_limit=3)] == [(2,), ()]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/router/agentic/test_plan_parse.py -q`
Expected: ERROR / FAIL with `ImportError: cannot import name 'SubTask'`

- [ ] **Step 4: Implement in `src/router/agentic/decompose.py`.**

Change the imports at the top:

```python
import re
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .llm_clients import message_text
```

Add `"SubTask"` and `"parse_plan_lines"` to `__all__`. Directly after the existing `_fold_tail` function, add:

```python
# "[after 2]" / "[after 1, 3]" / "[after 1 and 2]" at the start of a planner line
_AFTER_RE = re.compile(r"^\[after\s+([^\]]*)\]\s*", re.IGNORECASE)
# a planner line's own number: "3. ..." / "3) ..."
_LINE_NUMBER_RE = re.compile(r"^\s*(\d+)[.)]\s+")
_NONE_LINES = {"none", "n/a", "na", "-"}


@dataclass(frozen=True)
class SubTask:
    """One planned sub-task: self-contained ``text`` plus the 1-based positions
    of the earlier sub-tasks whose results it needs (``depends_on``)."""

    text: str
    depends_on: tuple[int, ...] = ()


def parse_plan_lines(text: str, *, max_parts: int = 8,
                     ref_limit: Optional[int] = None) -> list[SubTask]:
    """Planner lines -> :class:`SubTask`\\ s.

    A line may start with ``[after N]`` (or ``[after 1, 3]``) naming the
    earlier lines whose results it needs. ``N`` is the number the model gave
    that line (``"2. ..."``), else its position among the kept lines.
    Malformed references never invalidate the plan: a reference to an unknown,
    later or the same line is dropped, and one to a dropped repeat points at
    the line it repeated.

    ``ref_limit`` switches to *external* references: they name existing steps
    ``1..ref_limit`` (the quality checker's follow-ups refer to the steps it
    reviewed), so only that range is checked."""
    kept: list[tuple[str, list[int]]] = []
    label_to_pos: dict[int, int] = {}
    key_to_pos: dict[str, int] = {}
    for ln in text.splitlines():
        m = _LINE_NUMBER_RE.match(ln)
        label = int(m.group(1)) if m else None
        ln = _LINE_PREFIX_RE.sub("", ln).strip()
        refs: list[int] = []
        after = _AFTER_RE.match(ln)
        if after:
            refs = [int(x) for x in re.findall(r"\d+", after.group(1))]
            ln = ln[after.end():].strip()
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
        kept.append((ln, refs))

    out = []
    for pos, (ln, refs) in enumerate(kept, 1):
        if ref_limit is not None:
            deps = {r for r in refs if 1 <= r <= ref_limit}
        else:
            deps = {label_to_pos[r] for r in refs if r in label_to_pos and label_to_pos[r] < pos}
        out.append(SubTask(ln, tuple(sorted(deps))))
    return _fold_subtasks(out, max_parts, external=ref_limit is not None)


def _fold_subtasks(tasks: list[SubTask], max_parts: int, *, external: bool = False) -> list[SubTask]:
    """:func:`_fold_tail` for sub-tasks: the overflow folds into the last kept
    one, which keeps the folded tasks' dependencies on earlier lines (all of
    them, for ``external`` references)."""
    if len(tasks) <= max_parts:
        return tasks
    keep = max(max_parts, 1)
    tail = tasks[keep - 1:]
    deps = {d for t in tail for d in t.depends_on if external or d < keep}
    return tasks[: keep - 1] + [SubTask(" ".join(t.text for t in tail), tuple(sorted(deps)))]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/agentic/test_plan_parse.py tests/router/agentic/test_agentic_router.py -q`
Expected: all pass (the existing `_parse` dedupe tests still pass).

---

### Task 2: `LLMDecomposer.plan`, the `plan` prompt, `AgenticRouter.decompose_plan`

**Files:**
- Modify: `src/router/agentic/decompose.py` (`LLMDecomposer`)
- Modify: `src/router/agentic/router.py` (`decompose`, new `_skip_decomposition`, new `decompose_plan`, `from_settings`)
- Modify: `src/router/agentic/schemas.py` (`PlanSettings.planner_prompt` + validation)
- Modify: `configs/agentic.yaml` (`plan` prompt; header placeholder list)
- Test: `tests/router/agentic/test_plan_parse.py`

**Interfaces:**
- Consumes: `SubTask`, `parse_plan_lines` (Task 1).
- Produces:
  - `LLMDecomposer(..., plan_prompt: str | None = None)` and `LLMDecomposer.plan(prompt: str) -> list[SubTask]`;
  - `AgenticRouter.decompose_plan(prompt: str, depth: int, *, sequential: bool = False) -> list[SubTask]`;
  - `AgenticRouter._skip_decomposition(prompt: str, depth: int) -> bool`;
  - `PlanSettings.planner_prompt: str = "plan"`.

- [ ] **Step 1: Write the failing tests** (append to `tests/router/agentic/test_plan_parse.py`):

```python
class _Chat:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def invoke(self, text):
        self.prompts.append(text)
        return self.reply


class _WeakRouter(RouterModel):
    """Every prompt scores 0.3, so triage always asks to orchestrate."""

    kind = "weak_text"
    can_route_text = True

    def __init__(self):
        super().__init__(["a", "b"], name="weak")

    def predict_scores(self, query_ids):
        return pd.DataFrame(0.3, index=[str(q) for q in query_ids], columns=self._model_ids)

    def predict_scores_text(self, texts, *, encoder=None):
        return pd.DataFrame(0.3, index=range(len(texts)), columns=self._model_ids)


def _agent(decomposer):
    return AgenticRouter(_WeakRouter(), ClientRegistry.echo(["a", "b"]), decomposer=decomposer)


def test_llm_decomposer_plan_parses_dependencies_with_the_plan_prompt():
    chat = _Chat("1. Implement quicksort in Rust.\n2. [after 1] Test the Rust quicksort from step 1.")
    dec = LLMDecomposer(chat, plan_prompt="PLAN {prompt}")
    assert dec.plan("do it") == [SubTask("Implement quicksort in Rust."),
                                 SubTask("Test the Rust quicksort from step 1.", (1,))]
    assert chat.prompts == ["PLAN do it"]


def test_llm_decomposer_plan_falls_back_on_a_failed_call():
    class Broken:
        def invoke(self, text):
            raise RuntimeError("down")

    dec = LLMDecomposer(Broken(), fallback=lambda p: ["part one here", "part two here"])
    with pytest.warns(UserWarning, match="call failed"):
        assert dec.plan("x") == [SubTask("part one here"), SubTask("part two here")]


def test_decompose_plan_gates_atomic_prompts_like_decompose():
    chat = _Chat("1. A thing to do.\n2. Another thing to do.")
    agent = _agent(LLMDecomposer(chat))
    assert agent.decompose_plan("What is the capital of France?", 0) == [
        SubTask("What is the capital of France?")]
    assert agent.decompose_plan("Explain the cache miss.", 1) == [SubTask("Explain the cache miss.")]
    assert chat.prompts == []


def test_decompose_plan_calls_the_planner_for_a_multipart_prompt():
    chat = _Chat("1. Explain quicksort.\n2. [after 1] Implement the quicksort from step 1 in Rust.")
    tasks = _agent(LLMDecomposer(chat)).decompose_plan(
        "Explain quicksort and then implement it in Rust", 0)
    assert tasks[1].depends_on == (1,) and len(chat.prompts) == 1


def test_decompose_plan_with_a_plain_decomposer_chains_only_when_sequential():
    agent = _agent(lambda p: ["one part", "two part", "three part"])
    assert [t.depends_on for t in agent.decompose_plan("x", 0)] == [(), (), ()]
    assert [t.depends_on for t in agent.decompose_plan("x", 0, sequential=True)] == [(), (1,), (2,)]


def test_shipped_plan_prompt_formats():
    from router.agentic.schemas import default_prompts
    assert "[after N]" in default_prompts()["plan"].format(prompt="p")


def test_planner_prompt_setting_reaches_the_decomposer():
    from router.agentic.schemas import AgenticRouterSettings, load_settings

    base = load_settings()
    settings = AgenticRouterSettings.model_validate({
        **base.model_dump(), "warmup": False, "orchestrator": "plan",
        "prompts": {**base.prompts, "my_plan": "MY PLAN {prompt}"},
        "decomposer": {"kind": "routed"}, "plan": {"planner_prompt": "my_plan"},
    })
    agent = AgenticRouter.from_settings(settings, router=_WeakRouter())
    assert agent.decomposer.plan_prompt == "MY PLAN {prompt}"


def test_unknown_planner_prompt_is_rejected_for_the_plan_orchestrator():
    from router.agentic.schemas import AgenticRouterSettings, load_settings

    with pytest.raises(ValueError, match="planner_prompt"):
        AgenticRouterSettings.model_validate({
            **load_settings().model_dump(), "orchestrator": "plan",
            "plan": {"planner_prompt": "nope"}})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/router/agentic/test_plan_parse.py -q`
Expected: FAIL (`TypeError: ... unexpected keyword argument 'plan_prompt'`, `AttributeError: ... 'decompose_plan'`, `KeyError: 'plan'`)

- [ ] **Step 3: Add the `plan` prompt** to `configs/agentic.yaml`, under `prompts:`, directly after the `decompose:` prompt:

```yaml
  plan: |-
    Break the request below into a minimal ordered list of sub-tasks. Each sub-task is sent to a separate model that will NOT see the original request or the other sub-tasks, so each one must contain exactly the context needed to complete it -- nothing missing, nothing extra:
    - Replace pronouns and references ("it", "this", "the above", "that code") with what they refer to.
    - Carry over every detail from the request that this sub-task needs: its subject, inputs, data, constraints, language, format or audience.
    - Leave out everything this sub-task does not need.
    - Do not answer anything, and do not add requirements the request did not ask for.

    When a sub-task needs the RESULT of an earlier sub-task (for example, analysing code an earlier sub-task writes), start it with [after N], where N is that earlier sub-task's number, and name what it uses ("the Rust implementation from step 1"). That result will be given to it. Use [after N] only for real dependencies.

    Number the sub-tasks 1, 2, 3, ... one per line, and list each sub-task exactly once. Output only the numbered lines: no headings, notes or explanations. If the request is already a single task, return it unchanged as "1. <the request>".

    The example below shows only the format; never reuse its wording or constraints.
    Example request: "Write a haiku about rain, then translate it into Spanish."
    Example sub-tasks:
    1. Write a haiku about rain.
    2. [after 1] Translate the haiku about rain from step 1 into Spanish.

    Request:
    {prompt}
```

In the file's header comment, add these two lines to the placeholder list:

```yaml
#   plan              {prompt}
#   verify            {original} {steps} {answer}
```

- [ ] **Step 4: Implement `LLMDecomposer.plan` in `src/router/agentic/decompose.py`.**
  - Add the `plan_prompt` keyword to `__init__`'s signature (after `chat_for`).
  - Store it after `self.prompt = ...`.
  - Replace the existing `__call__` with `_ask` + `__call__` + `plan`.

```python
    def __init__(self, chat_model=None, *, max_parts: int = 8, max_prompt_chars: int = 4000,
                 fallback: Callable[[str], list[str]] | None = None,
                 prompt: str | None = None,
                 chat_for: Callable[[str], Callable[[str], object]] | None = None,
                 plan_prompt: str | None = None):
```

```python
        #: ``str.format`` template for :meth:`plan` (numbered sub-tasks with ``[after N]``)
        self.plan_prompt = plan_prompt or default_prompts()["plan"]
```

```python
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
            return [SubTask(s) for s in self._fallback(prompt)]
        return parse_plan_lines(text, max_parts=self.max_parts) or [SubTask(prompt)]
```

- [ ] **Step 5: Implement the gate refactor and `decompose_plan`** in
  `src/router/agentic/router.py`. Import `SubTask` from `.decompose`, next to
  `looks_atomic`. Then replace the existing `decompose` method body. Keep its
  docstring unchanged; it already describes the gate.

```python
        if isinstance(self.decomposer, LLMDecomposer) and self._skip_decomposition(prompt, depth):
            return [prompt]
        return self.decomposer(prompt)

    def _skip_decomposition(self, prompt: str, depth: int) -> bool:
        """The gate in :meth:`decompose`: nothing for an LLM decomposer to split."""
        atomic = not self._is_multipart(prompt)
        if depth == 0:
            atomic = atomic and looks_atomic(prompt)
        return atomic

    def decompose_plan(self, prompt: str, depth: int, *, sequential: bool = False) -> list[SubTask]:
        """:meth:`decompose` with dependencies: :class:`SubTask`\\ s whose
        ``depends_on`` name the earlier sub-tasks whose results they need. An
        :class:`LLMDecomposer` plans them (:meth:`LLMDecomposer.plan`) behind
        the same gate as :meth:`decompose`; any other decomposer gives
        independent sub-tasks, each depending on the one before when
        ``sequential``."""
        if isinstance(self.decomposer, LLMDecomposer):
            if self._skip_decomposition(prompt, depth):
                return [SubTask(prompt)]
            return self.decomposer.plan(prompt)
        subs = [s for s in (self.decomposer(prompt) or []) if str(s).strip()] or [prompt]
        return [SubTask(s, (i,) if sequential and i else ()) for i, s in enumerate(subs)]
```

In `from_settings`, pass the planner prompt to the LLM decomposer:

```python
        decomposer = naive if d.kind == "naive" else LLMDecomposer(
            max_parts=d.max_parts, max_prompt_chars=d.max_prompt_chars,
            fallback=naive, prompt=prompts[d.prompt],
            plan_prompt=prompts.get(s.plan.planner_prompt), **helper_kw(d, "decompose"))
```

- [ ] **Step 6: Add the setting and its validation** in
  `src/router/agentic/schemas.py`. In `PlanSettings` add:

```python
    #: the prompt the plan orchestrator's planner uses (numbered steps with [after N])
    planner_prompt: str = "plan"
```

In `AgenticRouterSettings._references_resolve`, before `return self`, add:

```python
        if self.orchestrator == "plan" and self.plan.planner_prompt not in self.prompts:
            raise ValueError(f"plan.planner_prompt: no prompt {self.plan.planner_prompt!r}; "
                             f"known: {sorted(self.prompts)}")
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/agentic -q`
Expected: all pass.

---

### Task 3: The orchestrator plans from dependencies and passes only successful outputs

**Files:**
- Modify: `src/router/execution/routed.py` (`plan`, new `_new_step`, new `_contextual_prompt`, `_run_step`)
- Test: `tests/router/execution/test_routed_orchestrator.py` (append)

**Interfaces:**
- Consumes: `AgenticRouter.decompose_plan` (Task 2).
- Produces:
  - `RoutedOrchestrator._contextual_prompt(step: PlanStep) -> tuple[str, str]`, returning (call prompt, routing text);
  - `RoutedOrchestrator._new_step(step_id, prompt, model_id, scores, dependencies, context_chars) -> PlanStep`;
  - step ids `"<task_id>.<i>"` with `PlanStep.dependencies` in the same form.

- [ ] **Step 1: Write the failing tests** (append to `tests/router/execution/test_routed_orchestrator.py`):

```python
# --------------------------------------------------------------------------- #
# adaptive plan orchestrator                                                  #
# --------------------------------------------------------------------------- #
class _PlanChat:
    """Planner stand-in: replies ``reply`` to every plan request."""

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def invoke(self, text):
        self.calls += 1
        return self.reply


def _planning_agent(reply, *, clients=None, router=None, **kw):
    from router.agentic import LLMDecomposer
    return AgenticRouter(router or TextRouter(), clients or _clients(),
                         decomposer=LLMDecomposer(_PlanChat(reply)), **kw)


DEP_PLAN = ("1. Implement quicksort in Rust.\n"
            "2. [after 1] Analyse the time complexity of the Rust quicksort from step 1.")


def test_planner_dependencies_become_plan_dependencies():
    orch = _orchestrator(_planning_agent(DEP_PLAN))
    orch.task = _task()
    plan = orch.plan(orch.task)
    assert [s.dependencies for s in plan.steps] == [[], ["root.0"]]


def test_malformed_planner_dependencies_still_give_a_valid_plan():
    orch = _orchestrator(_planning_agent(
        "1. [after 2] First task here.\n2. [after 1, 2] Second task here."))
    orch.task = _task()
    plan = orch.plan(orch.task)
    plan.validate()
    assert [s.dependencies for s in plan.steps] == [[], ["root.0"]]


def test_dependent_step_receives_its_dependency_output():
    sent = []
    agent = _planning_agent(DEP_PLAN, clients=_clients(sent=sent))
    assert _orchestrator(agent).run(_task()).success
    assert sent[0][1] == "Implement quicksort in Rust."
    assert "big answers" in sent[1][1]
    assert sent[1][1].endswith("Now: Analyse the time complexity of the Rust quicksort from step 1.")


def test_failed_dependency_output_is_not_passed_on():
    sent = []
    agent = _planning_agent(DEP_PLAN, clients=_clients(sent=sent, fail={"big"}))
    orch = _orchestrator(agent, max_workers=1)
    orch.limits = ExecutionLimits(max_replans=0)
    orch.run(_task())
    step2 = [p for _, p in sent if "Analyse the time complexity" in p]
    assert step2 and all("[error" not in p and "Results of earlier steps" not in p for p in step2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/router/execution/test_routed_orchestrator.py -q -k "planner or dependen"`
Expected: FAIL (the planner is never asked; dependencies are `[[], []]`; the error text is passed on).

- [ ] **Step 3: Implement in `src/router/execution/routed.py`.** Replace `plan()` with:

```python
    def plan(self, task: AgentTask) -> Plan:
        agent = self.agent
        subs = agent.decompose_plan(task.prompt, task.depth, sequential=self.sequential)
        texts = [s.text for s in subs]
        routes = [self._route_known(texts[0])] if len(texts) == 1 else self._route_batch(texts)
        context_chars = history_chars(agent._run_state().history)
        steps = []
        for i, (sub, (model_id, scores)) in enumerate(zip(subs, routes)):
            step_id = f"{task.task_id}.{i}"
            self._scores[step_id] = scores
            steps.append(self._new_step(step_id, sub.text, model_id, scores,
                                        [f"{task.task_id}.{d - 1}" for d in sub.depends_on],
                                        context_chars))
        plan = Plan(plan_id=f"{task.task_id}/plan0", task_id=task.task_id, steps=steps)
        self._score(plan)
        task.plan = plan
        self.plan_history.append(plan)
        return plan

    def _new_step(self, step_id: str, prompt: str, model_id: str, scores: dict[str, float],
                  dependencies: list[str], context_chars: int) -> PlanStep:
        profile = self._profile(model_id)
        in_tok = estimate_tokens(len(prompt) + context_chars)
        return PlanStep(
            step_id=step_id, prompt=prompt, model=profile,
            estimated_input_tokens=in_tok,
            estimated_output_tokens=self._cost_model.predict_output_tokens(profile, in_tok),
            dependencies=list(dependencies),
            predicted_quality=scores.get(model_id, float("nan")),
            alternatives=self._alternatives(scores, exclude=model_id),
        )
```

Replace `_run_step` and add `_contextual_prompt` above it:

```python
    def _contextual_prompt(self, step: PlanStep) -> tuple[str, str]:
        """``(call prompt, routing text)`` for ``step``. Only dependencies that
        *succeeded* contribute -- a failed one's ``[error: ...]`` placeholder
        is not a result. The routing text puts the task first, so encoder
        truncation drops context, not the task."""
        earlier = [self._calls[d] for d in step.dependencies if self._succeeded(d)]
        if not earlier:
            return step.prompt, step.prompt
        context = "\n\n".join(f"[{sc.prompt}]\n{sc.answer}" for sc in earlier)
        return (f"Results of earlier steps:\n{context}\n\nNow: {step.prompt}",
                f"{step.prompt}\n\nContext from earlier steps:\n{context}")

    def _run_step(self, step: PlanStep) -> SubCall:
        prompt, _ = self._contextual_prompt(step)
        sc = self.agent._call_model(prompt, step.model.model_id, self.task.depth + 1,
                                    predicted_quality=step.predicted_quality)
        sc.prompt = step.prompt
        sc.meta = {**sc.meta, "step_id": step.step_id}
        return sc
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/execution tests/router/agentic -q`
Expected: all pass, including the existing `test_sequential_steps_see_earlier_results` and `test_plan_routes_all_subprompts_in_one_batch`.

---

## Stage 2 — Late, context-aware routing

### Task 4: `route_late`

**Files:**
- Modify: `src/router/execution/routed.py` (`RoutedOrchestrator.__init__`, `plan`, `execute`, new `_assign_models`, `_run_step`, `_child_kwargs`; `PlanOrchestratorBridge.__init__` / `run`; `plan_orchestrator` docstring)
- Modify: `src/router/agentic/schemas.py` (`PlanSettings.route_late`)
- Modify: `src/router/agentic/router.py` (`from_settings` passes `route_late`)
- Test: `tests/router/execution/test_routed_orchestrator.py` (append)

**Interfaces:**
- Consumes: `_contextual_prompt` (Task 3), `_route_batch` (existing).
- Produces:
  - `RoutedOrchestrator(..., route_late: bool = False)` and `PlanOrchestratorBridge(..., route_late=False)`;
  - attributes `_planned: dict[str, str]` (step id → plan-time model) and `_pinned: set[str]`, which Task 5 fills;
  - `SubCall.meta["planned_model"]` and `meta["routed_model"]`;
  - `PlanSettings.route_late: bool = False`.

- [ ] **Step 1: Write the failing tests** (append):

```python
class ContextRouter(TextRouter):
    """Like TextRouter, but 'coder' tops any text containing code ('fn ')."""

    def predict_scores_text(self, texts, *, encoder=None):
        self.text_calls.append(list(texts))
        rows = [[0.7, 0.9, 0.95] if "fn " in t else [0.7, 0.9, 0.8] for t in texts]
        return pd.DataFrame(rows, index=range(len(texts)), columns=POOL)


def _scripted_clients(answer, sent=None):
    """Every model replies ``answer(model_id, prompt)``."""
    def make(m):
        def fn(prompt, **kw):
            if sent is not None:
                sent.append((m, prompt))
            return LLMResponse(answer(m, str(prompt)), cost=0.01, input_tokens=5, output_tokens=2)
        return LLMClient(CallableAdapter(m, fn))
    return ClientRegistry({m: make(m) for m in POOL})


def _code_answers(m, p):
    return "fn quicksort() {}" if p.startswith("Implement") else f"{m} answers"


def _plan_settings(**kw):
    from router.agentic.schemas import AgenticRouterSettings, load_settings
    return AgenticRouterSettings.model_validate(
        {**load_settings().model_dump(), "orchestrator": "plan", "warmup": False, **kw})


def test_late_routing_routes_a_dependent_step_on_its_context():
    router = ContextRouter()
    agent = _planning_agent(DEP_PLAN, router=router, clients=_scripted_clients(_code_answers))
    orch = _orchestrator(agent, route_late=True)
    assert orch.run(_task()).success
    step1, step2 = orch.subcalls
    assert step1.meta["planned_model"] == step1.meta["routed_model"] == "big"
    assert step2.model_id == "coder"
    assert step2.meta["planned_model"] == "big" and step2.meta["routed_model"] == "coder"
    assert router.text_calls[-1][0].startswith("Analyse the time complexity")   # task first


def test_without_late_routing_steps_keep_their_plan_time_model():
    agent = _planning_agent(DEP_PLAN, router=ContextRouter(), clients=_scripted_clients(_code_answers))
    orch = _orchestrator(agent)
    assert orch.run(_task()).success
    assert orch.subcalls[1].model_id == "big"


def test_route_late_setting_reaches_the_orchestrator():
    agent = AgenticRouter.from_settings(_plan_settings(plan={"route_late": True}), router=TextRouter())
    assert agent.orchestrator.route_late is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/router/execution/test_routed_orchestrator.py -q -k "late or route_late"`
Expected: FAIL (`TypeError: ... unexpected keyword argument 'route_late'`)

- [ ] **Step 3: Implement in `src/router/execution/routed.py`.**

`RoutedOrchestrator.__init__`:
- add the keyword `route_late: bool = False` after `sequential`;
- store `self.route_late = bool(route_late)`;
- after `self._scores = {}`, add:

```python
        self._planned: dict[str, str] = {}             # step_id -> plan-time model
        self._pinned: set[str] = set()                 # escalated this round: keep the model
```

In `plan()`, inside the loop, after `self._scores[step_id] = scores`, add `self._planned[step_id] = model_id`.

In `execute()`, change the first lines of the level loop to:

```python
        for level in _levels(plan):
            todo = [s for s in level if not self._succeeded(s.step_id)]
            todo = self._assign_models(plan, todo)
```

Add the method:

```python
    def _assign_models(self, plan: Plan, steps: list[PlanStep]) -> list[PlanStep]:
        """Late routing (``route_late``): re-route, in one batched router call,
        every step whose routing text now carries dependency results, and write
        the pick into ``plan``. A step pinned by escalation keeps its model; a
        step with no context yet keeps its plan-time route (same text -- no
        second encoder pass). A step the router can't score keeps its model."""
        if not self.route_late:
            return steps
        due = [s for s in steps if s.step_id not in self._pinned
               and self._contextual_prompt(s)[1] != s.prompt]
        if not due:
            return steps
        try:
            routes = self._route_batch([self._contextual_prompt(s)[1] for s in due])
        except LookupError:
            return steps
        changed = {}
        for s, (model_id, scores) in zip(due, routes):
            self._scores[s.step_id] = scores
            changed[s.step_id] = replace(s, model=self._profile(model_id),
                                         predicted_quality=scores.get(model_id, float("nan")))
        plan.steps = [changed.get(s.step_id, s) for s in plan.steps]
        return [changed.get(s.step_id, s) for s in steps]
```

In `_run_step`, replace the `sc.meta` line with:

```python
        sc.meta = {**sc.meta, "step_id": step.step_id,
                   "planned_model": self._planned.get(step.step_id, step.model.model_id),
                   "routed_model": step.model.model_id}
```

In `_child_kwargs`, add `"route_late": self.route_late` to the returned dict.

`PlanOrchestratorBridge.__init__`: add the keyword `route_late: bool = False` and store `self.route_late = route_late`. In `run()`, pass `route_late=self.route_late` to `RoutedOrchestrator(...)`. Add `route_late` to the list of keyword arguments in the `plan_orchestrator` docstring.

- [ ] **Step 4: Add the setting.** In `src/router/agentic/schemas.py`, `PlanSettings`:

```python
    #: route each step just before it runs, on its prompt plus dependency results
    route_late: bool = False
```

In `src/router/agentic/router.py` `from_settings`, add `route_late=s.plan.route_late` to the `plan_orchestrator(...)` call.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/execution tests/router/agentic -q`
Expected: all pass.

---

## Stage 3 — Escalation

### Task 5: Score-aware escalation replaces `alternatives[0]`

**Files:**
- Modify: `src/router/execution/routed.py` (`__init__`, `execute`, `_run_step`, `replan`, new `_escalation_pick`, `_escalate`, `_next_plan`)
- Test: `tests/router/execution/test_routed_orchestrator.py` (append)

**Interfaces:**
- Consumes: `_scores`, `_pinned`, `_profile`, `_score` (existing / Task 4).
- Produces:
  - `RoutedOrchestrator._escalation_pick(step: PlanStep) -> tuple[str, str]`, returning (model id, `"escalated"` or `"exhausted"`);
  - `_escalate(step) -> PlanStep`, which pins the step;
  - `_next_plan(plan, steps) -> Plan`;
  - attributes `_tried: dict[str, list[str]]` and `_escalation_note: dict[str, str]`;
  - `SubCall.meta["tried_models"]` and `meta["escalation"]`.

- [ ] **Step 1: Write the failing tests** (append; add `from dataclasses import replace` to the test file's imports):

```python
def test_escalation_never_retries_a_tried_model_and_takes_the_closest_step_up():
    orch = _orchestrator(AgenticRouter(TextRouter(), _clients(), decomposer=_decomposer(["a", "b"])))
    orch.task = _task()
    step = replace(orch.plan(orch.task).steps[0], model=orch._profile("small"))
    orch._scores[step.step_id] = {"small": 0.7, "big": 0.9, "coder": 0.8}
    assert orch._escalation_pick(step) == ("coder", "escalated")   # .8 is the closest step up
    orch._tried[step.step_id] = ["small", "coder"]
    assert orch._escalation_pick(step) == ("big", "escalated")
    orch._tried[step.step_id] = ["small", "coder", "big"]
    assert orch._escalation_pick(step) == ("small", "exhausted")


def test_escalation_prefers_the_cheapest_stronger_model():
    router = TextRouter(costs=[0.001, 0.001, 0.01])               # small, big, coder
    orch = _orchestrator(AgenticRouter(router, _clients(), decomposer=_decomposer(["a", "b"])))
    orch.task = _task()
    step = replace(orch.plan(orch.task).steps[0], model=orch._profile("small"))
    orch._scores[step.step_id] = {"small": 0.7, "big": 0.9, "coder": 0.8}
    assert orch._escalation_pick(step) == ("big", "escalated")


def test_step_failing_on_every_model_is_marked_exhausted():
    agent = AgenticRouter(TextRouter(), _clients(fail=set(POOL)), decomposer=_decomposer(["a", "b"]))
    orch = _orchestrator(agent, max_workers=1)
    orch.limits = ExecutionLimits(max_replans=3)
    assert not orch.run(_task()).success
    assert orch.subcalls[0].meta["tried_models"] == ["big", "coder", "small", "small"]
    assert orch.subcalls[0].meta["escalation"] == "exhausted"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/router/execution/test_routed_orchestrator.py -q -k "escalat or exhausted"`
Expected: FAIL (`AttributeError: ... '_escalation_pick'`)

- [ ] **Step 3: Implement in `src/router/execution/routed.py`.** In `__init__`, after `self._pinned`, add:

```python
        self._tried: dict[str, list[str]] = {}          # step_id -> models run, in order
        self._escalation_note: dict[str, str] = {}      # step_id -> "escalated" / "exhausted"
```

In `execute()`, in the loop that stores results, add the unpin line:

```python
            for step, sc in zip(todo, results):
                self._calls[step.step_id] = sc
                self._pinned.discard(step.step_id)
                fresh.append(sc)
```

In `_run_step`, record the model before the call and extend `meta`:

```python
    def _run_step(self, step: PlanStep) -> SubCall:
        prompt, _ = self._contextual_prompt(step)
        tried = self._tried.setdefault(step.step_id, [])
        tried.append(step.model.model_id)
        sc = self.agent._call_model(prompt, step.model.model_id, self.task.depth + 1,
                                    predicted_quality=step.predicted_quality)
        sc.prompt = step.prompt
        sc.meta = {**sc.meta, "step_id": step.step_id,
                   "planned_model": self._planned.get(step.step_id, step.model.model_id),
                   "routed_model": step.model.model_id, "tried_models": list(tried)}
        if step.step_id in self._escalation_note:
            sc.meta["escalation"] = self._escalation_note[step.step_id]
        return sc
```

Replace `replan()` with:

```python
    def replan(self, plan: Plan) -> Plan:
        """The next round: every step that didn't succeed is escalated
        (:meth:`_escalation_pick`) and pinned to its new model; steps that
        succeeded keep their output and are not re-run."""
        failed = {s.step_id for s in plan.steps if not self._succeeded(s.step_id)}
        steps = [self._escalate(s) if s.step_id in failed else s for s in plan.steps]
        return self._next_plan(plan, steps)

    def _escalation_pick(self, step: PlanStep) -> tuple[str, str]:
        """The model a re-run of ``step`` moves to. Never one already tried
        for it: the cheapest model scoring above the last one (the closest step
        up on equal cost), else the best untried one. ``"exhausted"`` when every
        scored model has been tried -- the step keeps its model."""
        scores = self._scores.get(step.step_id, {})
        current = step.model.model_id
        tried = set(self._tried.get(step.step_id, [])) | {current}
        untried = {m: q for m, q in scores.items() if m not in tried and math.isfinite(q)}
        if not untried:
            return current, "exhausted"
        base = scores.get(current, float("nan"))
        base = base if math.isfinite(base) else -math.inf
        stronger = [m for m, q in untried.items() if q > base]
        if stronger:
            costs = self.agent.router.cost_by_model()
            return min(stronger, key=lambda m: (costs.get(m, math.inf), untried[m])), "escalated"
        return max(untried, key=untried.get), "escalated"

    def _escalate(self, step: PlanStep) -> PlanStep:
        model_id, outcome = self._escalation_pick(step)
        self._escalation_note[step.step_id] = outcome
        self._pinned.add(step.step_id)
        profile = self._profile(model_id)
        return replace(
            step, model=profile,
            predicted_quality=self._scores.get(step.step_id, {}).get(model_id, float("nan")),
            estimated_output_tokens=self._cost_model.predict_output_tokens(
                profile, step.estimated_input_tokens),
        )

    def _next_plan(self, plan: Plan, steps: list[PlanStep]) -> Plan:
        new = Plan(plan_id=f"{plan.task_id}/plan{plan.revision + 1}", task_id=plan.task_id,
                   steps=steps, revision=plan.revision + 1)
        self._score(new)
        self.task.plan = new
        self.plan_history.append(new)
        return new
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/execution -q`
Expected: all pass. That includes the existing `test_failed_step_is_replanned_onto_its_next_best_model` (big → coder, the best untried model) and `test_replan_does_not_rerun_succeeded_steps`.

---

## Stage 4 — Final quality check

### Task 6: `QualityVerdict`, `parse_verdict`, `QualityChecker`, the `verify` prompt

**Files:**
- Create: `src/router/agentic/quality.py`
- Modify: `configs/agentic.yaml` (`verify` prompt)
- Test: `tests/router/agentic/test_quality.py` (create)

**Interfaces:**
- Consumes: `SubTask`, `parse_plan_lines(..., ref_limit=)` (Task 1); `message_text` (existing).
- Produces:
  - `QualityVerdict(passed: bool, score: float = nan, weak_steps: list[int] = [], issues: str = "", follow_ups: list[SubTask] = [], status: str = "ok", raw: str = "")` and `.as_dict() -> dict`;
  - `parse_verdict(text: str, *, pass_score: float = 7.0, n_steps: Optional[int] = None) -> QualityVerdict`;
  - `QualityChecker(chat_model=None, *, chat_for=None, prompt=None, pass_score=7.0, max_step_chars=1500, max_answer_chars=6000)`;
  - calling it, `QualityChecker.__call__(original: str, steps: Sequence[tuple[str, str]], answer: str) -> QualityVerdict`, sets `.on_call`.

- [ ] **Step 1: Invoke the `gating-new-classes` skill** for `QualityVerdict` (the parsed verdict) and `QualityChecker` (the LLM judge). Proceed as it directs.

- [ ] **Step 2: Write the failing tests** in `tests/router/agentic/test_quality.py`:

```python
"""The final quality check (router.agentic.quality)."""

from __future__ import annotations

import math

import pytest

from router.agentic.decompose import SubTask
from router.agentic.quality import QualityChecker, parse_verdict

PASS = "VERDICT: PASS\nSCORE: 8\nWEAK_STEPS: none\nFOLLOW_UPS:\nnone\nISSUES: none"
FAIL = ("VERDICT: FAIL\nSCORE: 4\nWEAK_STEPS: 2, 9\nFOLLOW_UPS:\n"
        "[after 2] Fix the compile error in the Rust quicksort from step 2.\n"
        "Add a test for an empty array.\n"
        "ISSUES: Rust code does not compile.")


def test_clean_pass():
    v = parse_verdict(PASS, n_steps=3)
    assert v.passed and v.score == 8 and v.status == "ok"
    assert v.weak_steps == [] and v.follow_ups == []


def test_fail_with_weak_steps_and_follow_ups():
    v = parse_verdict(FAIL, n_steps=3)
    assert not v.passed and v.status == "ok" and v.score == 4
    assert v.weak_steps == [2]                                   # 9 is not a reviewed step
    assert v.follow_ups == [
        SubTask("Fix the compile error in the Rust quicksort from step 2.", (2,)),
        SubTask("Add a test for an empty array."),
    ]
    assert v.issues == "Rust code does not compile."


def test_weak_step_zero_is_ignored():
    assert parse_verdict(FAIL.replace("2, 9", "0"), n_steps=3).weak_steps == []


def test_pass_below_pass_score_is_a_fail():
    v = parse_verdict(PASS.replace("SCORE: 8", "SCORE: 5"), pass_score=7)
    assert not v.passed and v.status == "ok"


def test_unparseable_reply_is_never_a_pass():
    v = parse_verdict("Looks great to me!")
    assert v.status == "unparsed" and not v.passed and math.isnan(v.score)


class _Chat:
    def __init__(self, reply):
        self.reply, self.requests = reply, []

    def invoke(self, text):
        self.requests.append(text)
        return self.reply


def test_checker_sends_steps_and_answer_and_truncates_long_text():
    chat = _Chat(PASS)
    checker = QualityChecker(chat, prompt="{original}|{steps}|{answer}",
                             max_step_chars=5, max_answer_chars=4)
    assert checker("the request", [("task one", "0123456789")], "abcdefgh").passed
    assert chat.requests == [
        "the request|Step 1: task one\nAnswer: 01234 …[truncated]|abcd …[truncated]"]


def test_checker_call_failure_is_an_error_verdict():
    class Broken:
        def invoke(self, text):
            raise RuntimeError("down")

    with pytest.warns(UserWarning, match="QualityChecker call failed"):
        v = QualityChecker(Broken(), prompt="{original}{steps}{answer}")("r", [], "a")
    assert v.status == "error" and not v.passed


def test_checker_counts_against_the_call_budget():
    calls = []
    checker = QualityChecker(_Chat(PASS), prompt="{original}{steps}{answer}")
    checker.on_call = lambda: calls.append(1)
    checker("r", [("t", "a")], "a")
    assert calls == [1]


def test_shipped_verify_prompt_formats():
    from router.agentic.schemas import default_prompts
    text = default_prompts()["verify"].format(original="o", steps="s", answer="a")
    assert "VERDICT:" in text and "WEAK_STEPS:" in text and "FOLLOW_UPS:" in text
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/router/agentic/test_quality.py -q`
Expected: ERROR (`ModuleNotFoundError: No module named 'router.agentic.quality'`)

- [ ] **Step 4: Add the `verify` prompt** to `configs/agentic.yaml`, under `prompts:`, after `plan:`:

```yaml
  verify: |-
    You are the final quality check for an answer that several models produced together.

    Original request:
    """
    {original}
    """

    The request was split into steps. Each step's task and answer (long answers are cut and marked [truncated]):
    {steps}

    Final answer given to the user:
    """
    {answer}
    """

    Check the final answer against the request: is every part answered, correct, consistent between steps, and (for code) plausible to compile and run? Then reply in exactly this format and nothing else:
    VERDICT: PASS or FAIL
    SCORE: a number from 0 to 10
    WEAK_STEPS: the numbers of the steps whose answers are wrong or incomplete, comma-separated, or none
    FOLLOW_UPS:
    one new self-contained task per line for anything missing or wrong that re-running a step would not fix; start it with [after N] if it needs step N's result; write none if there are none
    ISSUES: one line summarising the problems, or none
```

- [ ] **Step 5: Create `src/router/agentic/quality.py`:**

```python
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
_VERDICT_RE = re.compile(r"^\s*VERDICT:\s*(PASS|FAIL)\b", _FLAGS)
_SCORE_RE = re.compile(r"^\s*SCORE:\s*(\d+(?:\.\d+)?)", _FLAGS)
_WEAK_RE = re.compile(r"^\s*WEAK_STEPS:\s*(.*)$", _FLAGS)
_ISSUES_RE = re.compile(r"^\s*ISSUES:\s*(.*)$", _FLAGS)
_FOLLOW_RE = re.compile(r"^\s*FOLLOW_UPS:\s*(.*)$", _FLAGS)
_KEY_LINE_RE = re.compile(r"^\s*(VERDICT|SCORE|WEAK_STEPS|FOLLOW_UPS|ISSUES):", re.IGNORECASE)


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
        request = self.prompt.format(original=original[:2000], steps=body,
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/agentic -q`
Expected: all pass.

---

### Task 7: Checker rounds, best answer, `quality_check` in the result, settings

**Files:**
- Modify: `src/router/execution/routed.py`: `__init__` (`checker`), `execute`, new `_check`, `replan`, new `_last_failed_verdict`, module-level `_dependents` and `_quality_record`; `_plan_records`; `PlanOrchestratorBridge.__init__` / `run`; `plan_orchestrator` docstring.
- Modify: `src/router/agentic/router.py` (`_RunState.quality_check`, `_result`, `from_settings`)
- Modify: `src/router/agentic/result.py` (`AgenticResult.quality_check`)
- Modify: `src/router/agentic/schemas.py` (`QualityCheckSettings`, `PlanSettings.quality_check`, validation)
- Test: `tests/router/execution/test_routed_orchestrator.py` (append)

**Interfaces:**
- Consumes: `QualityVerdict`, `QualityChecker` (Task 6); `_escalate`, `_next_plan` (Task 5).
- Produces:
  - `RoutedOrchestrator(..., checker=None)`, where the checker is a callable `(original, [(task, answer)], answer) -> QualityVerdict`;
  - attributes `rounds: list[dict]` (`plan_id`, `answer`, `verdict`), `best: Optional[dict]` (`key`, `answer`, `round`, `verdict`) and `_round_changes: dict[str, dict]`;
  - `ExecutionResult.error_code == "quality_check_failed"`;
  - `PlanOrchestratorBridge(..., checker=None)`;
  - `AgenticResult.quality_check: Optional[dict]`;
  - `QualityCheckSettings(enabled=False, min_quality=0.8, pass_score=7.0, prompt="verify")`.

- [ ] **Step 1: Write the failing tests** (append; add `from router.agentic.quality import QualityVerdict` to the imports):

```python
class _ScriptedChecker:
    """Returns the next verdict per call (the last one repeats); records what it saw."""

    def __init__(self, *verdicts):
        self.verdicts, self.seen, self.on_call = list(verdicts), [], None

    def __call__(self, original, steps, answer):
        self.seen.append((original, list(steps), answer))
        return self.verdicts[min(len(self.seen), len(self.verdicts)) - 1]


def _fail(score, weak=(), follow_ups=()):
    return QualityVerdict(False, float(score), list(weak), "bad", list(follow_ups))


def _pass(score=9.0):
    return QualityVerdict(True, float(score))


THREE_STEP = ("1. Implement quicksort in Rust.\n"
              "2. [after 1] Analyse the Rust quicksort from step 1.\n"
              "3. Explain mergesort.")


def test_failed_check_escalates_the_weak_step_and_reruns_only_it_and_its_dependents():
    sent = []
    checker = _ScriptedChecker(_fail(4, weak=[1]), _pass())
    orch = _orchestrator(_planning_agent(THREE_STEP, clients=_clients(sent=sent)),
                         checker=checker, max_workers=1)
    orch.limits = ExecutionLimits(max_replans=2)
    assert orch.run(_task()).success and len(orch.rounds) == 2
    prompts = [str(p) for _, p in sent]
    assert sum(p == "Implement quicksort in Rust." for p in prompts) == 2       # step 1 re-ran
    assert sum("Analyse the Rust quicksort" in p for p in prompts) == 2         # its dependent re-ran
    assert sum("Explain mergesort" in p for p in prompts) == 1                  # untouched
    assert orch.subcalls[0].model_id == "coder"
    assert orch.subcalls[0].meta["escalation"] == "escalated"


def test_failing_check_runs_max_replans_plus_one_rounds_and_returns_the_best():
    checker = _ScriptedChecker(_fail(6.1, weak=[3]), _fail(7.4, weak=[3]), _fail(6.8, weak=[3]))
    agent = _planning_agent(THREE_STEP, orchestrator=plan_orchestrator(
        checker=checker, limits=ExecutionLimits(max_replans=2)))
    res = agent.run(COMPOUND)
    qc = res.quality_check
    assert qc["passed"] is False and qc["best_round"] == 1 and len(qc["rounds"]) == 3
    assert res.answer == checker.seen[1][2]


def test_a_passing_round_beats_a_higher_scoring_failing_round():
    checker = _ScriptedChecker(_fail(8, weak=[3]), _pass(7.5))
    agent = _planning_agent(THREE_STEP, orchestrator=plan_orchestrator(
        checker=checker, limits=ExecutionLimits(max_replans=2)))
    res = agent.run(COMPOUND)
    assert res.quality_check["passed"] is True and res.quality_check["best_round"] == 1
    assert res.answer == checker.seen[1][2]


def test_unparsed_check_returns_the_answer_flagged_without_a_repair_round():
    checker = _ScriptedChecker(QualityVerdict(False, status="unparsed"))
    agent = _planning_agent(THREE_STEP, orchestrator=plan_orchestrator(
        checker=checker, limits=ExecutionLimits(max_replans=2)))
    qc = agent.run(COMPOUND).quality_check
    assert len(qc["rounds"]) == 1 and qc["status"] == "unparsed" and qc["passed"] is False


def test_out_of_range_weak_steps_are_ignored():
    sent = []
    checker = _ScriptedChecker(_fail(4, weak=[0, 99]), _pass())
    agent = _planning_agent(THREE_STEP, clients=_clients(sent=sent), orchestrator=plan_orchestrator(
        checker=checker, limits=ExecutionLimits(max_replans=2)))
    assert agent.run(COMPOUND).quality_check["passed"] is True
    assert len(sent) == 3                                       # nothing re-ran


def test_over_budget_repair_round_returns_the_best_answer_not_an_error():
    checker = _ScriptedChecker(_fail(5, weak=[1, 3]))
    agent = _planning_agent(
        THREE_STEP, router=TextRouter(costs=[0.001, 0.01, 0.005]),
        orchestrator=plan_orchestrator(budget=0.04, checker=checker,
                                       limits=ExecutionLimits(max_cost=0.04, max_replans=2)))
    res = agent.run(COMPOUND)                                   # round 1 spends 0.03; repair needs 0.02
    assert "over budget" not in res.answer and res.answer == checker.seen[0][2]
    assert res.quality_check["best_round"] == 0


def test_quality_check_setting_builds_a_routed_checker():
    from router.agentic.quality import QualityChecker
    agent = AgenticRouter.from_settings(
        _plan_settings(plan={"quality_check": {"enabled": True, "min_quality": 0.5}}), router=TextRouter())
    assert isinstance(agent.orchestrator.checker, QualityChecker)


def test_unknown_verify_prompt_is_rejected():
    with pytest.raises(ValueError, match="quality_check.prompt"):
        _plan_settings(plan={"quality_check": {"enabled": True, "prompt": "nope"}})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/router/execution/test_routed_orchestrator.py -q -k "check or round or verify or weak or budget_repair"`
Expected: FAIL (`TypeError: ... unexpected keyword argument 'checker'`)

- [ ] **Step 3: Implement the orchestrator side in `src/router/execution/routed.py`.**

In the `TYPE_CHECKING` block, add `from ..agentic.quality import QualityVerdict`.

`RoutedOrchestrator.__init__`: add the keyword `checker=None` after `route_late`. Then:

```python
        #: final-answer judge: ``(original, [(task, answer)], answer) -> QualityVerdict``
        self.checker = checker
        #: one dict per checked round: ``plan_id``, ``answer``, ``verdict``
        self.rounds: list[dict] = []
        #: the best checked round so far (a passed round beats any failed one)
        self.best: Optional[dict] = None
        self._round_changes: dict[str, dict] = {}       # new plan_id -> what the repair changed
```

Add `"checker": self.checker` to `_child_kwargs`.

In `execute()`, replace the block from `failed = ...` through the `ExecutionResult(...)`
construction with:

```python
        failed = [sc for sc in self.subcalls if sc.error is not None]
        output = ""
        error_code = "step_failed" if failed else None
        if not failed:
            if len(self.subcalls) == 1:
                output = self.subcalls[0].answer
            else:
                output = self.agent.synthesizer(self.task.prompt,
                                                [(sc.prompt, sc.answer) for sc in self.subcalls])
            if self.checker is not None:
                error_code = self._check(plan, output)
        spent = total_cost(fresh)
        if spent is None:
            # a successful call with neither an actual cost nor an estimate:
            # the ledger needs a number, so commit 0 -- but say so, never silently
            warnings.warn(
                f"plan {plan.plan_id!r}: a step's cost is unknown (no prices, no usage, no "
                "router cost data); committing 0 for it to the budget ledger",
                stacklevel=2,
            )
            spent = float(sum(sc.cost for sc in fresh if sc.cost is not None))
        result = ExecutionResult(
            output=output,
            actual_cost=spent,
            actual_latency=time.perf_counter() - t0,
            input_tokens=int(sum(sc.meta.get("input_tokens") or 0 for sc in fresh)),
            output_tokens=int(sum(sc.meta.get("output_tokens") or 0 for sc in fresh)),
            error_code=error_code,
        )
        self.attempts.append((plan.plan_id, result))
        return result

    def _check(self, plan: Plan, output: str) -> Optional[str]:
        """Check this round's answer, record the round and the best answer so
        far. ``"quality_check_failed"`` for a parsed failing verdict; an
        unparseable or failed check never triggers a repair round -- the
        answer is returned flagged instead."""
        verdict = self.checker(self.task.prompt, [(sc.prompt, sc.answer) for sc in self.subcalls],
                               output)
        self.rounds.append({"plan_id": plan.plan_id, "answer": output, "verdict": verdict})
        key = (verdict.passed, verdict.score if math.isfinite(verdict.score) else -math.inf)
        if self.best is None or key >= self.best["key"]:
            self.best = {"key": key, "answer": output, "round": len(self.rounds) - 1,
                         "verdict": verdict}
        return "quality_check_failed" if verdict.status == "ok" and not verdict.passed else None
```

Replace `replan()` with the repair-round version, and add `_last_failed_verdict`:

```python
    def replan(self, plan: Plan) -> Plan:
        """The repair round (merge rule: previous plan + repair instructions).
        Re-run, escalated (:meth:`_escalate`): every step that failed and
        every step the last quality check named weak. Their dependents,
        transitively, lose their stored output and re-run with the new context.
        Every other step keeps its output and is not re-run."""
        verdict = self._last_failed_verdict(plan)
        weak = set()
        if verdict is not None:
            weak = {plan.steps[k - 1].step_id for k in verdict.weak_steps
                    if 1 <= k <= len(plan.steps)}
        failed = {s.step_id for s in plan.steps if not self._succeeded(s.step_id)}
        escalate = weak | failed
        for sid in weak | _dependents(plan, escalate):
            self._calls.pop(sid, None)
        steps = [self._escalate(s) if s.step_id in escalate else s for s in plan.steps]
        new = self._next_plan(plan, steps)
        self._round_changes[new.plan_id] = {"escalated": sorted(escalate)}
        return new

    def _last_failed_verdict(self, plan: Plan) -> Optional["QualityVerdict"]:
        """The verdict that failed ``plan``'s round, if a check failed it."""
        if self.rounds and self.rounds[-1]["plan_id"] == plan.plan_id:
            verdict = self.rounds[-1]["verdict"]
            if verdict.status == "ok" and not verdict.passed:
                return verdict
        return None
```

Add module-level helpers after `_levels`:

```python
def _dependents(plan: Plan, roots: set[str]) -> set[str]:
    """Every step depending on one of ``roots``, directly or through other
    steps (``roots`` themselves excluded)."""
    out: set[str] = set()
    frontier = set(roots)
    while frontier:
        frontier = {s.step_id for s in plan.steps if frontier & set(s.dependencies)} - out - roots
        out |= frontier
    return out


def _quality_record(orch: RoutedOrchestrator) -> dict:
    """``AgenticResult.quality_check``: the best round's verdict, which round
    it was, and every round with what its repair changed."""
    best = orch.best
    return {**best["verdict"].as_dict(), "best_round": best["round"],
            "rounds": [{"plan_id": r["plan_id"], **r["verdict"].as_dict(),
                        **orch._round_changes.get(r["plan_id"], {})} for r in orch.rounds]}
```

In `_plan_records`:
- before the loop, add `checks = {r["plan_id"]: r["verdict"] for r in orch.rounds}`;
- add these two keys to each row dict:

```python
            "quality_score": checks[plan.plan_id].score if plan.plan_id in checks else None,
            "quality_passed": checks[plan.plan_id].passed if plan.plan_id in checks else None,
```

`PlanOrchestratorBridge.__init__`: add the keyword `checker=None` and store `self.checker = checker`. In `run()`:
- pass `checker=self.checker` to `RoutedOrchestrator(...)`;
- right after constructing `orch`, add:

```python
        if self.checker is not None and hasattr(self.checker, "on_call"):
            self.checker.on_call = agent._use_call
```

And after `agent._run_state().plans.extend(...)`, before the `TERMINATED` check, add:

```python
        if orch.best is not None:
            # a checked answer wins over an over-budget or failed later round
            agent._run_state().quality_check = _quality_record(orch)
            return orch.best["answer"], list(orch.subcalls)
```

Add `checker` to the list of keyword arguments in the `plan_orchestrator` docstring.

- [ ] **Step 4: Thread `quality_check` into the result.**
  - `src/router/agentic/router.py`, `_RunState`, after `helper_calls`:

```python
    #: the plan orchestrator's final quality check (see AgenticResult.quality_check)
    quality_check: Optional[dict] = None
```

  - In `_result`, add `quality_check=req.state.quality_check,` to the `context = dict(...)`.
  - `src/router/agentic/result.py`, `AgenticResult`, after `helper_calls`:

```python
    #: the plan orchestrator's final quality check, when one ran: the best
    #: round's verdict (passed, score, weak_steps, issues, follow_ups, status),
    #: best_round, and every round with its escalations / follow-ups
    quality_check: Optional[dict] = None
```

- [ ] **Step 5: Add the settings.** In `src/router/agentic/schemas.py`, above `PlanSettings`:

```python
class QualityCheckSettings(BaseModel):
    """The plan orchestrator's final quality check (``router.agentic.quality``)."""

    model_config = _STRICT

    enabled: bool = False
    #: the judge = cheapest pool model predicted >= this for the request
    min_quality: float = Field(0.8, ge=0.0, le=1.0)
    #: a PASS verdict also needs SCORE >= this (0-10)
    pass_score: float = Field(7.0, ge=0.0, le=10.0)
    prompt: str = "verify"
```

In `PlanSettings`, add:

```python
    quality_check: QualityCheckSettings = Field(default_factory=QualityCheckSettings)
```

In `_references_resolve`, before `return self`, add:

```python
        q = self.plan.quality_check
        if self.orchestrator == "plan" and q.enabled and q.prompt not in self.prompts:
            raise ValueError(f"plan.quality_check.prompt: no prompt {q.prompt!r}; "
                             f"known: {sorted(self.prompts)}")
```

In `src/router/agentic/router.py` `from_settings`, inside the `if s.orchestrator == "plan":` branch, before `orchestrator = plan_orchestrator(...)`:

```python
            q = s.plan.quality_check
            checker = None
            if q.enabled:
                from .quality import QualityChecker
                # late-bound like the routed helpers: called only once `agent` exists
                checker = QualityChecker(
                    chat_for=lambda request: agent.helper_call("verify", request, q.min_quality),
                    prompt=prompts[q.prompt], pass_score=q.pass_score)
```

Then add `checker=checker` to the `plan_orchestrator(...)` call.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/execution tests/router/agentic tests/e2e -q`
Expected: all pass.

---

## Stage 5 — Follow-up steps

### Task 8: Checker follow-ups become new steps (merge rule, fan-out cap)

**Files:**
- Modify: `src/router/execution/routed.py` (`replan`, new `_follow_up_steps`)
- Test: `tests/router/execution/test_routed_orchestrator.py` (append)

**Interfaces:**
- Consumes: `QualityVerdict.follow_ups: list[SubTask]` (Task 6); `_new_step` (Task 3); `_route_batch`; `limits.max_fanout`.
- Produces: follow-up step ids `"<task_id>.r<revision>.<i>"`, and `_round_changes[plan_id]` keys `follow_ups` (texts) and `dropped_follow_ups` (texts).

- [ ] **Step 1: Write the failing tests** (append; add `from router.agentic.decompose import SubTask` to the imports):

```python
def test_checker_follow_up_becomes_a_step_with_its_dependency():
    sent = []
    follow = SubTask("Fix the compile error in the Rust quicksort from step 1.", (1,))
    checker = _ScriptedChecker(_fail(5, follow_ups=[follow]), _pass())
    orch = _orchestrator(_planning_agent(THREE_STEP, clients=_clients(sent=sent)),
                         checker=checker, max_workers=1)
    orch.limits = ExecutionLimits(max_replans=1)
    assert orch.run(_task()).success
    last = orch.task.plan.steps[-1]
    assert last.prompt == follow.text and last.dependencies == ["root.0"]
    fix = [str(p) for _, p in sent if "Fix the compile error" in str(p)]
    assert len(fix) == 1 and "Results of earlier steps" in fix[0]
    assert len(sent) == 4                                       # 3 original steps once + the follow-up
    assert len(checker.seen[1][1]) == 4                         # round 2 reviews the follow-up too


def test_follow_ups_respect_max_fanout():
    follows = [SubTask(f"Extra task number {i}.") for i in range(5)]
    checker = _ScriptedChecker(_fail(5, follow_ups=follows), _pass())
    orch = _orchestrator(_planning_agent(THREE_STEP), checker=checker)
    orch.limits = ExecutionLimits(max_replans=1, max_fanout=4)
    orch.run(_task())
    assert len(orch.task.plan.steps) == 4
    changes = orch._round_changes[orch.task.plan.plan_id]
    assert changes["follow_ups"] == ["Extra task number 0."]
    assert changes["dropped_follow_ups"] == [f"Extra task number {i}." for i in range(1, 5)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/router/execution/test_routed_orchestrator.py -q -k "follow"`
Expected: FAIL (the plan still has 3 steps; `KeyError: 'follow_ups'`).

- [ ] **Step 3: Implement in `src/router/execution/routed.py`.** In `replan()`,
  replace the last three lines (`steps = ...` through `return new`) with:

```python
        steps = [self._escalate(s) if s.step_id in escalate else s for s in plan.steps]
        added, dropped = self._follow_up_steps(plan, steps, verdict)
        new = self._next_plan(plan, steps + added)
        self._round_changes[new.plan_id] = {"escalated": sorted(escalate),
                                            "follow_ups": [s.prompt for s in added],
                                            "dropped_follow_ups": dropped}
        return new
```

Add the method:

```python
    def _follow_up_steps(self, plan: Plan, steps: list[PlanStep],
                         verdict: Optional["QualityVerdict"]) -> tuple[list[PlanStep], list[str]]:
        """New steps for the checker's follow-ups, up to ``limits.max_fanout``
        steps in the plan (the rest are dropped and reported). ``[after N]``
        names the N-th step the checker reviewed (plan order); follow-ups are
        routed at plan time like the first plan's steps."""
        if verdict is None or not verdict.follow_ups:
            return [], []
        room = max(0, self.limits.max_fanout - len(steps))
        take = verdict.follow_ups[:room]
        dropped = [t.text for t in verdict.follow_ups[room:]]
        if not take:
            return [], dropped
        routes = self._route_batch([t.text for t in take])
        context_chars = history_chars(self.agent._run_state().history)
        added = []
        for i, (task, (model_id, scores)) in enumerate(zip(take, routes)):
            step_id = f"{plan.task_id}.r{plan.revision + 1}.{i}"
            self._scores[step_id] = scores
            self._planned[step_id] = model_id
            deps = [plan.steps[d - 1].step_id for d in task.depends_on if 1 <= d <= len(plan.steps)]
            added.append(self._new_step(step_id, task.text, model_id, scores, deps, context_chars))
        return added, dropped
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/execution tests/router/agentic tests/e2e -q`
Expected: all pass.

---

## Wrap-up

### Task 9: Shipped config, docs, full suite, live run

**Files:**
- Modify: `configs/agentic.yaml` (`plan:` section)
- Modify: `docs/agentic_router.md` (plan orchestrator section), `docs/agentic_router_runbook.md` (§4.5 options table)
- Create (scratchpad, not the repo): `C:/Users/dogpl/AppData/Local/Temp/claude/c--Users-dogpl-VisualCode-Projects-agentic-llm-router/203f81df-032e-48c9-8ef0-69fdd28d26bd/scratchpad/run_plan_layers.py` (written `SCRATCH/run_plan_layers.py` below)

**Interfaces:**
- Consumes: everything above.
- Produces: documented config; the live verification log.

- [ ] **Step 1: Update the shipped `plan:` section in `configs/agentic.yaml`.**
  `orchestrator:` stays `recursive`. Replace the `plan:` block with:

```yaml
  plan:                           # used when orchestrator: plan
    budget: null                  # USD per request (null = unlimited)
    max_workers: 4
    sequential: false
    max_replans: 2                # repair rounds after a failed step or quality check
    route_late: true              # route each step on its prompt + dependency results
    planner_prompt: plan          # numbered steps with [after N] dependencies (LLM decomposers)
    quality_check:
      enabled: true               # judge the synthesized answer; repair weak steps / add follow-ups
      min_quality: 0.8            # judge = cheapest pool model predicted >= this
      pass_score: 7               # PASS also needs SCORE >= this (0-10)
      prompt: verify
```

- [ ] **Step 2: Document it.**
  - **`docs/agentic_router.md`:** under the plan orchestrator summary, add a
    short "Adaptive plans" paragraph:
    - the planner's `[after N]` dependencies, and that dependent steps
      receive successful dependency outputs;
    - `route_late` (`meta.planned_model` / `meta.routed_model`);
    - escalation (`meta.tried_models`, `meta.escalation`, never retrying a
      tried model);
    - the quality check and its repair rounds (`max_replans`, follow-ups,
      `max_fanout`);
    - `AgenticResult.quality_check`, which returns the best round, flagged.
  - **`docs/agentic_router_runbook.md` §4.5:** add rows for `route_late`,
    `planner_prompt` and `quality_check.{enabled,min_quality,pass_score,prompt}`,
    with the same meanings as the YAML comments above. Change the
    `max_replans` row to "repair rounds after a failed step or quality check".

- [ ] **Step 3: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/router/agentic tests/router/execution tests/e2e -q`, then `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass (the previous count, 1161 passed / 2 skipped, plus the new tests).

- [ ] **Step 4: Write the live runner** at `SCRATCH/run_plan_layers.py`, the
  scratchpad path given under **Files**.
  It reuses the local-Ollama stand-in mapping from the earlier
  `run_nirt_layers.py`:

```python
"""The quicksort prompt through the adaptive plan orchestrator on the real
nirt-2d-projected router, every pool id answered by a local Ollama stand-in."""

import sys
import time

import ollama
import yaml

sys.path.insert(0, "scripts/agentic")
from run_agentic_nirt import ORCHESTRATE_PROMPT

from router.agentic import AgenticRouter, ClientRegistry
from router.agentic.schemas import AgenticRouterSettings, load_settings
from router.llm.adapters.ollama import OllamaAdapter
from router.llm.client import LLMClient

TIER_TAG = {"strong": "qwen2.5:3b", "mid": "phi3.5", "small": "qwen2.5:1.5b"}
DIRECT = {"deepseek_chat": "deepseek-llm:7b-chat", "deepseek_coder": "deepseek-coder:1.3b"}
serve = yaml.safe_load(open("configs/serve_anthropic.yaml", encoding="utf-8"))
shared = ollama.Client(host="http://localhost:11434")
LOCAL = {}


class Capped(OllamaAdapter):
    def complete(self, prompt, **kw):
        kw.setdefault("max_tokens", 350)
        return super().complete(prompt, **kw)


def build(pool_id):
    tag = DIRECT.get(pool_id) or TIER_TAG[serve["pool"].get(pool_id, serve["default_tier"])]
    LOCAL[pool_id] = tag
    return LLMClient(Capped(tag, client=shared))


base = load_settings()
settings = AgenticRouterSettings.model_validate({
    **base.model_dump(), "orchestrator": "plan", "warmup": False,
    "decomposer": {"kind": "routed", "min_quality": 0.75},
    "synthesizer": {"kind": "routed", "min_quality": 0.75, "max_body_chars": 16000},
})
agent = AgenticRouter.from_settings(settings)
agent.clients = ClientRegistry({m: build(m) for m in agent.router.model_ids}, default_factory=build)

t0 = time.perf_counter()
res = agent.run(ORCHESTRATE_PROMPT)
wall = time.perf_counter() - t0
qc = res.quality_check or {}
for h in res.helper_calls:
    print(f"◆ {h['role'].upper()} → {h['model_id']} (local {LOCAL.get(h['model_id'])}, "
          f"q={h['predicted_quality']:.2f}, {h['latency']:.1f}s)\n  {h['output'][:600]}\n")
for i, sc in enumerate(res.steps, 1):
    m = sc.meta
    print(f"● step {i} [{m.get('step_id')}] planned {m.get('planned_model')} → routed "
          f"{m.get('routed_model')} (local {LOCAL.get(sc.model_id)}) tried={m.get('tried_models')} "
          f"escalation={m.get('escalation')}\n  task: {sc.prompt}\n  answer: {sc.answer[:500]}\n")
for r in qc.get("rounds", []):
    print(f"ROUND {r['plan_id']}: passed={r['passed']} score={r['score']} weak={r['weak_steps']} "
          f"escalated={r.get('escalated')} follow_ups={r.get('follow_ups')} issues={r['issues']}")
moved = sum(sc.meta.get("planned_model") != sc.meta.get("routed_model") for sc in res.steps)
print(f"\nMETRICS: rounds={len(qc.get('rounds', []))} best_round={qc.get('best_round')} "
      f"passed={qc.get('passed')} planned!=routed={moved}/{len(res.steps)} "
      f"helper_calls={len(res.helper_calls)} wall={wall:.1f}s")
print(f"\nFINAL ANSWER:\n{res.answer[:3000]}")
```

- [ ] **Step 5: Run it and check the acceptance criteria**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe SCRATCH/run_plan_layers.py`, with `SCRATCH` replaced by the scratchpad path from **Files**,, in the background. The local models take minutes.

Expected (structural acceptance):
- the step that analyses complexity has `[after N]` on the Rust step, and its answer refers to that code;
- `rounds` ≤ 3;
- every escalated step's `tried_models` has no repeats before an `exhausted`;
- `quality_check` is filled in.

Report in chat, mobile-friendly:
- per round: the steps, planned → routed model, escalations;
- the verdict: score, weak steps, follow-ups;
- the planned ≠ routed rate and wall time;
- the caveat that small local judges are unreliable.

Offer one `clients: anthropic` comparison run. It needs an API key and the user's OK.
