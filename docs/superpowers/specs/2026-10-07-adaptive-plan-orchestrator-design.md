# Adaptive plan orchestrator: follow-up tasks, context-aware re-routing, final quality check

## Context

Live runs of the agentic router on local models showed problems that a
stronger model would not fix:
- Sub-tasks never see earlier results. "Analyse the complexity of the
  *implemented* quicksort" ran without the Rust code from the step before.
- Each step's model is chosen once, from the bare sub-task text, before
  anything runs.
- Nothing checks the final answer. Broken code and wrong facts went straight
  to the user.

The user wants planning rethought along three lines. Decisions so far:
1. **Follow-up tasks.**
   - The planner declares dependencies between steps, and a dependent step
     receives the outputs of the steps it depends on.
   - When the final check fails, it can **add** new follow-up steps for
     anything missing or wrong.
2. **Re-routing using context.**
   - **Route late:** each step is routed just before it runs, scoring its
     prompt *together with* its dependency outputs.
   - **Escalate:** a step judged weak is re-run on a stronger pool model.
3. **Final quality check.**
   - One judge call per round reviews the synthesized answer and names the
     weak steps and the missing follow-ups. That drives the next repair round.
   - Rounds are capped (default 2). If the answer still fails, the run returns
     the **best-scoring answer from any round, flagged** as failed.

**Design constraint:** the planner only *describes* dependencies. Execution
owns context and model choice. A model picked at plan time is for forecasting
and budget reservation. The model picked at run time is the one that answers.

### Approach

Extend the existing plan orchestrator (`RoutedOrchestrator` in
[src/router/execution/routed.py](src/router/execution/routed.py), used as
`orchestrator: plan`). It already has most of the machinery:
- `PlanStep.dependencies`, run in topological order by `_levels()`;
- context injection in `_run_step()`;
- the `Orchestrator.run` loop: plan → reserve budget → execute → replan while
  unsuccessful, up to `limits.max_replans`
  ([execution/orchestrator.py:84](src/router/execution/orchestrator.py#L84));
- plan revisions and `plan_history`.

A failed quality check becomes an unsuccessful `ExecutionResult`, and
`replan()` becomes the repair round, so the template loop is unchanged.
`RecursiveOrchestrator` stays as it is.

## 1. Planner with dependencies (follow-ups declared up front)

- **New prompt `plan`** in `configs/agentic.yaml`. Same rules as `decompose`:
  self-contained steps, each listed once, the example's wording not reused.
  Plus: number the steps, and put `[after N]` on a step that needs an earlier
  step's result. Such a step names what it uses ("the Rust implementation
  from step 2"). Example:
  `1. Implement quicksort in Rust.` / `2. [after 1] Analyse the time complexity of the Rust quicksort from step 1.`
- **Parser:** `parse_plan_lines(text) -> list[SubTask]` in `src/router/agentic/decompose.py`.
  - `SubTask` is a small frozen dataclass: `text`, `depends_on: tuple[int, ...]`.
  - It reuses the `_parse` cleaning and dedupe logic.
  - `[after N]` references to unknown, forward or self steps are dropped,
    never fatal. No markers anywhere means independent steps.
- **New method `LLMDecomposer.plan(prompt) -> list[SubTask]`** sends the
  `plan` prompt through the same `chat_for` / routed-helper path. On any
  failure it falls back to `[SubTask(s) for s in fallback(prompt)]`.
- **New method `AgenticRouter.decompose_plan(prompt, depth) -> list[SubTask]`**
  applies the same atomic/multi-part gate as `decompose()`, so nothing runs
  away. A non-LLM decomposer gives independent steps, chained only when
  `plan.sequential` is true (today's behaviour).
- **`RoutedOrchestrator.plan()`** uses it and maps `depends_on` onto
  `PlanStep.dependencies`. `Plan.validate()` already rejects cycles and
  unknown ids.

## 2. Context-aware execution

- **`_run_step` context fix:** pass outputs only from dependencies that
  *succeeded*. Today a failed dependency's `[error: …]` placeholder goes
  downstream.
- **Late routing** (`plan.route_late: true`):
  - Before each level runs, route all its steps in one batched
    `route_text` call. Each step's routing text is the step prompt first,
    then its dependency outputs, so encoder truncation keeps the task.
  - Use the run's `lam`/`alpha` resolution, as `_route_batch` does.
  - The step's model becomes the late pick. Its fresh scores go into
    `_scores[step_id]`, and `meta` records `planned_model` vs `routed_model`.
  - Plan-time routing stays: it is what the forecaster prices and the ledger
    reserves before execution.
- **Escalation picks** (`_escalate(step)`), for steps the checker named weak
  and for steps that errored:
  - Score the step's contextual prompt.
  - Exclude every model already tried for that step (tracked in
    `meta["tried_models"]`).
  - Take the cheapest model scoring above the last model's score. If there
    isn't one, take the best remaining model. If nothing is left, keep the
    current model and record `escalation: "exhausted"`.
  - This replaces today's `alternatives[0]` swap. `alternatives` stays as the
    plan-time ranking for the forecaster.

## 3. Final quality check

- **New helper `QualityChecker`** in a new file
  `src/router/agentic/quality.py`. It is LLM-backed, with `chat_for` from
  `agent.helper_call("verify", request, min_quality)`, the same routed-helper
  mechanism as the decomposer and synthesizer.
  - Input: the original request, the numbered steps with their answers, and
    the synthesized answer.
  - Output: a `QualityVerdict`:
    - `passed`, `score` (0-10), `weak_steps` (step numbers), `issues`;
    - `follow_ups: list[SubTask]`, parsed with `parse_plan_lines`, so a
      follow-up can say `[after 2]`;
    - `status`: `ok` / `unparsed` / `error`.
- **New prompt `verify`** in `configs/agentic.yaml`. It asks for a strict
  block:
  ```
  VERDICT: PASS|FAIL
  SCORE: <0-10>
  WEAK_STEPS: <comma-separated step numbers, or none>
  FOLLOW_UPS:
  <one self-contained task per line, optional [after N]>
  ISSUES: <one line>
  ```
  A verdict counts as a pass if it says PASS and SCORE ≥ `pass_score`.
- **Unparseable or failed check:** returns `status="unparsed"`/`"error"`. No
  repair round runs, and the answer is returned flagged. It is never counted
  as passed.
- **`execute()`:** synthesize as today, then run the check. On a fail it
  returns `ExecutionResult(error_code="quality_check_failed")`, and the
  existing loop calls `replan()`. Every round's
  `(answer, verdict, plan_id)` goes into `self.rounds`.

## 4. Repair round (`replan`)

**Merge rule:** revised plan = previous plan + repair instructions. A repair
round never replaces the plan wholesale:
- every step that is not named weak, and doesn't depend on one, keeps its
  stored output;
- those outputs remain available as context to escalated steps and to new
  follow-ups;
- correct work from earlier rounds is never thrown away or re-paid for.

When the last result is `quality_check_failed`:
1. **Escalate weak steps.** Steps named in `weak_steps` get their stored
   attempt dropped and are marked to escalate (§2). Steps that errored are
   escalated too.
2. **Invalidate dependents.** Every step depending on a re-run step, directly
   or indirectly, has its attempt dropped, so it re-runs with fresh context.
   This also fixes today's stale-context bug.
3. **Add follow-ups.** `verdict.follow_ups` become new steps with ids
   `<task>.r<round>.<i>`. Their `[after N]` refer to existing step numbers.
   They are routed at plan time like other steps and priced by the
   forecaster.
4. **Fan-out cap:** the plan never grows past `limits.max_fanout` steps;
   extra follow-ups are dropped and recorded.

The new plan revision is scored, and the existing loop reserves its estimate
on the ledger. An over-budget repair round ends the loop, which then
returns the best round. A plain `step_failed` result (no check) keeps today's
behaviour, except it now uses the escalation pick.

## 5. Result and the best-answer rule

- **Best answer tracked during the loop:** `RoutedOrchestrator` keeps
  `best_answer` / `best_score` / `best_round`, updated after every check, so
  nothing is reconstructed afterwards.
  - Ties and unscored rounds go to the later round; if no round was scored,
    the last round wins.
  - `PlanOrchestratorBridge.run` returns `best_answer`. For example, rounds
    scoring 6.1 / 7.4 / 6.8 that never pass return round 2's answer,
    flagged as failed.
- **New field `AgenticResult.quality_check: Optional[dict]`**:
  - `passed`, `score`, `issues`, `weak_steps`;
  - `rounds` (per round: score, verdict, escalated steps, added follow-ups,
    checker model);
  - `best_round`, `status`.
  - It is passed through `_RunState`, the same way as `plans` and
    `helper_calls`.
- **`_plan_records`** gains `quality_score` / `quality_passed` per revision.
  Helper calls (planner, synthesizer, checker) keep being recorded in
  `helper_calls`.

## 6. Config (`configs/agentic.yaml`, `PlanSettings` in `src/router/agentic/schemas.py`)

```yaml
orchestrator: plan
plan:
  budget: null
  max_workers: 4
  sequential: false
  max_replans: 2            # = repair rounds
  route_late: true
  quality_check:
    enabled: true
    min_quality: 0.8        # judge = cheapest pool model predicted >= this
    pass_score: 7
    prompt: verify
  planner_prompt: plan
```

- **Code defaults are off:** `route_late` and `quality_check.enabled` default
  to off in the code, so existing callers and tests keep today's behaviour.
  The YAML above turns them on.
- **Decomposer:** dependencies need `decomposer.kind: llm|routed`. A naive
  decomposer still works, just with no dependencies.
- **Untouched:** `orchestrator: recursive` (still the default) and the
  synthesizer/decomposer kinds.

## Implementation order (incremental; each stage tested before the next)

Each stage is a working, test-covered change on its own. Each also isolates
one measurable source of value:

| Stage | Change | Question it answers |
|---|---|---|
| 1 | Planner dependencies + `SubTask`/`parse_plan_lines` + `decompose_plan` + context fix (successful dependencies only) | Do downstream steps get better when they see the outputs they depend on? |
| 2 | Late routing (`route_late`), `planned_model`/`routed_model` in meta | Does context change model choice, and how often? (planned ≠ routed rate) |
| 3 | Escalation (`_escalate`, `tried_models`, "exhausted") replacing `alternatives[0]` | Does a targeted retry recover failures more cheaply than always using the strong model? |
| 4 | `QualityChecker` + verdict parser + `quality_check_failed` result + best-answer tracking | Does verification improve final correctness? |
| 5 | Follow-up steps from the checker, merge rule, fan-out cap | Can the system repair the gaps it finds? |

The core change is stages 1-2. Stages 3-5 sit on the existing replan loop.
Config keys arrive with the stage that uses them.

## Files

- `src/router/execution/routed.py`:
  - `plan()` takes dependencies from the planner;
  - late routing, the context fix and `_escalate` in `execute()`/`_run_step`;
  - `replan()` does repair rounds;
  - the bridge's best-round selection and `quality_check` output.
- `src/router/agentic/decompose.py`: `SubTask`, `parse_plan_lines`, `LLMDecomposer.plan`.
- `src/router/agentic/quality.py` (new): `QualityChecker`, `QualityVerdict`, `parse_verdict`.
- `src/router/agentic/router.py`:
  - `decompose_plan()`;
  - `from_settings` wiring for the planner prompt, `QualityChecker`,
    `route_late` and the plan settings;
  - `quality_check` passed into `_RunState`/`_result`.
- `src/router/agentic/result.py`: the `quality_check` field.
- `src/router/agentic/schemas.py`: `PlanSettings` gets `route_late`,
  `quality_check` (a nested model) and `planner_prompt`. Prompt names are
  validated.
- `configs/agentic.yaml`: the `plan` and `verify` prompts and the `plan:` section.
- **Tests:** `tests/router/execution/test_routed_orchestrator.py`, plus new
  `tests/router/agentic/test_quality.py` and `tests/router/agentic/test_plan_parse.py`.

Any new class under `src/` (`SubTask`, `QualityVerdict`, `QualityChecker`)
goes through the repo's `gating-new-classes` skill before it is written.

## Tests (TDD, fakes only, call counts asserted)

- **Parser:**
  - `[after N]` becomes a dependency;
  - forward/self/unknown references are dropped;
  - no markers gives independent steps;
  - dedupe still works.
- **Verdict parser:**
  - a clean PASS;
  - a FAIL with weak steps and follow-ups (including `[after N]`);
  - unparseable text gives `unparsed`, which is never treated as a pass;
  - a score below `pass_score` with PASS counts as a fail.
- **Dependency context:** step 2 `[after 1]` receives step 1's output, and a
  failed dependency's error text is not passed on.
- **Late routing:** a fake router that favours `coder` when "fn " appears in
  the text. Step 1's output contains code, so dependent step 2 is routed to
  `coder` at run time, although plan time picked another model.
- **Escalation:** the fake checker fails with `WEAK_STEPS: 2`. Step 2 re-runs
  on a stronger, untried model, and step 3 (`[after 2]`) re-runs. Step 1 is
  not re-run (counted calls).
- **Follow-ups:** the checker returns 1 follow-up. Round 2 has one extra step,
  run with its declared dependency.
- **Bounds:**
  - with `max_replans=2` and a checker that always fails, there are exactly
    3 rounds;
  - the best-scoring round's answer is returned and `quality_check.passed`
    is False;
  - the fan-out cap is respected.
- **Unchanged:**
  - with `route_late=False` and no checker, existing plan tests pass as they
    are;
  - `orchestrator: recursive` tests pass untouched.

## Verification

1. `pytest tests/router/agentic tests/router/execution tests/e2e -q`, then `pytest -q`.
2. **Live run on local Ollama:** the quicksort prompt with `orchestrator: plan`,
   the routed decomposer and synthesizer, and the checker at min_quality 0.8,
   using the scratchpad runner extended for plan mode. Report per round:
   - the steps with their dependencies, the planned → routed model, and
     escalations;
   - the checker's verdict, score, weak steps and follow-ups;
   - call counts and wall time.
   Also report the stage metrics:
   - the planned ≠ routed rate;
   - escalations and whether each fixed its step's issue in the next verdict;
   - check scores per round;
   - follow-ups added and executed.

   Acceptance is structural:
   - the complexity step receives the Rust step's output;
   - at most 3 rounds;
   - every escalated step moves to a different model;
   - the final result has `quality_check` filled in.
3. **Caveat to report:** local 1.5-7B judges are unreliable. Judging the
   *quality* of the check needs a stronger model, e.g. one
   `clients: anthropic` run, which needs an API key and the user's OK.

## Status

Design approved 2026-10-07. Implementation plan:
`docs/superpowers/plans/2026-10-07-adaptive-plan-orchestrator.md`.
