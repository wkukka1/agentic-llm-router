# Code review: router.execution

**Files reviewed:** 7 (367 lines) · **Context-only:** `src/router/agentic/llm_clients.py`, `src/router/agentic/router.py`, `src/router/agentic/result.py`, `src/router/constraints.py`, `src/router/router.py`
**Summary:** 14 findings: 8 correctness, 6 design, 0 performance · 11 confirmed, 3 suspected

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| RX-01 | `src/router/execution/forecast.py:62-66` | correctness | medium | confirmed |
| RX-02 | `src/router/execution/budget.py:48-57` | correctness | medium | confirmed |
| RX-03 | `src/router/execution/task.py:47-55` | correctness | medium | confirmed |
| RX-04 | `src/router/execution/orchestrator.py:80-92` | correctness | medium | confirmed |
| RX-05 | `src/router/execution/orchestrator.py:74-78` | correctness | medium | confirmed |
| RX-06 | `src/router/execution/__init__.py:1-4` | design | low | confirmed |
| RX-07 | `src/router/execution/forecast.py:67-71` | design | low | confirmed |
| RX-08 | `src/router/execution/orchestrator.py:67-72` | design | low | confirmed |
| RX-09 | `src/router/execution/task.py:48-54` | correctness | low | confirmed |
| RX-10 | `src/router/execution/plan.py:27-48` | design | low | confirmed |
| RX-11 | `src/router/execution/budget.py:17-23` | design | low | confirmed |
| RX-12 | `src/router/execution/budget.py:48-60` | correctness | low | suspected |
| RX-13 | `src/router/execution/orchestrator.py:53,67-72` | correctness | low | suspected |
| RX-14 | `src/router/execution/results.py:24` | design | low | suspected |

---

## src/router/execution/__init__.py

### RX-06: Execution layer is scaffolding, not wired into any live path `:1-4`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
1:  """Planning, budget, and orchestration scaffolding for the production router
2:  design. Distinct from :mod:`router.agentic`, which is the actually-wired,
```

**What breaks:** Nothing today. `Orchestrator`, `PlanForecaster`, `AgentTask`, and `BudgetLedger` are not reachable from the live agentic flow, so the correctness findings below are latent. They will surface once this layer gets wired in.
**Evidence:** Ran `grep -rn "router\.execution\|BudgetLedger\|PlanForecaster\|OrchestratorFactory\|aggregate_children" src scripts tests configs docs` (excluding the package itself). Only hits: `src/router/router.py:34,49` (a TYPE_CHECKING import and a stored-but-unused `orchestrator_factory` attribute) and `tests/router/routing/test_router_scaffolding.py:11,78,97` (tests for `BudgetLedger` and `ExecutionLimits` only). By design, per the package docstring.
**Direction:** Keep as-is until wiring. When you wire it, add tests for `aggregate_children`, `spawn`/`can_spawn`, and `score_plan`, which have no coverage today.

---

## src/router/execution/budget.py

### RX-02: NaN or negative amounts silently defeat the ledger's `spent + reserved <= total` invariant `:48-57`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
49:        amount = float(amount)
50:        if amount > self.remaining():
51:            return False
```

**What breaks:** `reserve(tid, float("nan"))`: `nan > remaining()` is False, so the NaN gets stored. After that, `reserved` and `remaining()` are both NaN, and every later `amount > nan` is False. Every future `reserve` returns True, so the budget is permanently unlimited. `Orchestrator.can_spawn` (`remaining() > 0` -> False) meanwhile refuses every spawn. A negative `amount` in `reserve` or `actual` in `commit` raises `remaining()` above `total`. A NaN estimate is plausible from a forecaster that multiplies a missing per-token price or token count.
**Evidence:** Traced through Python float semantics at lines 42-53 and 55-57. The tests at `tests/router/routing/test_router_scaffolding.py:77-100` use only positive finite amounts.
**Direction:** In `reserve` and `commit`, reject non-finite or negative amounts with a `ValueError`, or clamp them explicitly.

### RX-11: Three independent cost/latency caps with no reconciliation `:17-23`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
18: class ExecutionLimits:
19:     max_cost: float = float("inf")
20:     max_latency: float = float("inf")
```

**What breaks:** Three separate places hold a spend limit: `ExecutionLimits.max_cost`, `BudgetLedger.total`, and `RoutingConstraints.max_cost` / `max_latency` (`src/router/constraints.py:47-48`). Nothing ties them together. An `OrchestratorConfig` can carry `limits.max_cost=1.0` alongside `ledger.total=100.0`, and `can_spawn` consults only the ledger. `max_latency`, `max_fanout`, and `max_replans` are read nowhere.
**Evidence:** Ran `grep -rn "max_fanout\|max_replans\|max_latency\|max_cost" src scripts tests configs docs`. Only hits are the definitions, `constraints.py:43-48`, an unrelated `battery.py:113` local, and `test_router_scaffolding.py:106` (asserts `max_fanout > 0`).
**Direction:** Pick one authority: derive `ledger.total` from `limits.max_cost` at construction, or drop the duplicate field.

### RX-12: `reserve` is check-then-act, and `commit` accepts task ids that were never reserved `:48-60`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
50:        if amount > self.remaining():
51:            return False
52:        self._reservations[task_id] = self._reservations.get(task_id, 0.0) + amount
```

**What breaks:** The ledger is meant to be shared across a whole orchestration tree (`orchestrator.py:89`, `ledger=self.ledger`), and `max_fanout=8` implies parallel children. If children reserve from threads or async tasks, two of them can both pass line 50 before either writes line 52, overspending despite the "no double-spend" docstring. Separately, `commit` on a task id with no reservation just adds to `spent`, and a mistyped id leaves the real hold dangling forever.
**Evidence:** No concurrency exists today (unwired). Suspected because it depends on whether children will run concurrently. Maintainer question: will sibling tasks execute in parallel against one ledger?
**Direction:** Guard reserve/commit/release with a lock if execution will be concurrent. Consider having `commit` warn or raise on unknown ids.

---

## src/router/execution/forecast.py

### RX-01: `score_plan` re-implements the utility formula and drops the risk term `:62-66`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
62:        utility = (
63:            objective.quality_weight * quality
64:            - objective.cost_weight * cost.total_cost
```

**What breaks:** `OptimizationObjective.utility` (`src/router/constraints.py:29-36`) is documented as "the one place `u(m, i, lambda)` is defined" and subtracts `risk_weight * (1 - confidence)`. `score_plan` hand-copies the other three terms and omits risk, even though `CostForecast.confidence` is available. Under any objective with `risk_weight > 0`, plan admission utility is systematically higher than model routing utility for the same objective. That is the drift this method's own docstring says it prevents.
**Evidence:** Compared `forecast.py:62-66` with `constraints.py:30-36`. Ran `grep -rn "score_plan\|PlanForecaster" src tests scripts`: no callers and no tests. With the default `risk_weight=0.0` the two agree, which is why no test would catch it.
**Direction:** Build a `ModelScore`-like record, or factor out a shared helper, and call `objective.utility(...)` so the formula lives in one place. Map `cost.confidence` or a quality confidence onto the risk term.

### RX-07: `within_budget` is hard-coded True `:67-71`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
70:        return PlanScore(quality=quality, cost=cost.total_cost, latency=latency,
71:                         utility=utility, within_budget=True)
```

**What breaks:** Any admission controller that gates on `PlanScore.within_budget` will admit every plan, including ones exceeding `BudgetLedger.remaining()`. The code comment acknowledges this as a placeholder.
**Evidence:** `grep -rn "within_budget" src scripts tests docs` finds only this file.
**Direction:** Pass a `BudgetLedger` or `ExecutionLimits` into `score_plan`, or make `within_budget` `Optional[bool] = None` so an unchecked result can't be read as a pass.

---

## src/router/execution/orchestrator.py

### RX-04: `spawn` rebuilds the child with only the base-class constructor arguments `:80-92`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
86:        return type(self)(
87:            OrchestratorConfig(
88:                name=self.name, version=self.version, llm=self.llm,
```

**What breaks:** A concrete subclass whose `__init__` takes extra required keyword arguments (e.g. a decomposer or client registry) raises `TypeError` on `spawn`. One with optional extras silently gets defaults in the child, so the child behaves differently from its parent. `OrchestratorFactory.create(config, cls, **kwargs)` at lines 96-104 explicitly forwards arbitrary `**kwargs`, so subclasses with extra constructor arguments are an anticipated shape. The rebuild also doesn't bind `child` to the new orchestrator: its `task` stays `None`, and the child is never appended to the parent task's `children`.
**Evidence:** Traced lines 43-53 (the base signature takes only `config`, `router_tool`, `plan_forecaster`) against 86-92. No subclasses or tests exist (`grep -rn "Orchestrator)" src tests`), so nothing covers this.
**Direction:** Make spawning a hook that subclasses override (e.g. `_child_kwargs()`), or clone via `copy.copy(self)` and then reset per-task state. Record the parent/child link on `AgentTask.children`.

### RX-05: `can_spawn` enforces depth and "any budget left" only; fanout and allocated budget are ignored `:74-78`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
76:            child.depth < self.limits.max_depth
77:            and self.ledger.remaining() > 0
```

**What breaks:**
- `ExecutionLimits.max_fanout` is never checked, so a planner can spawn any number of children at depth 1.
- `remaining() > 0` passes with a remaining balance of 1e-9, so a child is admitted with effectively no budget.
- `child.allocated_budget` is never reserved against the ledger.
- `child.depth` is caller-supplied, and nothing checks that it is parent depth + 1 or that `child.root_task_id == ledger.root_task_id`. A child built with `depth=0` bypasses the depth cap entirely.

**Evidence:** Read lines 74-92. `grep -rn "max_fanout\|allocated_budget"` finds no enforcement anywhere (see RX-11).
**Direction:** Have `can_spawn` check `len(parent.children) < max_fanout` and `ledger.reserve(child.task_id, child.allocated_budget)`, and derive `child.depth` from the parent rather than trusting the caller.

### RX-08: `run` template skips the ledger, replanning, and status transitions `:67-72`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
71:        self.task = task
72:        return self.execute(self.plan(task))
```

**What breaks:** The module docstring promises "replan on failure", but `run` never calls `replan`, never consults `max_replans`, never reserves the plan's `estimated_cost` or commits `ExecutionResult.actual_cost`, and never moves `task.status` or `plan.status`. If `execute` raises, the task is left in `PENDING`. A subclass that relies on the default template gets none of the budget accounting the package exists to provide.
**Evidence:** `grep -rn "replan\|max_replans" src tests` finds only the abstract method and the dataclass field.
**Direction:** Put the reserve -> execute -> commit/release (plus bounded replan) sequence in the template with try/finally, so a subclass can't forget to release a hold.

### RX-13: Per-task state stored on the orchestrator instance `:53,67-72`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
53:        self.task: Optional["AgentTask"] = None
71:        self.task = task
```

**What breaks:** Calling `run` on one orchestrator instance for two tasks, whether concurrently or reentrantly from inside `execute`, overwrites `self.task`. Code in `execute`/`replan` that reads `self.task` then sees the wrong task.
**Evidence:** No implementations exist yet. Suspected; depends on whether instances will be reused. Maintainer question: is an `Orchestrator` one-per-task?
**Direction:** Pass the task through `plan`/`execute` explicitly, or document and enforce one-task-per-instance.

---

## src/router/execution/plan.py

### RX-10: Plan-level estimates are unsynchronized with steps, and step dependencies are unvalidated `:27-48`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
34:    estimated_cost: float = 0.0
43:    estimated_cost: float = 0.0
46:    utility: float = 0.0
```

**What breaks:**
- `Plan.estimated_cost` and `PlanStep.estimated_cost` are independent fields. Nothing enforces that the plan total equals the sum of its steps, so reserving `plan.estimated_cost` can under-hold.
- `PlanForecaster.score_plan` returns a separate `PlanScore` and never writes back to `plan.estimated_*` / `utility` / `status`, so those fields stay at 0.0/DRAFT unless every caller copies them by hand.
- `dependencies` can reference unknown step ids or form cycles, and nothing checks either.

**Evidence:** Read `plan.py:27-48` and `forecast.py:54-71`. `grep -rn "estimated_cost\|PlanStatus" src tests` finds no other users.
**Direction:** Make the plan totals derived properties, or recompute them in the forecaster. Validate the dependency DAG when a plan leaves DRAFT.

---

## src/router/execution/results.py

### RX-14: `success` defaults to True `:24`
**Category:** design · **Severity:** low · **Confidence:** suspected

```python
24:    success: bool = True
25:    error_code: Optional[str] = None
```

**What breaks:** An error path that builds `ExecutionResult(error_code="timeout")` without setting `success=False` reports success. `AgentTask.aggregate_children` (task.py:54) then reads it as a successful child.
**Evidence:** No constructors exist yet (`grep -rn "ExecutionResult(" src tests` finds only `task.py:48`). Suspected: a latent footgun, not a traced failure.
**Direction:** Derive `success` from `error_code is None`, or make `success` a required field.

---

## src/router/execution/task.py

### RX-03: `aggregate_children` reports success when children failed without a result, or when there are no children `:47-55`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
54:            success=all(c.result.success for c in self.children if c.result),
```

**What breaks:** Children whose `result is None` are silently dropped from every aggregate. That covers a child that crashed, was `TERMINATED`, or is still `PENDING`/`RUNNING`. A parent with three children, two of them crashed with `status=FAILED, result=None`, aggregates to `success=True` with the one survivor's output and cost. With zero children, `all([])` is True and output is `""`. The cost sum also under-reports spend from children that were charged but produced no result.
**Evidence:** Traced lines 47-55. `TaskStatus.FAILED` / `TERMINATED` (lines 25-27) are never consulted. `grep -rn "aggregate_children" src tests scripts` finds no tests.
**Direction:** Treat a child with `result is None` or a non-COMPLETED status as a failure (or raise if any child is still in progress), and decide explicitly what an empty child list means.

### RX-09: Latency is summed across children, and child `error_code` is dropped `:48-54`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
51:            actual_latency=sum(c.result.actual_latency for c in self.children if c.result),
```

**What breaks:** Sibling subtasks are meant to run under `max_fanout`, so their latencies are presumably parallel. Wall-clock latency for parallel siblings is roughly the max, not the sum, so an `ExecutionLimits.max_latency` check on the aggregate would reject trees that actually met their deadline. The aggregate `ExecutionResult` also never sets `error_code`, so the reason any child failed is lost at the parent.
**Evidence:** Read lines 43-55. The docstring acknowledges "straight sum/concat", but sum is the wrong reduction for latency under parallel fanout.
**Direction:** Reduce latency with max (or with a dependency-aware critical path from `PlanStep.dependencies`), and propagate the first or collected child `error_code`.

---

## Checked and ruled out
- `task.py:50-54` uses `if c.result` while line 47 uses `if c.result is not None`. Dataclass instances without `__bool__`/`__len__` are always truthy, so the two filters are equivalent.
- `budget.py:55-57` letting `commit` push `spent` above `total`: intended, since actual cost can exceed the estimate. `remaining()` going negative correctly blocks `can_spawn`.
- `budget.py:52` accumulating repeated reservations for one task id, with `commit`/`release` popping the whole amount: consistent, no leak.
- `orchestrator.py:50,89` sharing `self.tools` with the child: `__init__` copies it via `list(config.tools)`, so no aliasing.
- `forecast.py` / `orchestrator.py` TYPE_CHECKING imports (`router.constraints`, `router.tools.base.Tool`, `router.tools.router_tool.RouterTool`, `router.llm.profile`): all target modules and classes exist.
- The `budget.py:42-43` `reserved` property is O(#reservations) on every `remaining()` call. Negligible at `max_fanout=8`; not reported.
- Import-linter: `router.execution` imports only `router.*` modules; no violation.
