# Cross-cutting review: duplication across packages

These findings span two or more subpackages, so no single per-package doc owns them. They merge the raw notes from reviewers A2, A3, A5, A6 and A8 with duplication found while finishing the review. IDs in parentheses are the original raw IDs.

**Summary:** 14 findings: 1 correctness, 13 design · 14 confirmed

| ID | Topic | Spans | Severity |
|----|-------|-------|----------|
| XD-01 | Two `effective_rank` definitions | training.nirt, training.nirt.baseline | low |
| XD-02 | Two oracle tie-break implementations | evaluation.nirt, evaluation.routing | low |
| XD-03 | Hard-coded family keyword maps | evaluation.nirt, training.data, training.models | low |
| XD-04 | RouterBench wide-pickle parsing copied | evaluation.data, training.data | low |
| XD-05 | "Read or build the observation table" in several places | router.data, training.data, training.retrieval, training.taxonomy | low |
| XD-06 | Routing regret implemented three times | training.nirt, evaluation.nirt, evaluation.routing | low |
| XD-07 | Cost-aware utility defined in three places | router, router.nirt, evaluation.routing | low |
| XD-08 | `compare_routers` duplicates `Router.aligned_scores` | evaluation.routing, router.routing | low (correctness) |
| XD-09 | `pairwise_logit_diff` re-implements `NIRTModel.forward` | training.nirt, router.nirt | low |
| XD-10 | Reliability binning duplicated | training.nirt, training.nirt.baseline | low |
| XD-11 | Repo-root path resolution duplicated | training, router, training.nirt | low |
| XD-12 | Two `LLMClient` / `LLMResponse` stacks | router.agentic, router.llm | low |
| XD-13 | Parallel task/result record hierarchies | router.agentic, router.execution | low |
| XD-14 | `ExecutionTrace` copies `RoutingDecision` fields | router.tracing, router | low |

---

### XD-01: Two `effective_rank` functions with different definitions (A2-X7, TB-X1)
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/training/nirt/metrics.py:161-178`, `src/training/nirt/baseline/diagnostics.py:55-62`

```python
161:def effective_rank(theta: np.ndarray) -> float:                        # participation ratio of Cov eigenvalues
55:def effective_rank(singular_values, eps: float = 1e-12) -> float:       # exp(entropy) of singular values
```

**What breaks:** One name computes two different quantities from two different inputs:
- **`metrics.effective_rank`:** the participation ratio of θ's covariance eigenvalues. It drives the P7d collapse threshold of 1.3 and returns NaN below 2 rows.
- **`diagnostics.effective_rank`:** the Roy–Vetterli entropy rank of singular values. It returns 0.0 on a collapsed θ.

Both appear in reports as "theta effective rank", and `compare_response_models` / route-compare tables put them side by side. Passing θ to the entropy version, or singular values to the participation-ratio version, runs without error and returns nonsense.
**Evidence:** Grep for `def effective_rank` over `src scripts` finds exactly these two.
**Direction:** Rename them (`participation_ratio`, `entropy_rank`), or keep one definition for both arms.

---

### XD-02: The oracle's cost tie-break is implemented twice and disagrees on equal cost
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/evaluation/nirt/routing.py:94-104`, `src/evaluation/routing/oracle.py:138-146`, `src/evaluation/routing/oracle.py:223`

```python
103:    at_max = true >= true.max(axis=1, keepdims=True) - tol
104:    return np.where(at_max, cost, np.inf).argmin(axis=1)        # routing.py: cost, then column order
144:        best = cands[np.lexsort(([models[c] for c in cands], tiebreak[i, cands]))[0]]   # oracle.py: cost, then model_id
```

**What breaks:** `oracle_choice` breaks remaining ties (equal score *and* equal cost, which `cost=None` makes universal) by column order. `oracle_labels` breaks them by lexicographic `model_id`. `routing_evaluation` computes `oracle_hit_rate` with `oracle_choice` (`:223`), while the label table's `oracle_model_id` comes from `oracle_labels`. For the same data, the per-query "oracle model" in the label table can differ from the one the hit rate was scored against, whenever pool order isn't alphabetical.
**Evidence:** Read both. `tests/evaluation/routing/test_routing_eval.py` covers each function separately.
**Direction:** Keep one `oracle_choice(true, cost, model_ids)` with the documented cost-then-id rule, and have `oracle_labels` call it.
**Status (2026-09-14): fixed.** `oracle_choice(..., model_ids=)` implements the canonical cost-then-id rule, and every call site in `evaluation/routing/oracle.py` passes `model_ids`. `model_ids=None` remains as a documented legacy mode (column-order tie-break) for `routing_report` and the pool-expansion battery.

---

### XD-03: Benchmark-family keyword maps are hard-coded outside `training.data.families`
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/evaluation/nirt/evaluate.py:249-261`, `src/training/data/families.py:15-26`, `src/evaluation/nirt/ood.py:23`, `src/training/models/profiles.py:31-32`

```python
251:    fam_kw = {"math": ["gsm", "math"], "code": ["mbpp", "humaneval", "code"],
252:              "knowledge": ["mmlu", "arc"], "reasoning": ["hellaswag", "winogrande"],
```

**What breaks:** `_family_breakdown` keeps a private keyword map that diverges from `profiles.task_families`:
- **Keywords:** `"arc"` matches any dataset containing "arc", and there are no `piqa`, `bbh`, `ceval` or `cmmlu` entries.
- **Family names:** it says `chinese` where the config says `language_zh`.
- **Overlap:** a dataset can match several families, since the masks aren't exclusive.

Cold-start per-family numbers are therefore not comparable with profile, taxonomy or OOD families. Separately, `ood.py` imports the underscore aliases `_family_of` / `_task_family_map` from `training.models.profiles`, which re-exports them from `training.data.families`, rather than the public module. TD-05's substring-order bug affects every consumer of the shared map.
**Evidence:** Read all four spans.
**Direction:** Use `training.data.families.family_labels` in `_family_breakdown`, and import from `training.data.families` in `ood.py`.

---

### XD-04: RouterBench wide-pickle parsing is copied into the judge pipeline
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/evaluation/data/judge_responses.py:48-60`, `src/training/data/loaders.py:109-140`

```python
48:    model_cols = [
49:        c for c in wide.columns
50:        if c not in _META_COLS
51:        and "|" not in c
52:        and f"{c}|total_cost" in wide.columns
```

**What breaks:** Model-column detection, the meta-column set and `query_id` reconstruction are re-typed from `_melt_routerbench`. The module docstring says the ids are "byte-identical" because it uses "the EXACT SAME two calls". That holds only while both copies stay in sync, and a change on one side (a new meta column, multishot ids) silently breaks the 1:1 join. EV-02 is already a divergence: the loader collapses duplicates, and the copy doesn't.
**Evidence:** Diffed the two blocks.
**Direction:** Expose `routerbench_model_columns(wide)` and `routerbench_query_ids(wide, shot)` from `training.data.loaders`, and call them from both places.

---

### XD-05: "Read `nirt_observations.parquet` or build it" is re-implemented in many places
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/router/data/nirt.py:229-236`, `src/training/data/nirt.py:123-128`, `src/training/data/facade.py:318-321`, `src/training/retrieval/knn_impute.py:50-58`, `src/training/retrieval/warmup.py:43`, `src/training/retrieval/query_bank.py:36-39`, `src/training/taxonomy/clustering.py:125-131`

```python
319:        if path.exists() and not build_kw:        # facade.py
126:            pd.read_parquet(path) if path.exists()  # training/data/nirt.py
130:    col = "nirt_observations.parquet" if p.exists() else "queries.parquet"   # clustering.py
```

**What breaks:** Seven call sites hard-code the file path, and each has its own fallback when the file is missing:
- **Raise:** `router/data/nirt.py`, `knn_impute`.
- **Build without saving:** `training/data/nirt.py`, the facade.
- **Silently widen to all queries:** `query_bank`, `clustering`.
- **Crash on `read_parquet`:** `warmup`.

TD-08, TRV-03 and TX-02 are symptoms of these inconsistencies.
**Evidence:** Grep for `nirt_observations.parquet` across `src`.
**Direction:** One `training.data.nirt.observations(cfg, *, build=False, **build_kw)` accessor used everywhere on the training side (serving keeps its explicit `observations=` argument).

---

### XD-06: Routing regret is implemented three times (A2-X5)
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/training/nirt/train.py:227-241`, `src/evaluation/nirt/evaluate.py:41`, `src/evaluation/routing/oracle.py:225`

```python
239:        reg += float(yy.max() - yy[int(pp.argmax())])
```

**What breaks:** Training early-stops on `_val_regret`, a long-format per-group loop with no tie handling. Evaluation reports `evaluate.py:41` (dense matrix) and `oracle.py:225` (tie-aware, clipped at 0). The `val_regret` a run selects on can differ from the regret evaluation reports for the same predictions, for example on ties or on queries with fewer than 2 models. `training` can't import `evaluation`, which forced the copy.
**Evidence:** Grep for `regret` across `src/evaluation` and `src/training`.
**Direction:** Put one dense-matrix regret function in `router.nirt` (importable by both) and use it everywhere.

---

### XD-07: The cost-aware utility `q − λ·C` is defined in three places despite a "single definition" rule (A5-X2)
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/router/constraints.py:17-36`, `src/router/nirt/routing_decision.py:38-39`, `src/evaluation/routing/oracle.py:258`, `src/evaluation/routing/oracle.py:321`

```python
19:    """Weights the four terms a routing utility trades off. ``utility()`` is
20:    the one place ``u(m, i, lambda)`` is defined -- offline signal analysis
```

**What breaks:** `OptimizationObjective.utility` says evaluation must import it. Instead:
- `oracle.py` re-implements `true - lam * C` twice, without `routing_decision`'s non-finite masking.
- `routing_decision` has a third copy.
- `evaluation.nirt.routing.route` uses a fourth, different formula (XA-02).

If the objective gains its latency or risk terms, the offline oracle silently stops matching it.
**Evidence:** Grep for `.utility(` finds no caller in `evaluation/`.
**Direction:** Expose one vectorised `utility(pred, lam, costs)` and use it from all of them.

---

### XD-08: `compare_routers` duplicates `Router.aligned_scores` and diverges on id types (A5-X3)
**Category:** correctness · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/evaluation/routing/oracle.py:513-515`, `src/router/routing/base.py:247-250`, `src/evaluation/routing/oracle.py:73`

```python
514:        mat = r.predict_scores(query_ids).reindex(index=query_ids, columns=model_ids)
```

**What breaks:** `evaluate_router` goes through `aligned_scores`, which stringifies ids. `compare_routers` calls `predict_scores` with raw labels instead. With non-string query ids, routers that stringify internally return `str` indexes, the reindex produces all-NaN rows, and every query falls back to column 0 (RR-03). `evaluate_router` on the same inputs works. Two routers with the same `name` also overwrite each other in `preds`.
**Evidence:** Traced both functions. All tests use string ids and distinct names.
**Direction:** Use `r.aligned_scores(query_ids)` and reject duplicate router names.

---

### XD-09: `pairwise_logit_diff` re-implements `NIRTModel.forward`'s score (A2-X2)
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/training/nirt/pairwise.py:95-109`, `src/router/nirt/model.py:310-325`

```python
98:    if model.difficulty == "vector":
99:        z_a = (a_a * (theta - b_a)).sum(-1)
```

**What breaks:** The scalar and vector difficulty branches and the interaction residual are copied line for line. A future change to the forward silently desynchronises the auxiliary battle objective from the correctness objective. XA-04 is already a divergence: centring differs between the two paths.
**Evidence:** Diffed `pairwise.py:98-108` with `model.py:318-324`.
**Direction:** Add a `NIRTModel.score(theta, a, b)` helper and use it from both paths.

---

### XD-10: `_binned` re-implements `reliability_curve`'s binning (TB-X2)
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/training/nirt/baseline/continuous_eval.py:192-201`, `src/training/nirt/metrics.py:70-94`

```python
193:    edges = np.linspace(0, 1, n_bins + 1)
194:    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)
```

**What breaks:** It uses the same edges, digitize and clip steps with different output keys and no ECE. The ZOIB boundary calibration therefore reports no ECE, and the two binning implementations can drift apart.
**Evidence:** Read both. `_binned` is also imported by `tests/training/nirt/baseline/test_phase2_evaluation.py:10`.
**Direction:** Call `reliability_curve` and map its bins.

---

### XD-11: Repo-root path resolution is duplicated (A2-X6)
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/training/cli.py:41-44`, `src/router/config.py:70-72`, `src/training/nirt/train.py:638-647`

```python
43:    p = Path(path)
44:    return p if p.is_absolute() else Path(root or REPO_ROOT) / p
```

**What breaks:** There are three copies of "absolute, else join to root". Nothing is wrong today; the risk is drift. EP-02 shows a fourth variant that hard-codes paths instead of resolving them.
**Evidence:** Read all three spans.
**Direction:** Keep one `router.config.resolve_path(path, root=None)`.

---

### XD-12: Two incompatible `LLMClient` / `LLMResponse` stacks (A6-X01, A8-X1)
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/router/agentic/llm_clients.py:36-51`, `src/router/agentic/llm_clients.py:114-124`, `src/router/llm/client.py:22-43`, `docs/architecture.md:231-237`

```python
# router/llm/client.py
24:    content: str
27:    cost: float = 0.0
# router/agentic/llm_clients.py
38:    text: str
40:    cost: Optional[float] = None
```

**What breaks:** The two classes share names and nothing else:
- **Text field:** `content` versus `text`.
- **Cost default:** `0.0` versus `None`, so "unknown cost" means different things.
- **Methods:** `complete` / `stream` versus `invoke`.
- **Model id:** absent in one, required in the other.
- **Tokens:** first-class fields versus buried in `meta["usage_metadata"]`.

Token and latency accounting exists only in the unwired `router.llm` stack, while the live path records no cost (RA-03). Importing the wrong `LLMResponse` fails only at runtime with `AttributeError`. `router/llm` is known scaffolding, so this is a design note, not dead code.
**Evidence:** Grep finds two classes named `LLMResponse` and two named `LLMClient`. `docs/architecture.md:231-237` already warns not to conflate them.
**Direction:** Settle on one response schema with token, cost and latency fields, and make the agentic clients produce it.

---

### XD-13: Parallel task/result record hierarchies (A8-X4)
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/router/execution/results.py:17-25`, `src/router/execution/task.py:30-55`, `src/router/agentic/result.py:20-21,40`

```python
# execution/results.py
20:    actual_cost: float = 0.0
# agentic/result.py
21:    cost: Optional[float] = None
```

**What breaks:** Two task-tree models record the same facts (prompt, answer, cost, depth, children) with different field names and different cost nullability. A cost roll-up written for one silently mishandles the other, because `None` and `0.0` sum differently.
**Evidence:** Read both. `execution/results.py:5-8` acknowledges the overlap.
**Direction:** When wiring `router.execution`, make `SubCall` a view over `AgentTask` + `ExecutionResult`, or the reverse.

---

### XD-14: `ExecutionTrace` duplicates provenance fields already on `RoutingDecision` (A6-X04)
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/router/tracing/traces.py:22-35`, `src/router/decision.py:37-45`

```python
28:    artifact_id: str
29:    policy_version: str
32:    decision: "RoutingDecision"
```

**What breaks:** The trace stores `artifact_id`, `policy_version` and `request_id` next to a `decision` that carries the same fields. That gives two sources of truth for "which artifact decided this". This is latent, because nothing constructs either class today.
**Evidence:** Read both dataclasses. Grep finds no `ExecutionTrace(` construction.
**Direction:** Derive the trace fields from `decision` and `request` instead of storing copies.
