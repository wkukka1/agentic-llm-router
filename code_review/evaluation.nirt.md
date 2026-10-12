# Code review: evaluation.nirt

**Files reviewed:** 1 (277 lines): routing.py
**Summary:** 4 findings: 3 correctness, 1 design · 2 confirmed, 2 plausible

**Diff scope note:** the branch diff here is 4 lines, all docstring/type-hint text
updating `Router` -> `RouterModel` (a `TYPE_CHECKING`-only import) to match the class
rename already landed in `router.routing.base`. Zero runtime behavior change. Per the
task's full-file-depth instructions, findings below are from reviewing the whole file,
not the diff hunks. Checked for recurrence of the known prior bug ("`cold_start_eval`
ignores `query_features`/`query_pathway`") -- not applicable, that function lives in
`evaluation/nirt/coldstart_eval.py`, not this file; no analogous pattern found here.

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| EN-01 | `src/evaluation/nirt/routing.py:130-140` | correctness | high | confirmed |
| EN-02 | `src/evaluation/nirt/routing.py:106-134` | design | medium | confirmed |
| EN-03 | `src/evaluation/nirt/routing.py:64-74` | correctness | low | confirmed |
| EN-04 | `src/evaluation/nirt/routing.py:188-193` | correctness | low | plausible |

---

## src/evaluation/nirt/routing.py

### EN-01: `oracle_choice`'s canonical tie-break can pick a below-max-score model when cost is NaN at every tied-max cell `:130-140`
**Category:** correctness · **Severity:** high · **Confidence:** confirmed

```python
130:    true = np.asarray(true, np.float64)
131:    at_max = true >= true.max(axis=1, keepdims=True) - tol
132:    masked_cost = np.where(at_max, np.asarray(cost, np.float64), np.inf)
133:    if model_ids is None:
134:        return masked_cost.argmin(axis=1)
...
138:    id_rank = np.argsort(np.argsort(ids, kind="stable"), kind="stable")
139:    id_key = np.broadcast_to(id_rank, true.shape)
140:    return np.lexsort((id_key, masked_cost), axis=-1)[:, 0]
```

**What breaks:** if the cost is missing (`NaN`) for every model that attains a query's
max score, but a strictly worse-scoring model has a real, finite cost, the canonical
(`model_ids` given) branch returns that **worse-scoring** model as the "oracle" --
silently violating the function's own contract ("the oracle model: the max observed
score, ties broken by cost"). Reproduced directly:

```
true = [[0.9, 0.9, 0.5]]           # models 0,1 tied for max; model 2 is worse
cost = [[nan, nan, 0.01]]           # cost missing for both max-scoring models
oracle_choice(true, cost, model_ids=['m0','m1','m2'])  -> [2]  ('m2', the WORST model)
```

The bug is `np.sort`/`np.lexsort` treating `NaN` as **larger than `inf`**, so the two
`NaN`-cost max-scoring cells sort *after* the `inf`-cost (non-max) cell, and `[:, 0]`
of the lexsort picks the `inf` one. The legacy branch (`model_ids=None`,
`masked_cost.argmin`) does not have this failure mode for the same input (`argmin`
returns index 0, one of the genuinely tied-max models) -- so the "recommended, new
code should always use this" canonical path is actually the more fragile one here.

**Evidence:** reproduced with `.venv/Scripts/python.exe`; confirmed `np.sort([inf, nan]) == [inf, nan]` and `np.lexsort` orders `NaN` after `inf`. Neither `oracle_choice` nor
its callers (`oracle_labels`, `routing_evaluation`, `oracle_classifier_matrix` in
`evaluation/routing/oracle.py`) validate that `cost` is finite -- only `true` gets a
`_require_finite` check (see `ER-02` for the caller-side gap). A `cost_df` that lags
`true_df` in coverage (plausible whenever a new model has graded results before its
price is backfilled -- the module's own docstring calls out that cost is "the only
[signal] RouterBench carries") reaches this path directly.

**Direction:** treat a `NaN` cost at an `at_max` cell as "unknown, prefer it over
nothing but never let it look cheaper than a real cost" -- e.g. replace `NaN` costs
with `+inf` (not leave them as `NaN`) before the `lexsort`/`argmin`, so ties only ever
resolve among models with *known* costs, falling back to `id_key` alone when no
tied-max model has a real cost. Add a regression test with `NaN` cost at all
tied-max cells.

---

### EN-02: `oracle_choice`'s safe tie-break is opt-in; the default silently reproduces the tie-break-by-position bug class `:106-134`
**Category:** design · **Severity:** medium · **Confidence:** confirmed

```python
106:def oracle_choice(
107:    true: np.ndarray, cost: np.ndarray, *, tol: float = 1e-9,
108:    model_ids: Optional[Sequence[str]] = None,
109:) -> np.ndarray:
...
117:    * ``model_ids`` given -- the **canonical** rule: max score -> min cost ->
118:      lexicographically smallest ``model_id``. Independent of column order; new
119:      code should always pass ``model_ids``.
120:    * ``model_ids=None`` -- **legacy compatibility**: max score -> min cost -> first
121:      column. Kept so ``routing_report`` and the pool-expansion battery reproduce
122:      their historical numbers.
...
133:    if model_ids is None:
134:        return masked_cost.argmin(axis=1)
```

**What breaks:** nothing today -- every current caller either deliberately omits
`model_ids` (the two call sites the docstring names) or passes it correctly. But the
signature makes the *unsafe* behavior (residual ties broken by raw column position --
exactly the ER-01..06 bug class the task asked to check for recurrence of) the
silent default, and the *safe* behavior opt-in. Any new caller who writes
`oracle_choice(true, cost)` -- the natural two-positional-arg call, matching every
existing call site's happy path -- gets the legacy, order-dependent tie-break with no
warning, `DeprecationWarning`, or lint signal.

**Evidence:** `grep` across the repo shows exactly two intentional legacy callers
(`routing_report` at `:164` and `evaluation.pool_expansion.battery:334`, both cited by
name in this docstring) and several canonical callers that already pass `model_ids`
(`evaluation/routing/oracle.py:191`, `:286`, `:586`). The design is currently correct
by convention, not by the type system or a runtime guard.

**Direction:** either make `model_ids` a required positional/keyword argument and give
the two legacy call sites an explicit `oracle_choice(..., model_ids=None)` opt-out (so
the risky choice is visibly deliberate at the call site, not the silent default), or
emit a one-time warning when `model_ids` is omitted so a future caller notices they're
on the legacy path.

---

### EN-03: `train_quality`'s `metric` argument silently accepts unrecognized values as `"quality"` `:64-74`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
64:def train_quality(data, models: list[str], metric: str = "quality") -> dict[str, float]:
...
71:    y = obs["target"].to_numpy(np.float64)
72:    if metric == "accuracy":
73:        y = (y >= 0.5).astype(np.float64)
74:    return pd.Series(y).groupby(obs["model_id"].to_numpy()).mean().to_dict()
```

**What breaks:** only `"accuracy"` is checked; any other string (including a typo like
`"Accuracy"` or `"acc"`) silently falls through to the `"quality"` behavior instead of
raising. A caller who mistypes the metric name gets a plausible-looking but wrong
"best fixed model" baseline in `routing_report`, with no error to catch it.

**Evidence:** no validation anywhere in the function; only two call sites in the repo
currently use the literal strings `"quality"`/`"accuracy"` so it hasn't manifested,
but the function is public API (`scripts/data/select_anchor_model.py`,
`scripts/nirt/ab_orientation.py`, `scripts/nirt/shrinkage_eval.py` all import it).

**Direction:** `raise ValueError(f"metric must be 'quality' or 'accuracy', got {metric!r}")` for anything else.

---

### EN-04: `routing_report`'s `cost_savings_vs_ref` divides by the reference model's mean cost with no zero guard `:188-193`
**Category:** correctness · **Severity:** low · **Confidence:** plausible

```python
188:    ref_row = df[df["policy"] == f"fixed: {reference}"]
189:    oracle_q = df.iloc[0]["quality"]
190:    if not ref_row.empty:
191:        rc, ra = float(ref_row["cost"].iloc[0]), float(ref_row["accuracy"].iloc[0])
192:        df["cost_savings_vs_ref"] = 1.0 - df["cost"] / rc
193:        df["d_accuracy_vs_ref"] = df["accuracy"] - ra
```

**What breaks:** if the `reference` model's mean cost over the evaluated split is
exactly `0.0` (e.g. a free/local model used as reference, or a synthetic
all-zero-cost test fixture), `df["cost"] / rc` produces `inf`/`-inf`/`NaN` for every
row instead of a clear error, silently corrupting `cost_savings_vs_ref` for the whole
report.

**Evidence:** the guard at `:190` only checks `ref_row.empty` (reference present in
`model_ids`), not `rc != 0`; `reference` defaults to `"gpt-4-1106-preview"` (non-zero
in practice) so this hasn't been hit by the default path, but the function accepts an
arbitrary `reference` string.

**Direction:** guard `rc > 0` alongside the existing `not ref_row.empty` check, or emit
`NaN`/skip the column with a comment rather than let the division silently propagate.
