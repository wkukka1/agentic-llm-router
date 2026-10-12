# Code review: evaluation.routing

**Files reviewed:** 1 (674 lines): oracle.py
**Summary:** 4 findings: 3 correctness, 1 performance/reuse · 2 confirmed, 2 plausible

**Diff scope note:** the branch diff here is 10 lines, all docstring/type-hint text
updating `Router` -> `RouterModel` (a `TYPE_CHECKING`-only import) to match the class
rename already landed in `router.routing.base`. Zero runtime behavior change. Checked
the known prior finding (oracle-classifier baseline labeling ties by column order,
ER-01..06 / XD-02, reportedly fixed 2026-09-14): confirmed **still fixed** and **not
reintroduced** by this diff -- `oracle_classifier_matrix` (`:586`) still calls
`oracle_choice(tr_true, tr_cost, model_ids=model_ids)` with `model_ids` passed, i.e.
the canonical tie-break, and the diff doesn't touch that line or its neighbors.
However, see ER-01 below: a *different* function in this same file (`compare_routing_strategies`'s hard-oracle row) has its own hand-rolled tie-break that
does not go through the canonical `oracle_choice` path and can fall back to
column-order ties.

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| ER-01 | `src/evaluation/routing/oracle.py:478-485` | correctness | medium | plausible |
| ER-02 | `src/evaluation/routing/oracle.py:160-193` | correctness | medium | plausible |
| ER-03 | `src/evaluation/routing/oracle.py:616-652` | correctness | low | confirmed |
| ER-04 | `src/evaluation/routing/oracle.py:484` | reuse | low | confirmed |

---

## src/evaluation/routing/oracle.py

### ER-01: the "hard oracle (upper bound)" row's tie-break is a hand-rolled cost perturbation that reverts to column order on residual ties, diverging from the canonical `oracle_choice` rule `:478-485`
**Category:** correctness · **Severity:** medium · **Confidence:** plausible

```python
478:    if include_hard_oracle:
479:        # route on the truth; break exact-quality ties toward the cheaper model
480:        # so the upper bound is also the cheapest way to reach max quality.
481:        tb = true.astype(np.float64).copy()
482:        if cost is not None:
483:            c = np.asarray(cost, np.float64)
484:            tb = tb - 1e-9 * (c - c.min()) / (np.ptp(c) + 1e-12)
485:        _run("hard oracle (upper bound)", tb, 0.0)
```

**What breaks:** `_run` feeds `tb` as `pred` into `routing_evaluation`, whose
selection is `routing_decision(tb, lam=0, ...)` -- plain `argmax`, which (per
`routing_decision`'s own docstring) breaks exact ties by **lowest column index**,
deliberately. So this block's cost perturbation only resolves the *score* tie toward
lower cost; if two or more models are **also** tied on the (per-cell) `cost` value
after that perturbation is applied identically to both, the residual tie falls back to
column order -- not `oracle_choice`'s documented "max score -> min cost ->
lexicographically smallest model_id" rule used everywhere else in this module
(`oracle_labels`, and `routing_evaluation`'s own `o_idx` at `:286-288`, which the
"hard oracle" row's `oracle_hit_rate` is compared against at `:292`). Two concrete
consequences: (1) the "hard oracle (upper bound)" summary row is not guaranteed
`oracle_hit_rate == 1.0` even though it is supposed to be the upper bound reached by
routing on ground truth; (2) its reported `mean_selected_cost` can differ from what
the canonical oracle would have reported for the same query set.

**Evidence:** traced `_run` -> `routing_evaluation` -> `routing_decision` ->
`cost_aware_utility` (`src/router/nirt/routing_decision.py:48-68`), whose docstring
states plainly: "Exact ties go to the lowest column index... deliberately independent
of `evaluation.routing.oracle`'s tie-inclusive oracle labels." `docs/routing_evaluation.md:69-70` confirms exact score ties are the norm for this domain
("heavily-tied targets (binary MC correctness)"), which is the precondition for this
path to matter; a *residual* cost tie on top of that is less certain for continuous
per-query costs (lower likelihood, hence "plausible" not "confirmed"), but plausible
for models sharing identical per-token pricing or in small/synthetic eval sets.

**Direction:** build the "hard oracle" selection from `oracle_choice(true, cost,
model_ids=model_ids)` directly (it already returns the canonical per-query pick)
instead of re-deriving an equivalent ranking through a perturb-then-argmax trick that
silently drops the model_id tie-break on the last step.

---

### ER-02: `oracle_labels` (and its callers) never validate that `cost` is finite before handing it to `oracle_choice`, so a missing per-cell cost at a tied-max query can silently corrupt `oracle_model_id` `:160-193`
**Category:** correctness · **Severity:** medium · **Confidence:** plausible

```python
177:    models = list(true_df.columns)
178:    query_ids = list(true_df.index)
179:    true = true_df.to_numpy(np.float64)
180:    _require_finite(true, "oracle_labels", query_ids=query_ids, model_ids=models)
181:    cost = (
182:        cost_df.reindex(index=true_df.index, columns=models).to_numpy(np.float64)
183:        if cost_df is not None else None
184:    )
...
190:    # deterministic single oracle model: max score -> cheapest -> smallest id
191:    oracle_col = oracle_choice(
192:        true, cost if cost is not None else np.zeros_like(true), tol=tol, model_ids=models,
193:    )
```

**What breaks:** this is the caller-side half of `EN-01` (in
`src/evaluation/nirt/routing.py`, `oracle_choice`'s canonical branch): a `NaN` cost at
every model tied for a query's max score makes `oracle_choice` return a
strictly-worse-scoring model as `oracle_model_id`, with no exception raised anywhere
in this call chain. `_require_finite` is called on `true` (`:180`) -- explicitly
documented as needed because "the oracle... is only defined on a dense matrix" -- but
the exact same argument applies to `cost` once it's used as a tie-break key, and
`cost` gets no equivalent check here, in `routing_evaluation` (`:286-288`, same
pattern), or inside `oracle_choice` itself.

**Evidence:** `oracle_choice` reproduced independently (see `EN-01`); this function
and `routing_evaluation` are the two production call sites that reindex a
possibly-partial `cost_df` (`cost_df.reindex(...)`, which introduces `NaN` for any
`(query, model)` combination missing from `cost_df`) directly into `oracle_choice`
without a finiteness check.

**Direction:** either extend `_require_finite` (or a variant) to also validate `cost`
at the cells that matter (the `at_max` cells, once `oracle_choice`'s own fix from
EN-01 lands), or validate `cost_df` density up front the same way `true_df` already
is, with the same clear error message pattern.

---

### ER-03: `compare_routers` silently drops a router's own models that aren't in `true_df`'s columns, where `evaluate_router` raises for the same situation `:616-652`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
84:    model_ids = list(router.model_ids)
85:    # reindexing to a model the outcome matrix lacks would add an all-NaN column
86:    # and silently turn every regret / hit rate into NaN
87:    for name, frame in (("true_df", true_df), ("cost_df", cost_df)):
88:        missing = sorted(set(model_ids) - set(frame.columns)) if frame is not None else []
89:        if missing:
90:            raise ValueError(
91:                f"{name} has no column for router pool models {missing}; "
92:                "restrict the router's pool or supply their outcomes"
93:            )
```
```python
642:    model_ids = list(true_df.columns)
643:    query_ids = list(true_df.index)
644:    preds: dict[str, np.ndarray] = {}
645:    for r in routers:
646:        if r.name in preds:
647:            raise ValueError(
648:                f"compare_routers: duplicate router name {r.name!r}; give each "
649:                "router passed in a distinct .name"
650:            )
651:        mat = r.aligned_scores(query_ids).reindex(columns=model_ids)
652:        preds[r.name] = mat.to_numpy(float)
```

**What breaks:** `evaluate_router` treats "router's pool has a model the outcome
matrix lacks" as a hard error (`:87-93`, with a comment explaining exactly why --
silent all-`NaN` corruption of every metric). `compare_routers` -- which shares the
same underlying risk, since it also feeds router scores into the same
`routing_evaluation` machinery -- instead narrows `model_ids` to `true_df.columns`
(`:642`) and reindexes each router's own scores down to that set (`:651`), which
*drops* any of a router's own models that aren't covered by `true_df` rather than
raising or NaN-ing them. A router mis-configured with a pool that doesn't match
`true_df` is silently evaluated on a subset of its own candidate set with no warning,
while the same misconfiguration hits a loud, actionable error in `evaluate_router`.

**Evidence:** direct comparison of the two functions' validation logic; both are
public entry points into the same routing-evaluation pipeline and both are documented
as "compare against ground truth" evaluation code (module docstring `:1-9`).

**Direction:** either apply the same missing-model check per router in
`compare_routers` (skip with a warning, or raise, consistent with `evaluate_router`),
or explicitly document in `compare_routers`'s docstring that only the models common to
all inputs are compared, so the silent-narrowing behavior is a documented contract
rather than an inconsistency between two sibling functions.

---

### ER-04: the hard-oracle tie-break perturbation hardcodes `1e-9` instead of reusing the module's `_TOL` constant `:484`
**Category:** reuse · **Severity:** low · **Confidence:** confirmed

```python
57:_TOL = 1e-9
...
484:            tb = tb - 1e-9 * (c - c.min()) / (np.ptp(c) + 1e-12)
```

**What breaks:** nothing today -- the two values are equal -- but `_TOL` is used
throughout this file (`_require_finite`'s implicit tie tolerance via `oracle_choice`'s
default, `oracle_labels`'s `flag`, `routing_evaluation`'s `hit_any_best`,
`per_query_table`'s `oracle_hit_any_best`) as *the* tolerance below which two scores
are considered tied. This perturbation's magnitude is meaningful only in relation to
that same tolerance (it must stay smaller than it, or it could flip a "genuine"
near-tie), yet it's a second, independent literal that would silently drift out of
sync with `_TOL` if either is ever changed.

**Evidence:** `grep -n "1e-9" src/evaluation/routing/oracle.py` shows `_TOL` defined
once at `:57` and reused by name elsewhere in the file, except at `:484`.

**Direction:** `tb = tb - _TOL * (c - c.min()) / (np.ptp(c) + 1e-12)`.
