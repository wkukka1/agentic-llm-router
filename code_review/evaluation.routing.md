# Code review: evaluation.routing

**Files reviewed:** 2 (533 lines): `__init__.py`, `oracle.py` · **Context-only:** `src/router/nirt/routing_decision.py`, `src/router/routing/base.py`, `src/evaluation/nirt/routing.py`
**Summary:** 6 findings: 5 correctness, 0 design, 1 performance · 6 confirmed, 0 suspected

**Status (2026-09-14): ER-01..ER-06 fixed on `cleanup/consolidate-nirt`, together with XD-02.**
- **XD-02 / ER-01:** one oracle rule, `oracle_choice(..., model_ids=)`: max score → min cost → smallest `model_id`. It is used by `routing_evaluation`, `oracle_labels` and the baseline D labels, and `route_eval.py` passes the train cost.
- **ER-02 / ER-03:** a shared `_cost_aware_oracle` helper applies `eligible` to both the summary and the per-query table. Rows with no eligible candidate are excluded and counted (`n_queries_no_eligible` / `n_queries_evaluated_cost_aware`).
- **ER-04 / ER-05:** a missing pool column or a non-finite outcome raises `ValueError`.
- **ER-06:** `oracle_labels` is vectorised, keeping the exact group-anchor rank semantics.

Filed elsewhere, not repeated here:
- **XA-01:** the evaluated router uses a different cost vector than the served router (`oracle.py:217-220`).
- **XD-07:** the cost-aware utility is duplicated.
- **XD-08:** `compare_routers` duplicates `aligned_scores`.
- **XD-02:** oracle tie-breaking differs from `oracle_choice`.

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| ER-01 | `src/evaluation/routing/oracle.py:479-485` | correctness | medium | confirmed |
| ER-02 | `src/evaluation/routing/oracle.py:319-325` | correctness | low | confirmed |
| ER-03 | `src/evaluation/routing/oracle.py:258-275` | correctness | low | confirmed |
| ER-04 | `src/evaluation/routing/oracle.py:73-80` | correctness | low | confirmed |
| ER-05 | `src/evaluation/routing/oracle.py:134-145` | correctness | low | confirmed |
| ER-06 | `src/evaluation/routing/oracle.py:141-165` | performance | low | confirmed |

---

## src/evaluation/routing/oracle.py

### ER-01: The oracle classifier's labels break ties by column order, against the module's own tie policy `:479-485`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
479:    y_train = train_true_df.reindex(columns=model_ids).to_numpy(np.float64).argmax(axis=1)
480:    clf = LogisticRegression(max_iter=1000, random_state=seed)
```

**What breaks:** The module's tie policy (`:28-34`) forbids leaving the oracle to column order, and `oracle_choice` / `oracle_labels` break ties by cost. This baseline uses a plain `argmax`, so every tied row is labelled with the first tied model in pool order. RouterBench targets are heavily tied, with about 79% boundary mass and many rows where several models score 1.0. The classifier therefore largely learns to predict the earliest column. Baseline D's routing numbers reflect pool ordering rather than the query embedding, and they change when the pool is reordered.
**Evidence:** Read `:462-486` against `evaluation/nirt/routing.py:94-104` and `oracle.py:138-145`.
**Direction:** Label with `oracle_choice(train_true, train_cost)`, or fit soft targets (`soft_oracle_targets`, `:168-175`).

### ER-02: `per_query_table`'s cost-aware oracle ignores the eligibility mask that the summary applies `:319-325`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
321:            util = true - float(lam) * C[None, :]
322:            ca_idx = util.argmax(axis=1)
323:            rows["cost_aware_oracle_model_id"] = [model_ids[k] for k in ca_idx]
```

**What breaks:** `routing_evaluation` masks ineligible cells before computing its cost-aware oracle (`:259-260`), but its call to `per_query_table` at `:280-283` doesn't pass `eligible`. With a constraint mask, the per-query `cost_aware_oracle_model_id` and `cost_aware_oracle_utility` can name an ineligible model. They then disagree with the summary's `cost_aware_oracle_model_mix` and `mean_utility_regret` for the same run.
**Evidence:** Traced `:280-283`. `tests/evaluation/routing/test_routing_eval.py` covers `per_query_table` without a mask.
**Direction:** Pass `eligible` through and apply the same mask.

### ER-03: One query with no eligible model turns the cost-aware summary into NaN `:258-275`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
260:            util = np.where(np.asarray(eligible, bool), util, -np.inf)
263:        sel_util = util[qi, selected]
272:            "mean_utility_regret": float(np.maximum(ca_oracle_util - sel_util, 0.0).mean()),
```

**What breaks:** For an all-ineligible row, `routing_decision` still returns column 0 (RR-03), and the oracle argmax also lands on column 0. Both utilities are `-inf`, and `-inf - -inf` is NaN. NaN survives `np.maximum` and `.mean()`, so a single such query turns `mean_utility_regret` into NaN, and `mean_selected_utility` / `mean_cost_aware_oracle_utility` into `-inf`, for the whole report.
**Evidence:** Traced the numpy semantics: `np.maximum(nan, 0)` returns `nan`.
**Direction:** Exclude and count rows with no eligible candidate before aggregating.

### ER-04: `evaluate_router` turns every regret into NaN when a pool model is missing from `true_df` `:73-80`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
75:    true = true_df.reindex(columns=model_ids).to_numpy(np.float64)
```

**What breaks:** Reindexing to the router's pool adds an all-NaN column for any pool model that isn't in the outcome matrix. `true.max(axis=1)` is then NaN on every row, so regret, hit rates and oracle quality are all NaN, with no error. The docstring asks the caller to restrict `true_df` to the pool, but nothing checks it.
**Evidence:** Read `:54-91`. `tests/router/routing/test_routing_interface.py:128` only covers a matching subset.
**Direction:** Raise when `set(model_ids) - set(true_df.columns)` isn't empty.

### ER-05: `oracle_labels` raises `IndexError` on any NaN row `:134-145`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
134:    oracle_score = true.max(axis=1)
143:        cands = np.flatnonzero(flag[i])
144:        best = cands[np.lexsort(([models[c] for c in cands], tiebreak[i, cands]))[0]]
```

**What breaks:** A NaN cell makes the row's `oracle_score` NaN. `flag` is then all `False`, `cands` is empty, and `[0]` raises `IndexError`. `per_query_model_table` (`:329-346`) passes arbitrary arrays into it. The same data makes `routing_evaluation` silently return NaN (ER-04), so the two entry points fail differently on the same input.
**Evidence:** Read `:128-147`.
**Direction:** Check density up front and raise a clear error.

### ER-06: Oracle labels build one Python dict per cell `:141-165`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
149:    for i, qid in enumerate(true_df.index):
150:        ranks = _dense_rank_desc(true[i], tol)
151:        for j, m in enumerate(models):
```

**What breaks:** A 36.5k-query × 20-model pool produces about 730k dicts. Each row also runs its own Python `lexsort` and a Python dense-rank loop. Expect tens of seconds and several hundred MB of memory.
**Evidence:** Read `:97-165`.
**Direction:** Build the columns with `np.repeat` / `np.tile`, compute dense ranks with a sorted cumsum over `diff > tol`, and pick oracle models with one matrix-wide `lexsort`.

---

## src/evaluation/routing/__init__.py

No findings.

---

## Checked and ruled out
- **Hard-oracle tie-break perturbation (`:389-392`):** it shifts scores by at most 1e-9, which equals `_TOL`, so only within-tolerance ties are reordered, as intended.
- **`per_task_argmax_baseline` with an unseen family:** falls back to the global best, as documented.
- **`soft_oracle_targets` overflow:** subtracting `z.max()` keeps it stable.
