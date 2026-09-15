# Code review: evaluation.nirt

**Files reviewed:** 5 (700 lines): `__init__.py`, `coldstart_eval.py`, `evaluate.py`, `ood.py`, `routing.py` · **Context-only:** `src/router/nirt/predict.py`, `src/router/nirt/routing_decision.py`, `src/training/nirt/metrics.py`, `scripts/nirt/coldstart.py`
**Summary:** 8 findings: 7 correctness, 0 design, 1 performance · 4 confirmed, 4 suspected

Filed elsewhere, not repeated here:
- **XA-02:** `routing.route` is a second cost-aware decision rule.
- **XD-02:** `oracle_choice` versus `oracle_labels` tie-breaking.
- **XD-03:** `_family_breakdown`'s hard-coded family keywords.

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| EN-01 | `src/evaluation/nirt/evaluate.py:161-179` | correctness | medium | confirmed |
| EN-02 | `src/evaluation/nirt/coldstart_eval.py:60-69` | correctness | low | confirmed |
| EN-03 | `src/evaluation/nirt/routing.py:134-136` | correctness | low | confirmed |
| EN-04 | `src/evaluation/nirt/coldstart_eval.py:52-67` | correctness | low | suspected |
| EN-05 | `src/evaluation/nirt/ood.py:70-101` | correctness | low | suspected |
| EN-06 | `src/evaluation/nirt/evaluate.py:183-189` | correctness | low | suspected |
| EN-07 | `src/evaluation/nirt/routing.py:224-229` | correctness | low | suspected |
| EN-08 | `src/evaluation/nirt/ood.py:57-60` | performance | low | confirmed |

---

## src/evaluation/nirt/evaluate.py

### EN-01: `cold_start_eval` ignores the run's `query_pathway` and `query_features` `:161-179`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
161:    warm_pred = predict_matrix(model, model_index, data, split, pathway).reindex(
164:    train_ds = data.nirt_dataset(split="train", pathway=pathway)
179:        e_q = q_store.gather(qids)
```

**What breaks:** `evaluate_split` accepts `query_pathway` and `query_features`, but `cold_start_eval` has neither parameter. It always feeds the raw `pathway` query store with no feature concat. Two failure modes:
- **Features run (`data.query_features` set):** the model's query head expects `D_q + F` inputs but receives `D_q`, so `_predict_for_model` raises `RuntimeError: mat1 and mat2 shapes cannot be multiplied`.
- **kNN-imputed run (`data.query_pathway: knn10w`):** it silently scores cold models on the un-imputed representation the model never saw, so cold-start numbers for kNN runs measure a mismatched input.

**Evidence:** Traced `scripts/nirt/coldstart.py:63,91` → `cold_start_eval(model, idx, split=..., pathway=pathway, data=d)` → `:161`, `:164`, `:179`. The same gap exists in `training/trainers/mlp_router.py` (TR-05).
**Direction:** Add `query_pathway` and `query_features` parameters, take their defaults from the run config returned by `load_run`, and build the query inputs through `data.nirt_dataset(...)`.

### EN-06: NaN cells in the warm prediction matrix propagate into the `warm_mean` / `nearest_warm` baselines `:183-189`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
187:            idx = warm_pred.index.get_indexer(qids[in_w])
188:            wm_pred[in_w] = warm_pred.to_numpy()[idx].mean(1)
```

**What breaks:** `warm_pred` is reindexed to the dense `true_w` grid. Any warm model missing from the profile store, or any query missing from the query store, leaves NaN cells, because `NIRTDataset` drops those rows (RD-02). `.mean(1)` then yields NaN for the query, and `prediction_metrics` has no NaN handling (`training/nirt/metrics.py:97-107`). The pooled `warm_mean` BCE therefore becomes NaN, and `beats_global_mean`-style comparisons silently evaluate to `False`.
**Evidence:** Read `:157-212` and `prediction_metrics`. Whether real stores ever miss a warm model was not checked.
**Question for you:** can the `irt` profile store lack any warm pool model? If so, use `np.nanmean` and report how many cells were missing.

---

## src/evaluation/nirt/coldstart_eval.py

### EN-02: Large-`k` points on the learning curve silently drop the models with the fewest observations `:60-69`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
66:                    probe_idx = rng.choice(idx_all, size=min(int(k), n), replace=False)
67:                eval_idx = np.setdiff1d(idx_all, probe_idx)
68:                if len(eval_idx) == 0:
69:                    continue
```

**What breaks:** A model with `n <= k` observations has nothing left to evaluate at that `k`, and its rows are skipped. The curve at `k=100` therefore averages over a different, larger-`n` subset of models than the curve at `k=10`. Improvement with `k` partly reflects which models survive (survivorship bias), not only more probe data.
**Evidence:** Read `:52-78`.
**Direction:** Evaluate every `k` on a fixed per-model held-out set drawn once, or report curves only for models with `n > max(k_values)`.

### EN-04: A query can appear in both the probe set and the eval set through different metric rows `:52-67`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
56:        qids = g["query_id"].to_numpy()
66:                    probe_idx = rng.choice(idx_all, size=min(int(k), n), replace=False)
```

**What breaks:** Sampling is over observation rows, not queries. For sources with several correctness metrics per `(query, model)` (lm-harness emits both `acc` and `acc_norm`), one metric row can land in the probe set and the other in the eval set. The fit then sees an almost identical label for the same θ_q it is scored on, which inflates the probe benefit.
**Evidence:** `loaders.py:357-393` writes one row per available metric. Whether `obs` passed to `lomo_eval` contains multi-metric sources was not traced.
**Question for you:** is `lomo_eval` ever run on observations that include lm-harness rows? If so, sample probe sets by `query_id`.

---

## src/evaluation/nirt/ood.py

### EN-05: `ood_datasets` can't build datasets for a `query_features` run `:70-101`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
98:        NIRTDataset(tr, q_store, p_store),
100:        NIRTDataset(ood_obs, q_store, p_store),
```

**What breaks:** `NIRTDataset` is built without a `feature_store`, and the function has no `query_features` parameter. An OOD pass for a model trained with structured features either crashes on the dimension mismatch or, if a new model is trained here, silently drops the features (the confound the capacity workstream fixed). This mirrors EN-01.
**Evidence:** Read `:70-101`. No caller was found that passes a features run, so the impact is unverified.
**Direction:** Accept `query_features` and pass `feature_store=data.query_features(name)`.

### EN-08: `split_observations` rebuilds the whole observation table on every call `:57-60`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
57:    from training.data.nirt import build_nirt_observations
60:    obs = build_nirt_observations(data.cfg, data=data, score_kind=score_kind)
```

**What breaks:** Each call re-derives correctness views, the split map and a `groupby` merge over all responses. `ood_datasets`, `ood_matrices` and `family_summary` each call it, so a typical OOD script rebuilds the table three times. `family_of_query` also iterates all distinct `(query_id, dataset)` pairs in Python.
**Evidence:** Read `:45-117`.
**Direction:** Build once and pass the frame, or read the parquet when `score_kind` matches.

---

## src/evaluation/nirt/routing.py

### EN-03: The "cheapest" fixed baseline is chosen with evaluation-split costs `:134-136`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
134:    cheapest = int(np.argmin(cost.mean(0)))
135:    rows.append(_policy_row(f"fixed: {model_ids[cheapest]} (cheapest)",
```

**What breaks:** The best-quality fixed model is chosen on train (`train_quality`), but the cheapest one uses mean cost on the evaluated split. With RouterBench's per-query costs, which track output length, the reference row is picked using test-set information. This row anchors the savings numbers in `routing_report` and the pool-expansion ledger.
**Evidence:** Read `:124-159`.
**Direction:** Choose the cheapest model from train-split costs, as the served router does (`default_model_costs`).

### EN-07: `aiq` interpolates over cost points that may repeat `:224-229`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
226:    xs = np.clip(pts[:, 0], lo, hi)
227:    ys = np.maximum.accumulate(pts[:, 1])
229:    q_r = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
```

**What breaks:** `np.interp` requires strictly increasing `xp`. Clipping to the pool's cost range collapses several frontier points onto `lo` or `hi`, and different λ values can produce the same selection and therefore the same cost. With repeated `xs`, `np.interp` doesn't check and returns implementation-defined values at those points. The trapezoid area, and so `aiq_improvement` in the ledger, can shift slightly.
**Evidence:** Read `:207-241`. Not probed against a real frontier.
**Direction:** Collapse points with equal cost to their maximum quality before interpolating.

---

## src/evaluation/nirt/__init__.py

No findings.

---

## Checked and ruled out
- **`ranking_metrics` on NaN predictions:** `cold_start_eval` filters to rows with `notna().all(1)` before calling it (`:220-221`).
- **`nearest_profile_model` with zero-norm vectors:** the `+ 1e-12` terms prevent division by zero.
- **`dense_matrices` dropping queries that lack any model:** documented (dense-only evaluation).
- **`oracle_choice` ignoring NaN in `true`:** its callers pass dense matrices from `dense_matrices`.
