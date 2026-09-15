# Code review: router.routing

**Files reviewed:** 4 (848 lines) · **Context-only:** `src/evaluation/routing/oracle.py`, `src/training/data/facade.py` (plus `router/nirt/{routing_decision,predict,baselines_infer,checkpoint,frames}.py`, `router/data/nirt.py`, `training/trainers/mlp_router.py`, `router/agentic/router.py` read to trace call paths)
**Summary:** 17 findings: 10 correctness, 5 design, 2 performance · 15 confirmed, 2 suspected

Behavioural claims marked "probe" were checked with a throwaway script run against the real classes (`.venv/Scripts/python.exe`, scratchpad `probe.py`), not only by reading the code.

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| RR-01 | `src/router/routing/routers.py:256-264` | correctness | medium | confirmed |
| RR-02 | `src/router/routing/base.py:266-268` | correctness | medium | confirmed |
| RR-03 | `src/router/routing/base.py:270-279` | correctness | medium | confirmed |
| RR-04 | `src/router/routing/base.py:126-132` | correctness | medium | confirmed |
| RR-05 | `src/router/routing/base.py:171-195` | correctness | medium | confirmed |
| RR-06 | `src/router/routing/routers.py:95-101` | correctness | medium | suspected |
| RR-07 | `src/router/routing/routers.py:145-156` | correctness | low | confirmed |
| RR-08 | `src/router/routing/base.py:156-158` | correctness | low | confirmed |
| RR-09 | `src/router/routing/base.py:186-191` | design | low | confirmed |
| RR-10 | `src/router/routing/routers.py:123-130` | correctness | low | confirmed |
| RR-11 | `src/router/routing/routers.py:222-231` | design | low | confirmed |
| RR-12 | `src/router/routing/routers.py:222-234` | performance | low | confirmed |
| RR-13 | `src/router/routing/routers.py:401-407` | design | low | confirmed |
| RR-14 | `src/router/routing/base.py:261-265` | design | low | confirmed |
| RR-15 | `src/router/routing/base.py:45` | performance | low | confirmed |
| RR-16 | `src/router/routing/registry.py:25-26` | design | low | suspected |
| RR-17 | `src/router/routing/routers.py:366-371` | correctness | low | suspected |

---

## src/router/routing/base.py

### RR-02: A missing cell in an `eligible` DataFrame becomes *eligible* `:266-268`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
266:        elig = eligible
267:        if isinstance(eligible, pd.DataFrame):
268:            elig = eligible.reindex(index=scores.index, columns=self._model_ids).to_numpy(bool)
```

**What breaks:** `reindex` fills missing labels with `NaN`, and `.to_numpy(bool)` turns `NaN` into `True`. If a query row or model column is missing from the eligibility mask, it counts as eligible, so the mask fails open. Probe: `MatrixRouter(true).route(["q1","q2"], eligible=<frame with only q1>)` routes `q2` to its unrestricted argmax `c`.
**Evidence:** `test_eligible_mask_blocks_models` (`tests/router/routing/test_routing_interface.py:100-105`) builds a mask with full coverage, so the reindex never inserts NaN. No test covers a partial mask.
**Direction:** Use `reindex(..., fill_value=False)` so missing cells fail closed. Otherwise, raise when the mask doesn't cover `scores`.

---

### RR-03: Rows with nothing selectable silently route to column 0, even if that model is ineligible `:270-279`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
270:        mat = scores.to_numpy(np.float64)
271:        selected = routing_decision(mat, lam=lam, model_costs=costs, eligible=elig)
272:        return RoutingResult(
```

**What breaks:** `routing_decision` forces `util[rows_all_bad, 0] = -1e18` and says the fallback is "flagged by caller" (`router/nirt/routing_decision.py:42-44`). `_route_from_scores` is that caller, and it sets no flag. `RoutingResult` has no field for it either. Two cases trigger the fallback:
- A query id that is absent from the embedding store. `NIRTDataset` drops the row, `knn_router_matrix` and `mlp_router_matrix` filter it out, and the row comes back all NaN.
- An all-False `eligible` row. The probe `route(["q1"], eligible=np.zeros((1,3),bool))` selects `a`, a model the caller explicitly excluded.

In both cases `AgenticRouter` then dispatches the prompt to the first model in the pool.
**Evidence:** `test_query_order_and_missing_rows_preserved` (`test_routing_interface.py:68-74`) asserts the column-0 fallback, so the fallback itself is intended. No test checks that the result lets you detect it, or covers an ineligible fallback.
**Direction:** Add a `fallback: np.ndarray[bool]` (or `selected = -1`) to `RoutingResult`. Never return an ineligible column when an eligibility mask was supplied.

---

### RR-04: A NaN cost always wins the argmax when `lam > 0` `:126-132`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
126:    arr = np.asarray(costs, dtype=np.float64).ravel()
127:    if arr.shape != (len(model_ids),):
128:        raise ValueError(
```

**What breaks:** `_resolve_costs` checks the shape of the costs but not whether they are finite. `MatrixRouter.__init__` (`routers.py:149`) doesn't check either. `routing_decision` masks non-finite values in `pred` *before* it subtracts `lam * C`, so a NaN cost gives a NaN utility, and `np.argmax` returns the first NaN. Probe: scores `[0.9, 0.5, 0.1]`, `model_costs=[0, nan, 0]`, `lam=0.1` → selects `b`. The NaN cost also bypasses masking: a model with `pred = -inf` (missing prediction) but a NaN cost is selected. Only `_mean_train_costs` imputes NaN costs; explicit `model_costs` (Mapping or array) and `MatrixRouter(model_costs=...)` do not.
**Evidence:** Traced `base.py:261-271` → `routing_decision.py:35-45`. No test passes a non-finite cost.
**Direction:** Reject or impute non-finite entries in `_resolve_costs` and in `MatrixRouter`. Ideally `routing_decision` should also re-mask `~isfinite(util)` after subtracting the cost.

---

### RR-05: The hard constraints `max_cost` / `max_latency` / `min_quality` are never enforced `:171-195`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
180:        supported = set(self.supported_models())
181:        required = set(getattr(constraints, "required_capabilities", None) or [])
```

**What breaks:** `RoutingConstraints` documents `max_cost`, `max_latency` and `min_quality` as hard filters and points to this method (`router/constraints.py:43-45`). `filter_candidates` reads only `required_capabilities`. `RoutingPipeline.route` (`router/router.py:87`) does no filtering of its own, so a request with `max_cost=0.01` can still get an expensive model back.
**Evidence:** Grep for `max_cost|min_quality|max_latency` over src/scripts/tests/configs/docs: the only hits are the dataclass fields in `constraints.py:47-49`, `router/execution/budget.py` (unwired scaffolding) and an unrelated local in `battery.py:113`. `test_router_pipeline.py` never sets these fields.
**Direction:** Implement the three filters. `max_cost` can use candidate cost attributes; `min_quality` needs scores, so apply it after scoring. If they're deferred, drop the "hard filters" claim and raise when a constraint is set but not supported.

---

### RR-08: An iterator passed as `model_ids` is consumed by the emptiness check, leaving an empty pool `:156-158`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
156:        if not len(list(model_ids)):
157:            raise ValueError("a router needs a non-empty candidate pool")
158:        self._model_ids = [str(m) for m in model_ids]
```

**What breaks:** A generator or `map(...)` passes the emptiness check and is then empty when line 158 iterates it. Probe: `RandomRouter((m for m in M)).model_ids == []`. Every later route then reindexes to zero columns, and `argmax` on a `[Q, 0]` array raises.
**Evidence:** Probe. All in-repo callers pass lists or sorted lists. `MLPRouter`, `RandomRouter` and `build_router` pass user input through as-is.
**Direction:** Materialise once (`ids = [str(m) for m in model_ids]`) and check that list.

---

### RR-09: `UnsupportedCandidatePolicy.SCORE_WITH_PRIOR` behaves exactly like `DROP` downstream `:186-191`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
190:                # SCORE_WITH_PRIOR: kept, scored later with whatever prior predict_scores gives an
191:                # unseen column (typically NaN -> excluded by routing_decision's -inf fallback).
```

**What breaks:** The candidate is kept, but nothing ever scores it:
- `aligned_scores` (`:250`) and `_route_from_scores` (`:260`) reindex columns to `self._model_ids`, which by definition excludes the candidate.
- `RoutingPipeline.route` only builds `ModelScore`s from `row.items()` over the router's pool (`router/router.py:93-104`).

The kept candidate is therefore silently dropped, with no prior and no error.
**Evidence:** `test_filter_candidates_score_with_prior_keeps_unsupported` (`test_router_scaffolding.py:307-310`) checks only the output of `filter_candidates`, not scoring. Grep for `SCORE_WITH_PRIOR`: only `base.py` and that test.
**Direction:** Either implement a prior (e.g. a pool-mean score column for kept candidates in the pipeline) or remove the enum member until one exists.

---

### RR-14: `lam > 0` silently degrades to quality-only routing when no cost vector resolves `:261-265`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
261:        costs = (
262:            _resolve_costs(model_costs, self._model_ids)
263:            if model_costs is not None
264:            else self.default_model_costs
```

**What breaks:** `default_model_costs` returns `None` in these cases:
- `RandomRouter` and plain `Router`s always.
- Data-backed routers when the observation table has no numeric `cost` (`routers.py:71-72,75-76`).

`routing_decision` then skips the cost term (`if lam and model_costs is not None`). `RoutingResult.lam` still records the requested `lam`, so a lam sweep reports identical "cost-aware" results with no warning.
**Evidence:** Traced `base.py:261-278` → `routing_decision.py:38`.
**Direction:** Raise or warn when `lam != 0` and `costs is None`, or record the effective lam (0) in the result.

---

### RR-15: `import router.routing` eagerly imports torch, contrary to the stated lazy-import design `:45`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
45:from ..nirt.routing_decision import routing_decision
```

**What breaks:** Importing `router.nirt.routing_decision` first runs `router/nirt/__init__.py`, which imports `.checkpoint` and `.model` (`router/nirt/__init__.py:26-27`), and `model.py:37-39` imports torch at module top. The docstring claims the opposite (`routers.py:17-18`: "imported lazily ... so `import router.routing` stays cheap"). `router/nirt/__init__.py:13-15` itself warns against importing the namespace. Every lightweight consumer (`MatrixRouter`, `RandomRouter`, `RoutingPipeline`, tests) pays for torch start-up.
**Evidence:** Grep `^import torch` in `router/nirt/model.py` → line 37. The package `__init__` re-exports trigger it.
**Direction:** Move `routing_decision` to a module outside the `router.nirt` package (or make `router/nirt/__init__` lazy via `__getattr__`).

---

## src/router/routing/routers.py

### RR-01: The text path zero-pads any short embedding: a wrong encoder or a features run is silently accepted `:256-264`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
257:        want = int(getattr(model, "query_dim", e_q.shape[1]))
258:        if e_q.shape[1] < want:                       # run used structured query features
259:            e_q = np.hstack([e_q, np.zeros((len(e_q), want - e_q.shape[1]), np.float32)])
```

**What breaks:** There are two distinct failures.
1. **Features run.** The model was trained on `e_q ⊕ features`, where features are the task-family one-hot, a 5-shot flag, char length, log word count and n_choices (`training/data/query_features.py:108-120`, never zero-centred). At serving time they are all zero, meaning "no family, empty prompt, 0 choices". Every live prompt is scored off the training distribution, and `route(query_ids)` and `route_text(same text)` disagree for the same query.
2. **Wrong encoder.** Any encoder with a smaller dim than the model (e.g. a 384-d MiniLM passed as `AgenticRouter(encoder=...)` to a 768-d run; `agentic/router.py:87` forwards it) is zero-padded instead of rejected. The result is plausible-looking but meaningless scores.
**Evidence:** `model.query_dim` = `D_emb + D_f` (`_contracts.md` §1, `train.py:608`). `test_nirt_router_scores_and_routes` (`test_routing_interface.py:263-265`) asserts that padding happens. It only checks shape, never whether the padded prediction matches the id-path prediction or that a mismatched encoder is refused.
**Direction:** Pad only by exactly the feature dim (`want - q_store.dim`, where `self._query_features` is set), and error on any other mismatch. Better, compute the text-derivable features (length, word count, 5-shot flag) for live prompts rather than zeros.

---

### RR-06: `from_run` loads the default `phase0.yaml` data even for runs trained against another phase-0 config `:95-101`
**Category:** correctness · **Severity:** medium · **Confidence:** suspected

```python
95:def _load_training_data(data=None):
96:    if data is not None:
97:        return data
98:    from router.config import load_config
```

**What breaks:** `scripts/nirt/train_nirt.py` accepts `--phase0-config` (e.g. `configs/irt_router.yaml`, the IRT-Router suite) and passes it to `fit` separately (`train_nirt.py:79,91`). The stored run `config` (`train.py:605`) doesn't record which phase-0 config was used. `NIRTRouter.from_run(run)` with `data=None` (the documented usage, `agentic/router.py:18`, `base.py:14`) loads `load_config()` = `configs/phase0.yaml`, which has different `paths.*`, embedding stores and model pool. Likely results:
- Models missing from the profile store drop out of `NIRTDataset` (NaN columns).
- Query ids don't resolve, so rows fall back to column 0 (RR-03).
- If an identically named pathway exists in both configs, predictions come silently from the wrong stores.
**Evidence:** Grep for `phase0_config`/`config_path` in `training/nirt/train.py`: only `"config": nirt_cfg` is stored. The maintainer should confirm whether `irt_router.yaml` and `phase0.yaml` share `paths.processed` / pathway store dirs. If they don't, the failure is a loud FileNotFoundError; if they do, it is silent.
**Direction:** Persist the phase-0 config path (or its `paths` block) in the run config, and have `from_run` load that config when `data` is None.

---

### RR-07: `MatrixRouter` stringifies the pool but not the score frame's labels, so int-labelled frames route on all-NaN `:145-156`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
147:        super().__init__(list(scores.columns), name=name)
148:        self._scores = scores.astype(np.float64)
156:        return self._scores.reindex(index=[str(q) for q in query_ids])
```

**What breaks:** `Router.__init__` converts the pool to `str`, but `self._scores` keeps its original labels. For a frame with integer query ids or model columns, both reindexes (`:156` and `aligned_scores`) miss every label. Probe: `MatrixRouter(DataFrame([[0.1,0.9]], index=[101], columns=[1,2])).route([101])` → scores `[[nan, nan]]`, selected `'1'` via the column-0 fallback, with no error.
**Evidence:** Probe. Tests only use string labels.
**Direction:** Normalise `self._scores.index`/`.columns` to `str` in `__init__`.

---

### RR-10: The broad `except Exception` in `_encode_texts` can silently switch to another pathway's encoder `:123-130`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
123:    try:
124:        enc = load_encoder(data.cfg, pathway)
125:    except Exception:  # synthetic pathway / missing config -> try the fallback
```

**What breaks:** The fallback is meant for synthetic kNN-imputed query pathways. It also fires on transient errors: a model download or network failure, CUDA OOM, a bad `device`. For a run whose `query_pathway` is a *real* different pathway (e.g. `retrieval` vs `irt`), prompts are then encoded with the `pathway` encoder. If the dims match, no error surfaces and predictions come from the wrong embedding space. The fallback encoder is also cached under the primary key (`:129-130`), so the wrong encoder is used for the rest of the router's life.
**Evidence:** Traced `NIRTRouter.predict_scores_text` (`:237-240`) → `_encode_texts`.
**Direction:** Catch only the "pathway not configured" error (KeyError / a specific exception from `load_encoder`), or check that `pathway` is in `available_pathways(cfg)` before falling back.

---

### RR-11: The id path and the text path of `NIRTRouter` handle missing profiles and free runs differently `:222-231`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
222:        q_store = self._data.query_embeddings(self._query_pathway)
223:        m_store = self._data.profile_embeddings(self._pathway)
224:        if q_store is None or m_store is None:
```

**What breaks:**
- **Id path** (`predict_scores`): requires a profile store even for `model_params="free"` runs, which never use profiles. It also silently drops pool models absent from the profile store (`NIRTDataset` mask), producing NaN columns that are never selected.
- **Text path** (`_score_embeddings`, `:250-278`): needs the store only when projected, and a model missing from it raises `KeyError` from `m_store.get` (`encoder.py:364-365`).

So the same router can route a query id but crash on text, or the reverse.
**Evidence:** Traced both paths; `NIRTDataset` mask at `router/data/nirt.py:70-76`.
**Direction:** Factor a single "score `[N, D_q]` against the pool" routine and have `predict_scores` gather `e_q` (+features) by id and call it. That removes the duplicated forward logic too.

---

### RR-12: Each `predict_scores` call reloads the stores, materialises a Q×M grid and pivots `:222-234`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
231:        ds = NIRTDataset(_cross_obs(ids, self._model_ids), q_store, m_store, feature_store=feat)
232:        _, prob = predict_dataset(self._model, ds, self._model_index)
233:        long = pd.DataFrame({"query_id": ds.query_ids, "model_id": ds.model_ids, "pred": prob})
```

**What breaks:** Each call does the following:
- (a) `TrainingData._load_store` re-opens the embedding stores from disk; there is no caching (`facade.py:255-262`).
- (b) It builds a Python-level `itertools.product` DataFrame of Q·M rows.
- (c) `gather()` copies `[Q·M, D_q]` and `[Q·M, D_m]` float32. For Q=10k test queries, M=20 and 768-d that is about 1.2 GB. The same query vector is copied M times.
- (d) It runs `pivot_table` with a `mean` aggregation.

On the text path (`:270-279`), the model is run M times, recomputing the query head each time, and the profile vector is `np.repeat`ed N times. It is also called once per prompt in agentic serving. `KNNRouter.predict_scores` likewise reloads the store each call via `knn_router_matrix`.
**Evidence:** Traced through `router/data/nirt.py:143-170`, `predict.py:23-47`, `facade.py:255-262`.
**Direction:** Cache stores on the router instance. Compute θ/`latent_query` once per query and the model parameters once per model, then combine them; that uses the IRT factorisation instead of the cross join.

---

### RR-13: `RandomRouter` re-seeds on every call, so scores depend on row position rather than the query `:401-407`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
403:        rng = np.random.default_rng(self._seed)
404:        return pd.DataFrame(
405:            rng.random((len(ids), len(self._model_ids))),
```

**What breaks:** Row *i* always gets the same random vector whatever its query. Every single-query call picks the same model (probe: `route(["x"])`, `route(["y"])` and `route(["z"])` all → `a`). The picks for a query also change when the batch order changes. As a "chance" floor on single-query or streaming evaluation, it measures one fixed model rather than random routing.
**Evidence:** Probe. `test_random_router_is_deterministic` only checks the same batch twice.
**Direction:** Seed per query, e.g. `default_rng([seed, stable_hash(qid)])`, so scores are deterministic per query and independent of batch composition.

---

### RR-17: `MLPRouter`'s `pathway` isn't tied to the pathway the MLP was fit on `:366-371`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
366:    def __init__(self, model, model_ids: Sequence[str], *, data=None,
367:                 pathway: str = "irt", name: Optional[str] = None):
```

**What breaks:** `fit_mlp_router(pathway=...)` returns only `(model, model_ids)` (`trainers/mlp_router.py:59`), so the pathway isn't carried with the model. Wrapping the result manually with the default `pathway="irt"` after fitting on another pathway either raises a shape error or, with equal dims, scores in the wrong embedding space. `mlp_router_matrix` also silently drops query ids absent from the store (→ RR-03). `fit_mlp_router_as_router` passes the pathway correctly, so only the manual wrapping route is affected. `scripts/nirt/compare.py` and `capacity_diagnostics.py` use `fit_mlp_router` + `mlp_router_matrix`, not `MLPRouter`.
**Evidence:** Traced `trainers/mlp_router.py:19-72`. Suspected because no in-repo caller builds `MLPRouter` manually; the maintainer should confirm whether that is a supported entry point.
**Direction:** Have the fitter return (or attach to the model) the pathway and input dim, and validate `X.shape[1]` against the first Linear's `in_features` in `mlp_router_matrix`.

---

## src/router/routing/registry.py

### RR-16: The duplicate-kind check rejects re-registration after a module reload `:25-26`
**Category:** design · **Severity:** low · **Confidence:** suspected

```python
25:    if kind in REGISTRY and REGISTRY[kind] is not cls:
26:        raise ValueError(f"router kind {kind!r} already registered to {REGISTRY[kind].__name__}")
```

**What breaks:** `importlib.reload(router.routing.routers)` (notebooks, autoreload) creates new class objects with the same `kind`, and `@register` then raises `ValueError`, so the module can't be reloaded. The identity check can't tell "same class, reloaded" from "a real collision".
**Evidence:** Suspected; no reload in the repo (grep found no `importlib.reload`). Test `test_register_custom_router_roundtrips` pops its entry manually to avoid the same issue.
**Direction:** Compare `f"{cls.__module__}.{cls.__qualname__}"` instead of identity, or allow overwriting when the qualified name matches.

---

## src/router/routing/__init__.py

No findings. Re-exports only. The docstring example `build_router("bandit", model_ids=[...])` is consistent with `registry.build_router`.

---

## Checked and ruled out
- **`build_router` unused:** `_contracts.md` says it has no callers. Grep finds it exported in `routing/__init__.py:44,59`, exercised by `tests/router/routing/test_routing_interface.py:140-171`, and documented in `docs/routing_interface.md:89`. Not dead code.
- **Duplicate query ids through `aligned_scores`:** `reindex` with identical duplicate source and target labels doesn't raise (probe), and `_cross_obs` duplicates are averaged by `pivot_qm`. No crash found.
- **`NIRTRouter` pool ordering `sorted(model_index, key=model_index.get)`:** matches the training row order; free-mode refs use `model_index[m]` directly, so column order doesn't affect correctness.
- **`_mean_train_costs` NaN fill with `nanmax`:** a deliberate pessimistic imputation. All-NaN is handled by returning None (see RR-14 for the consequence).
- **`route_text` overwriting the index with `"0".."N-1"`:** documented, and `predict_scores_text` implementations already return positional rows.
- **`KNNRouter` default pool vs the fitted columns:** both come from the train split of `nirt_observations`, and `pivot_table` sorts columns, so they agree. Extra user-supplied ids become NaN columns (never selected).
- **Router → training import in `_load_training_data`:** a sanctioned exemption.
- **Encoder cache key when an explicit `encoder` is passed:** the explicit encoder bypasses the cache and is not stored, so it doesn't pollute later calls.
