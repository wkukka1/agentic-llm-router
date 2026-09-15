# Code review: training (top level, cli, phase0, trainers)

**Files reviewed:** 5 (290 lines) · **Context-only:** `src/router/nirt/model.py`, `src/router/data/nirt.py`
**Summary:** 5 findings: 2 correctness, 2 design, 1 performance · 4 confirmed, 1 suspected

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| TR-01 | `src/training/phase0.py:86-95` | correctness | low | confirmed |
| TR-02 | `src/training/phase0.py:72-76` | performance | low | confirmed |
| TR-03 | `src/training/trainers/mlp_router.py:46-59` | design | low | confirmed |
| TR-04 | `src/training/cli.py:47-59` | correctness | low | confirmed |
| TR-05 | `src/training/trainers/mlp_router.py:37-41` | design | low | suspected |

---

## src/training/__init__.py

No findings (docstring only).

---

## src/training/cli.py

### TR-04: `write_json(default=float)` turns numpy ints and bools into floats and raises on arrays or Paths `:47-59`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
56:    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
```

**What breaks:** `json.dumps` calls `default` only for non-serialisable objects:
- `np.int64` counts (e.g. `n_eval`, value counts) are written as `12.0`, and `np.bool_` as `1.0`, so consumers that expect ints see floats.
- A `Path`, `pd.Timestamp`, or multi-element `ndarray` raises `TypeError: float() argument must be…` after the parent directory has been created.

The docstring says this helper replaced the dump boilerplate in about 11 scripts and the pool-expansion battery, so all of them inherit the behaviour.
**Evidence:** Grep `write_json` finds users in `scripts/route_compare.py`, `scripts/nirt/{ab_orientation,misroute_analysis,knn_impute_sweep,compare,shrinkage_eval,coldstart,route_eval}.py`. No test covers `training.cli` (grep of `tests/` for `training.cli`/`write_json`: zero hits).
**Direction:** Use a `default` that maps `np.integer→int`, `np.bool_→bool`, `np.floating→float`, `ndarray→tolist()`, and `Path→str`.

---

## src/training/phase0.py

### TR-01: Bert-pathway query stores hold only correctness queries, so Arena/Judge battles cannot join and `train.preference` is silently empty `:86-95`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
86:        nirt_qids = set(nirt_obs["query_id"])
89:            # bert pathways only need the NIRT queries (huge CPU saving)
90:            qsub = q[q["query_id"].isin(nirt_qids)] if enc.cfg.backend == "bert" else q
```

**What breaks:** The `irt` pathway is `backend: bert` (`configs/phase0.yaml:89-90`). `nirt_obs` holds only correctness-metric rows (`training/data/nirt.py:55,78`), so the `irt` query store excludes every Arena/GPT-4-Judge query. `train.preference` with `query_pathway: null` falls back to `data.pathway = irt` (`training/nirt/train.py:376-379`). `build_pairwise_arrays` then drops every battle (`pairwise.py:64-71`), and `fit` quietly trains without the auxiliary loss (TN-07 in `training.nirt.md`). The yaml comment hints that a separate `irt_arena_sample` store is needed (`configs/nirt.yaml:118-120`), but nothing in phase0 builds one and nothing fails loudly. The same scoping means bert-pathway stores built here also lack queries that exist only for cold-start models (dropped as unsplit).
**Evidence:** Traced phase0 stage 6 → stage 7 filter → `fit` → `build_pairwise_arrays` join → `train.py:443` guard.
**Direction:** Also encode the pairwise query ids on bert pathways when pairwise sources are loaded, or have `fit` fail when preference is enabled and zero battles survive (see TN-07).

---

### TR-02: Stage 6 reloads every processed table from disk even though they are in memory `:72-76`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
75:    nirt_obs = build_nirt_observations(cfg)
```

**What breaks:** `build_nirt_observations(cfg, data=None)` calls `load_training_data(cfg)` (`training/data/nirt.py:52-54`). That re-reads `queries`, `models`, `responses`, splits, and profiles from parquet or JSON, all of which stages 1-5 just wrote and still hold (`tables`, `splits`, `profiles`). The responses table is the largest artefact in the pipeline, so this adds a full parquet round-trip.
**Evidence:** Read `phase0.py:52-76` and `training/data/nirt.py:36-56`.
**Direction:** Build a `TrainingData` from the in-memory `tables`/`splits`/`profiles` and pass `data=`.

---

## src/training/trainers/__init__.py

No findings (docstring only).

---

## src/training/trainers/mlp_router.py

### TR-03: MLP router fit has no validation or early stopping, a hard-coded batch size, and logs only the last batch's loss `:46-59`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
46:    n, bs = len(Xt), 4096
48:    for epoch in range(epochs):
57:            print(f"[mlp-router] epoch {epoch} loss {loss.item():.4f}")
```

**What breaks:** This is the "IRT-free ablation" that NIRT is compared against (`scripts/nirt/compare.py`, `capacity_diagnostics.py`). NIRT gets early stopping on validation BCE (`train.py:564-574`), but this baseline always runs a fixed `epochs=40` with no validation, so the comparison is between a tuned model and an untuned one. The verbose log prints `loss` from the final mini-batch, not the epoch mean. If the train matrix has zero queries, `loss` is never bound and the verbose print raises `NameError`. `bs=4096` cannot be configured (the `baselines.mlp_router` yaml block has no key for it).
**Evidence:** Read `:46-59`. The only test, `tests/evaluation/nirt/test_nirt_routing.py:125-133`, checks output shape and range only.
**Direction:** Accept `batch_size` and an optional validation split with patience, mirroring `training.nirt.train.fit`. Log the epoch-mean loss.

---

### TR-05: The MLP baseline ignores `data.query_pathway` and `data.query_features`, so it sees different inputs than the NIRT run it is compared with `:37-41`
**Category:** design · **Severity:** low · **Confidence:** suspected

```python
37:    C, model_ids, tr_qids = _train_correctness_matrix(data, pathway=pathway, train_obs=train_obs)
41:    X = np.asarray(data.query_embeddings(pathway).gather(list(tr_qids)), dtype=np.float32)
```

**What breaks:** NIRT runs can use a kNN-imputed query store (`data.query_pathway`) and concatenated structured features (`data.query_features`, which gave a measured val-BCE gain per `configs/nirt.yaml:22-25`). `fit_mlp_router` and `mlp_router_matrix` only read the raw `pathway` store. A head-to-head in `compare.py`/`capacity_diagnostics.py` against such a NIRT run therefore credits the input change to the IRT structure.
**Evidence:** Signature `:19-31` has no query_pathway or features parameter, and `MLPRouter` inference (`baselines_infer.py:109-117`) has none either. **Maintainer question:** do the comparison scripts ever pit the MLP against a NIRT run with `query_pathway`/`query_features` set? If they never do, this is latent only.
**Direction:** Thread `query_pathway` and `query_features` through fit and inference, matching `NIRTDataset.from_config`.

---

## Checked and ruled out
- `cli.float_table(**extra)`: flattening `extra.items()` into `option_context` positional pairs works, and dotted option names are passed via `**{"display.max_colwidth": 40}` (`scripts/route_compare.py:88`).
- `cli.zeroshot_only`: a boolean list in `.loc[...]` is correct, including on empty frames.
- `cli.resolve`: correct. Its duplication with `Config.resolve` is logged in `_cross_cutting_raw/A2.md`.
- `phase0.main`: leakage is checked before splits are written, and `report.raise_for_errors()` runs before tables are written.
- `mlp_router`: NaN cells are masked out of the BCE, and `mt[b].sum().clamp(min=1)` prevents division by zero on all-missing batches. `tr_qids` is pre-filtered to the query store, so `gather` cannot KeyError.
- `fit_mlp_router_as_router`: the trainer→router direction is allowed. The private-name imports are logged as cross-cutting.
- Coverage note, not a finding: no test imports `training.phase0` or `training.cli` (grep of `tests/`).
