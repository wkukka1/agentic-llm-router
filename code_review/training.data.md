# Code review: training.data

**Files reviewed:** 16 (2,760 lines): `__init__.py`, `benchmark_id.py`, `chance_correction.py`, `embeddings.py`, `facade.py`, `families.py`, `loaders.py`, `model_registry.py`, `nirt.py`, `normalize.py`, `pricing.py`, `quality.py`, `query_features.py`, `response_matrix.py`, `schemas.py`, `splits.py` · **Context-only:** `src/router/data/nirt.py`, `src/router/embeddings/encoder.py`, `configs/phase0.yaml`, `configs/irt_router.yaml`, `scripts/data/run_lm_harness.py`
**Summary:** 13 findings: 9 correctness, 1 design, 3 performance · 11 confirmed, 2 suspected

Claims marked "probe" were checked with `.venv/Scripts/python.exe` (scratchpad `probe_data.py`) against the real functions and `configs/phase0.yaml`.

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| TD-01 | `src/training/data/response_matrix.py:96-107` | correctness | medium | confirmed |
| TD-02 | `src/training/data/chance_correction.py:92-101` | design | medium | confirmed |
| TD-03 | `src/training/data/embeddings.py:63-70` | correctness | medium | confirmed |
| TD-04 | `src/training/data/splits.py:71-80` | correctness | low | confirmed |
| TD-05 | `src/training/data/families.py:21-26` | correctness | medium | confirmed |
| TD-06 | `src/training/data/normalize.py:26-42` | correctness | low | confirmed |
| TD-07 | `src/training/data/loaders.py:340-354` | correctness | medium | suspected |
| TD-08 | `src/training/data/nirt.py:123-128` | correctness | low | confirmed |
| TD-09 | `src/training/data/splits.py:169-179` | correctness | low | suspected |
| TD-10 | `src/training/data/loaders.py:420-421` | correctness | low | confirmed |
| TD-11 | `src/training/data/quality.py:175-222` | performance | low | confirmed |
| TD-12 | `src/training/data/loaders.py:145-147` | performance | low | confirmed |
| TD-13 | `src/training/data/pricing.py:100-107` | performance | low | confirmed |

---

## src/training/data/response_matrix.py

### TD-01: Anchor-judge rows take over `dataset` and `source` for the RouterBench queries they judge `:96-107`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
97:    first = df.sort_values("source").groupby("query_id", as_index=False).agg(
98:        query=("query", "first"),
99:        dataset=("dataset", "first"),
```

**What breaks:** `load_anchor_judge` emits rows on existing `routerbench:` query ids with `dataset="anchor_judge"` and `source="anchor_judge"` (`loaders.py:549-557`). `_build_queries` sorts by source and keeps the first row per query id, and `"anchor_judge"` sorts before `"routerbench"`. Every judged query's row in `queries.parquet` therefore reports `dataset="anchor_judge"`. That breaks everything keyed on `queries.dataset`:
- `family_labels` / `family_of` return no family, so the query falls to `"other"`.
- `query_features` loses its family one-hot.
- Taxonomy labels are affected.
- `evaluate._family_breakdown` and the OOD split both read `responses.dataset`, and anchor rows add a second, unmapped dataset there as well.

The loader also reads `queries.parquet` from the previous build (`loaders.py:532-535`). On a fresh build that file doesn't exist, the `FileNotFoundError` is swallowed by `load_all`, and the source is skipped. A second build then includes it, so the result depends on how many times the pipeline has run (XA-11).
**Evidence:** Probe: a routerbench row plus an anchor_judge row on the same id gives `_build_queries` output `{'dataset': 'anchor_judge', 'source': 'anchor_judge'}`. `tests/training/data/test_anchor_judge_loader.py` tests the loader in isolation, not `build_tables`.
**Direction:** Build `queries.parquet` from correctness rows only (or prefer the source that minted the id), and read query text for the anchor loader from the in-memory frame of the current build.

---

## src/training/data/chance_correction.py

### TD-02: With `clip: true` the correction is an identity on 0/1 scores, so `score_effective` equals raw for binary MC data `:92-101`
**Category:** design · **Severity:** medium · **Confidence:** confirmed

```python
96:        corrected = (s - c) / (1.0 - c)
97:        if clip:
98:            corrected = np.clip(corrected, 0.0, 1.0)
```

**What breaks:** For a single observation, `s = 0` maps to `-c/(1-c)`, which is clipped back to 0, and `s = 1` maps to 1. On binary multiple-choice scores (most RouterBench MC items), the "chance-corrected" target is therefore identical to the raw target. The module docstring says a model at pure chance maps to 0, but that only holds for averages, and per-observation clipping destroys the average. The adjustment `docs/` and the facade rely on to justify pooling `accuracy` with `mc_accuracy` (`facade.py:165-170`) never happens for binary items.
**Evidence:** Probe: `normalized_correction(0.0, 4)` gives `0.0` and `normalized_correction(1.0, 4)` gives `1.0`. `tests/training/data/test_chance_correction.py` exists. Whether it asserts a non-trivial effect on binary scores was not checked.
**Direction:** Decide deliberately. Either drop `clip` for the per-observation target and let the model see negative corrected targets, or apply the correction at the aggregate level, and document which one is used.

---

## src/training/data/embeddings.py

### TD-03: `--split X --limit N` overwrites the full split store, and `limit` is taken before sorting `:63-70`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
63:    if limit:
64:        queries = queries.head(limit)
65:    queries = queries.sort_values("query_id").reset_index(drop=True)
69:    suffix = f"__{split}" if split else ("__smoke" if limit else "")
```

**What breaks:**
- **Wrong directory:** with both `split` and `limit`, the smoke run writes to `query__<pw>__<split>`, the same directory as the full split store. The id fingerprint differs, so `build_store` re-creates the memmap in `w+`, destroying hours of encoding, and leaves a 100-row store in its place (see RE-02 for the partial-write window).
- **Wrong subset:** `head(limit)` runs before `sort_values`, so the smoke set is the first N queries in file order, not "the first N by query_id" as the docstring says.

**Evidence:** Read `:54-76` and `encoder.py:439-461`.
**Direction:** Use `__{split}__smoke` when `limit` is set, and sort before `head`.

---

## src/training/data/splits.py

### TD-04: The per-source cold-start lottery gives multi-source models several draws `:71-80`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
72:    for src, g in responses.groupby("source"):
75:        k = int(round(f_cold * len(eligible)))
77:        cold_set.update(ranked[:k])
```

**What breaks:** Stratifying by source is meant to keep about `1 - f_cold` of each source's models warm. A model present in several sources (for example `gpt-4` in RouterBench and Arena) is ranked in each source's lottery, and the union makes it cold if any draw picks it. Two effects:
- More than `f_cold` of all models end up cold.
- A dense source such as RouterBench can lose models to a sparse source's draw, which the stratification was added to prevent.

**Evidence:** Read `:66-83`. `tests/training/data/test_splits.py` covers `make_splits`. Whether it uses overlapping sources was not checked.
**Direction:** Run one lottery per model, assigning each model to its dominant source, or subtract already-cold models before each source's draw.

### TD-09: `make_presplit` assigns a query found in several CSVs to the first origin in row order `:169-179`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
169:    origin_of = origin.groupby(responses["query_id"]).first()
173:        if org == "test1":
```

**What breaks:** `query_id` is `hash(task, question)`, so the same question in `train.csv` and `test1.csv` (or `test2.csv`) collapses to one id. Its split is the first origin seen, which is `train` because the loader concatenates train first. Its test1 observations are then used for training. `collapse_duplicate_observations` also averages the train and test scores for the same `(q, m, metric)` into one row. The paper's test set shrinks silently, and the numbers are no longer comparable to the paper.
**Evidence:** Read `loaders.py:424-433` (concatenation order) and `:169-179`. Whether IRT-Router's CSVs share questions across files was not checked.
**Question for you:** do `train.csv` and `test1.csv` / `test2.csv` share any `(task, question)` pairs? If yes, count them and decide which split wins, or include `origin` in the query id.

---

## src/training/data/families.py

### TD-05: Substring family matching puts MMLU math subjects in `math` and depends on YAML key order `:21-26`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
23:    d = str(dataset or "").lower()
24:    if d in fam_map:
25:        return fam_map[d]
26:    return next((fam for key, fam in fam_map.items() if key in d), None)
```

**What breaks:** For names without an exact key, the first substring hit in dict insertion order wins. `mmlu-high-school-mathematics` contains `math`, and the math family's keys come before `mmlu`, so it maps to `math` instead of `knowledge`. Other MMLU subjects map to `knowledge`. The effects:
- **OOD:** the default holdout (`math`, `code`) quietly removes the MMLU math subjects from training (`evaluation/nirt/ood.py`).
- **Features:** the `query_features` family one-hot, profile `correctness_by_family` and taxonomy labels all inherit the split.
- **Fragility:** reordering keys in `configs/phase0.yaml` changes families.

**Evidence:** Probe with `configs/phase0.yaml`: `mmlu-high-school-mathematics → math`, `mmlu-college-mathematics → math`, `mmlu-abstract-algebra → knowledge`.
**Question for you:** is MMLU math meant to be in `math` (and hence OOD)? Either way, prefer the longest matching key, or match on token boundaries, so the result doesn't depend on YAML order.

---

## src/training/data/normalize.py

### TD-06: `render_prompt` rewrites any prompt that looks like a list literal `:26-42`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
35:    if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
37:            parsed = ast.literal_eval(s)
```

**What breaks:** A genuine prompt that starts with `[` and ends with `]` (a bare list question, a JSON array task) is split into "turns" and joined with blank lines. That changes the stored query text and its content hash, so both the split group and the query id change. Only `ValueError` and `SyntaxError` are caught. A deeply nested bracket string raises `RecursionError` or `MemoryError` and aborts the whole load. The same function is applied to model responses (EV-03).
**Evidence:** Probe: `render_prompt('[1, 2, 3]')` returns `'1\n\n2\n\n3'`.
**Direction:** Unpack only when every element is a `str` (or a role/content dict), and catch `Exception`.

---

## src/training/data/loaders.py

### TD-07: lm-harness task and model are parsed from a filename layout that current lm-eval doesn't produce `:340-354`
**Category:** correctness · **Severity:** medium · **Confidence:** suspected

```python
342:    task = stem.split("_samples")[0].split("__")[-1]
343:    model_native = stem.split("__")[0] if "__" in stem else "unknown"
351:            row.get("arguments", [[None]])[0][0]
```

**What breaks:** The parser expects `<model>__<task>_samples_<ts>.jsonl`. lm-eval ≥ 0.4 writes `<output_path>/<model_dir>/samples_<task>_<timestamp>.jsonl`, and `scripts/data/run_lm_harness.py:79-80` passes `--output_path <root>/<mid>` with no filename control. On that layout, `stem.split("_samples")` finds no `_samples`, so the task becomes the whole stem including the timestamp, and the model becomes `"unknown"`. Consequences:
- Every run is its own "dataset", so `_mc_choices` misses and query ids differ per run.
- All models collapse into one `model_id`, which `collapse_duplicate_observations` then averages together.
- In ≥ 0.4, `arguments` is a dict (`{"gen_args_0": {...}}`), so `[0]` raises `KeyError`.

**Evidence:** No test covers `load_lm_harness` or `_parse_lm_harness_samples` (grep of `tests/`), and `lm-eval` isn't pinned (`requirements.txt:33-34` has it commented out). The filename format depends on the installed lm-eval version, so this is unverified.
**Question for you:** which lm-eval version produced the logs you plan to ingest? If ≥ 0.4, derive the model from the parent directory and the task from `samples_<task>_`.

### TD-10: The IRT-Router loader ignores `multiple_choice.by_prefix` `:420-421`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
420:    mc_by_task = {str(k): int(v) for k, v in
421:                  (section(cfg, "multiple_choice").get("by_task", {}) or {}).items()}
```

**What breaks:** `_mc_choices` (`:44-54`) honours both `by_task` and `by_prefix`, but `load_irt_router` reads only `by_task`. A prefix rule added for IRT-Router tasks silently leaves those items un-flagged as MC, with no chance correction. This is latent: `configs/irt_router.yaml:81` has `by_prefix: {}`.
**Evidence:** Compared `:44-54` with `:420-454`.
**Direction:** Call `_mc_choices(task, cfg)` per unique task.

### TD-12: `_melt_routerbench` recomputes per-row MC lookups for every model column `:145-147`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
145:    for m in model_cols:
146:        n_choices_by_task = base["eval_name"].map(lambda e: _mc_choices(e, cfg))
147:        is_mc = n_choices_by_task.notna()
```

**What breaks:** The value doesn't depend on `m`, yet a Python `section()` and prefix scan runs for every row × model × shot (about 36k × 11 × 2). The metadata comprehension (`:174-189`) also builds a dict per cell.
**Evidence:** Read `:143-197`.
**Direction:** Hoist the computation above the loop, and map unique `eval_name`s once.

---

## src/training/data/nirt.py

### TD-08: `nirt_dataset_from_config` ignores `build_kw` when the parquet exists, and never saves what it builds `:123-128`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
125:        observations = (
126:            pd.read_parquet(path) if path.exists()
127:            else build_nirt_observations(cfg, data=d, **build_kw)
```

**What breaks:** Passing `score_kind="raw"` or `models="all"` (via `TrainingData.nirt_dataset(**kw)`) is silently ignored whenever `nirt_observations.parquet` exists, and the dataset carries whatever target the parquet was built with. When the parquet doesn't exist, the table is rebuilt from `responses` on every call and discarded. `TrainingData.nirt_observations` does honour `build_kw` (`facade.py:319`), so the two entry points behave differently.
**Evidence:** Read `:105-133` and `facade.py:313-336`.
**Direction:** Build whenever `build_kw` is non-empty (matching the facade), and cache or persist the default build.

---

## src/training/data/quality.py

### TD-11: Quality checks normalise and hash every response row's text two more times `:175-222`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
175:    norm = df.assign(_norm=df["query"].map(normalize_query_text))
187:    raw_var = df.groupby("query_id")["query"].nunique()
222:        "Distinct content hashes": df["query"].map(content_hash).nunique(),
```

**What breaks:** The text is per query, but the checks run NFKC normalisation, lowercasing and a whitespace regex on every observation row (about 800k, many of them long 5-shot prompts). `_summarize` then hashes every row again. `build_tables` has already computed `content_hash` per row (`response_matrix.py:89`).
**Evidence:** Read the check and summary code.
**Direction:** Work on `df.drop_duplicates("query_id")` and reuse the `content_hash` column when it's present.

---

## src/training/data/pricing.py

### TD-13: `fill_costs` fills cells one at a time in Python `:100-107`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
103:    for idx in out.index[need]:
104:        filled.append((idx, cost_for(str(out.at[idx, "model_id"]),
```

**What breaks:** Each lm-harness or IRT-Router row needing a cost does a price-dict lookup, a possible `canonical_model_id` regex pass, and two `.at` writes. That is slow for large harness ingests.
**Evidence:** Read `:76-108`.
**Direction:** Map `model_id → (in_rate, out_rate)` once and compute the costs in one vectorised expression.

---

## src/training/data/facade.py, schemas.py, model_registry.py, benchmark_id.py, query_features.py, `__init__.py`

No findings beyond those above.

---

## Checked and ruled out
- **Averaged token counts failing the `Int64` cast in `collapse_duplicate_observations`:** probe with token counts 10 and 11 collapsed fine.
- **`pricing.load_prices` resolving the price file relative to CWD:** `Config.root` exists (`router/config.py:61-62`), so the path is repo-anchored.
- **`TrainingData.query_embeddings` loading a new `EmbeddingStore` on every call:** the serving-side cost is filed as RR-12.
- **Chance-correction division by zero for `n_choices=1`:** no config maps a task to 1 choice.
- **`quality.py:91` computing an unused `s`:** harmless, and out of scope as style.
- **`make_splits` leakage across content-hash groups:** groups are assigned wholesale, and `check_leakage` covers id overlap.
- **The failing tests `test_facade.py::test_{query,profile}_embeddings_read_only_by_default_returns_none`:** the toy config resolves `embedding.cache_dir` against the real repo root, where a real `query__irt` store exists. This is a test-isolation issue in `tests/`, not a `src` bug. See README.
