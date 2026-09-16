# Code review: router.data

**Files reviewed:** 2 (266 lines): `__init__.py`, `nirt.py` · **Context-only:** `src/router/embeddings/encoder.py`, `src/training/data/nirt.py`, `src/training/data/facade.py`
**Summary:** 3 findings: 2 correctness, 0 design, 1 performance · 3 confirmed, 0 suspected

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| RD-01 | `src/router/data/nirt.py:91-96` | correctness | low | confirmed |
| RD-02 | `src/router/data/nirt.py:70-79` | correctness | low | confirmed |
| RD-03 | `src/router/data/nirt.py:91-93` | performance | low | confirmed |

---

## src/router/data/nirt.py

### RD-01: An empty observation frame with a feature store crashes the constructor `:91-96`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
93:            rows = np.array([feature_store.row_of(q) if q in feature_store else -1 for q in qids])
95:            hit = rows >= 0
96:            self._feat[hit] = np.asarray(feature_store.matrix)[rows[hit]]
```

**What breaks:** When no observations survive the mask, `np.array([])` has dtype `float64`, and indexing with the empty float array `rows[hit]` raises `IndexError: arrays used as indices must be of integer (or boolean) type`. The same empty frame builds fine without a feature store. Trigger: `NIRTDataset.from_config(cfg, data=d, split="ood", query_features="default")` on a config with no `ood` rows, or any filter that removes every row.
**Evidence:** Probe on the repo's numpy: indexing `np.zeros((0,3))` with an empty float array raises that `IndexError`. No test builds an empty dataset with `feature_store`.
**Direction:** Build `rows` with `dtype=np.int64`.

### RD-02: Mismatched id types silently drop every observation `:70-79`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
70:        mask = (
71:            observations["query_id"].isin(query_store._index)
72:            & observations["model_id"].isin(profile_store._index)
```

**What breaks:** Membership is tested against the store's private id dict. Stores loaded from parquet hold `str` ids. An observation frame with integer ids (a live router's in-memory frame, or a test) matches nothing and yields a zero-length dataset. The only trace is `self.dropped`, and no caller reads it. Downstream, `predict_matrix` returns an empty pivot and routing falls back to column 0 (RR-03). The same private `_index` is read directly in `router/nirt/baselines_infer.py:29,70,113` and `evaluation/nirt/evaluate.py:149,152`.
**Evidence:** `EmbeddingStore` already exposes `__contains__` (`encoder.py:355-356`). `build_store` stringifies ids (`encoder.py:431`), but the `EmbeddingStore(...)` constructor and `.build` do not (RE-07), so both id types occur.
**Direction:** Cast both sides to `str` before the join, warn when `dropped == len(observations)`, and use the public `in store` API.

### RD-03: The feature join is a Python loop over every observation `:91-93`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
92:            qids = self.obs["query_id"].tolist()
93:            rows = np.array([feature_store.row_of(q) if q in feature_store else -1 for q in qids])
```

**What breaks:** The table has one row per `(query, model, metric)`, so each query id repeats about once per model (20 or more times). The loop does two dict lookups per row in Python. The same per-row pattern runs twice more in `rows_of` at `:78-79`. Construction runs on every `data.nirt_dataset(...)` call, and one evaluation builds several datasets.
**Evidence:** Read `:78-96`. `nirt_observations.parquet` is long-format (`training/data/nirt.py:28-31`).
**Direction:** Resolve the unique ids once (`pd.Index(store.ids).get_indexer(uniques)`) and broadcast the result with `pd.factorize`.

---

## src/router/data/__init__.py

No findings.

---

## Checked and ruled out
- **`Series.isin(dict)`:** it iterates the dict keys. Probe: `pd.Series(['a','b']).isin({'a':0})` gives `[True, False]`.
- **`gather()` ignoring `return_ids`:** callers read `dataset.query_ids` / `model_ids` separately (`router/nirt/predict.py:55-57`).
- **NaN `cost` kept as float32 NaN:** routing reads cost from its own matrices.
- **`from_config` re-implementing "read the observation parquet":** a cross-package duplication, filed as XD-05.
