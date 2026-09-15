# Code review: training.retrieval

**Files reviewed:** 4 (481 lines): `__init__.py`, `knn_impute.py`, `query_bank.py`, `warmup.py` · **Context-only:** `src/training/data/loaders.py`, `src/training/data/splits.py`, `src/router/embeddings/encoder.py`
**Summary:** 5 findings: 3 correctness, 1 design, 1 performance · 5 confirmed, 0 suspected

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| TRV-01 | `src/training/retrieval/query_bank.py:151-162` | correctness | medium | confirmed |
| TRV-02 | `src/training/retrieval/knn_impute.py:153-157` | correctness | low | confirmed |
| TRV-03 | `src/training/retrieval/query_bank.py:26-40` | correctness | low | confirmed |
| TRV-04 | `src/training/retrieval/warmup.py:50-56` | design | low | confirmed |
| TRV-05 | `src/training/retrieval/query_bank.py:194-207` | performance | low | confirmed |

---

## src/training/retrieval/query_bank.py

### TRV-01: Self-exclusion drops only the exact id, so text-identical siblings still impute a query from itself `:151-162`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
151:        over = min(k + (1 if exclude_ids is not None else 0), len(self.ids))
159:            row = [(i, s) for i, s in zip(ids[r], scores[r]) if i != drop][:k]
```

**What breaks:** RouterBench 0-shot and 5-shot items are separate query ids that carry identical text (`loaders.py:73-82,123-126`), so they have identical embeddings. They also share a content hash, so both land in the same split (`splits.py:88-103`). For a train query `X:5shot`, the nearest neighbour in the bank is its sibling `X`, at cosine 1.0. That sibling isn't the excluded id, so it survives. Exact duplicates across sources behave the same way. Two consequences:
- **Imputation mismatch:** the "self-excluded" kNN vector for train multishot items is dominated by the item's own embedding, while validation and test queries (whose siblings are also held out) get genuine neighbour means. Train and eval representations therefore come from different distributions for the `knn*` pathways, which inflates train fit relative to eval.
- **Short neighbour lists:** only one extra neighbour is fetched (`over = k + 1`). When several copies exist, the kept list is shorter than `k` and silently padded with empty ids at score 0.

**Evidence:** Traced `build_knn_imputed_store` → `neighbor_weighted_mean` → `search(exclude_ids=chunk)`. The loader's 0/5-shot policy puts the same `query` text on both ids.
**Direction:** Exclude by content hash (an equivalence-class id) rather than by query id, and over-fetch by the size of the largest duplicate group.

### TRV-03: Without `nirt_observations.parquet` the bank silently indexes every split query `:26-40`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
36:    p = cfg.path("processed") / "nirt_observations.parquet"
37:    if p.exists():
38:        obs = pd.read_parquet(p, columns=["query_id", "split"])
```

**What breaks:** The docstring promises "training queries that carry a NIRT correctness observation". If the observation table hasn't been built yet, the bank instead holds every train query, including Arena and judge prompts that only have pairwise data. kNN imputation and warm-up then average over neighbours that have no correctness labels, with no warning, and the manifest doesn't record which variant was built. This is one instance of the build-order issue in XA-11.
**Evidence:** Read `:26-40`. The manifest written at `:82-89` has no field for it.
**Direction:** Raise when the parquet is missing, or record the choice in the manifest.

### TRV-05: Neighbour means use a per-row Python loop with one `gather` per row `:194-207`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
194:        for r in range(n):
195:            pairs = [(i, s) for i, s in zip(res.ids[r], res.scores[r]) if i and i in store]
200:            vecs = store.gather(nbr).astype(np.float64)
```

**What breaks:** About 36k queries each do a fancy-indexed memmap gather in Python. The same per-row loop appears in `search` (`:158-161`) and `neighbor_mean` (`:169-172`).
**Evidence:** Read `:141-207`.
**Direction:** Map the whole `(n, k)` id matrix to rows at once with `store.rows_of`, masking missing ids, then compute the weighted mean in one vectorised step.

---

## src/training/retrieval/knn_impute.py

### TRV-02: `load_or_build_imputed_store` reuses a store built from a different bank or target pathway `:153-157`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
153:    if not rebuild and EmbeddingStore.exists(d):
156:        if m.get("knn_k") == int(k) and m.get("weighting") == weighting:
157:            return store
```

**What breaks:** Staleness is judged on `k` and `weighting` alone. Passing `bank_dir=` (the family-excluded OOD bank), `target_pathway=`, `retrieval_pathway=` or `query_ids=` through `**build_kw` returns whatever store was built first under that name. The manifest records `retrieval_pathway`, `target_pathway` and `bank_holdout_families`, but none of them is compared.
**Evidence:** Read `:116-135` and `:146-159`.
**Direction:** Compare every manifest field that `build_kw` can change, plus a bank fingerprint.

---

## src/training/retrieval/warmup.py

### TRV-04: Warm-up re-implements `QueryBank.neighbor_mean` inline `:50-56`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
52:        res = bank.search(retr_vecs[i : i + B], k=k, exclude_ids=chunk)
54:            nbr = [n for n in res.ids[r] if n and n in model_store]
56:                out[i + r] = model_store.gather(nbr).mean(axis=0)
```

**What breaks:** This loop duplicates `QueryBank.neighbor_mean` (`query_bank.py:164-173`) and adds the empty-id filter that `neighbor_mean` lacks. `knn_impute` uses a third variant, `neighbor_weighted_mean`. Fixes such as TRV-01 must be made in three places.
**Evidence:** Diffed the three loops.
**Direction:** Call `bank.neighbor_weighted_mean(..., weighting="uniform")` here.

---

## src/training/retrieval/__init__.py

No findings.

---

## Checked and ruled out
- **FAISS `-1` indices wrapping to the last id:** `over <= len(self.ids)`, and flat indexes return `ntotal >= over` results.
- **Validation or test queries leaking into the bank:** `_bank_query_ids` restricts to the requested split.
- **`flat_ip` on unnormalised vectors:** both `_prep` and `build_query_bank` unit-normalise when `normalized=True`.
- **The imputed store's manifest saying `normalized: False`:** accurate, because a weighted mean of unit vectors isn't unit-norm.
