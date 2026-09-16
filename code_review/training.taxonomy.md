# Code review: training.taxonomy

**Files reviewed:** 4 (391 lines): `__init__.py`, `clustering.py`, `relevance.py`, `taxonomy.py` · **Context-only:** `src/training/data/families.py`, `src/router/embeddings/encoder.py`
**Summary:** 2 findings: 2 correctness, 0 design, 0 performance · 1 confirmed, 1 suspected

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| TX-01 | `src/training/taxonomy/relevance.py:44-62` | correctness | low | suspected |
| TX-02 | `src/training/taxonomy/clustering.py:125-131` | correctness | low | confirmed |

---

## src/training/taxonomy/relevance.py

### TX-01: Nothing ties a saved `r_q` store to the centroids that produced it `:44-62`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
55:    manifest = {"kind": "query_relevance", "pathway": pw, "dim": int(centroids.shape[0]),
56:                "id_field": "query_id", "count": len(ids), "tau": float(tau), "normalized": True,
```

**What breaks:** `load_relevance` locates its store through the current `clustering.json` (`_pathway`, `:25-26`). The relevance manifest records no centroid fingerprint or clustering timestamp. If `cluster_queries.py` is re-run (new HDBSCAN params, so a different C or cluster order) without rebuilding relevance, the old `r_q` still loads. Its column `c` no longer means centroid `c`. `DiscriminationHead.w_r` either fails on a dimension mismatch or, worse, matches by coincidence.
**Evidence:** Read `build_relevance`, `load_relevance` and `write_clusters` (`clustering.py:154-170`). No consumer in this package compares `rel.dim` with `load_centroids(cfg).shape[0]`. The Phase 1 data join was not traced.
**Question for you:** does `training/nirt/baseline/data.py` validate `r_q` against the current centroids? If not, store a centroid hash in both manifests and assert it on load.

---

## src/training/taxonomy/clustering.py

### TX-02: Clustering silently falls back to all queries when the observation table isn't built `:125-131`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
129:    p = cfg.path("processed") / "nirt_observations.parquet"
130:    col = "nirt_observations.parquet" if p.exists() else "queries.parquet"
```

**What breaks:** The docstring says "query ids carrying a NIRT correctness observation". Without the observation table, the clusters and centroids are computed over every query, including Arena and judge prompts that only have pairwise data. The ability taxonomy, and every `r_q` derived from it, therefore depends on build order, and `clustering.json` doesn't record which table was used. TRV-03 has the same pattern (see XA-11).
**Evidence:** Read `:125-151` and the metadata written at `:164-169`.
**Direction:** Raise when the parquet is missing, or write the source table into `clustering.json`.

---

## src/training/taxonomy/taxonomy.py, __init__.py

No findings. The family mapping `taxonomy.py` relies on has its own problem, filed as TD-05.

---

## Checked and ruled out
- **Empty-cluster centroid:** HDBSCAN labels are contiguous `0..C-1`, and the `(labels == c).any()` guard covers the rest.
- **`seed_everything` in `cluster_embeddings`:** process-wide seeding is filed once, as RT-06.
- **UMAP `n_jobs=1`:** slower, but required for `random_state` determinism.
- **Overflow in `relevance_from_embeddings`:** subtracting `z.max()` keeps the softmax numerically stable.
- **`_top_terms` on tiny clusters:** the empty-vocabulary `ValueError` is caught.
