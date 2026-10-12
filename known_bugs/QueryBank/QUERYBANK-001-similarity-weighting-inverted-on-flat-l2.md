# QUERYBANK-001: `weighting="similarity"` weights neighbours by *distance* on a `flat_l2` bank

- **Status:** reproduced, latent (`known_bugs/_repros/repro_bank_l2_and_judge_retries.py`, section 1); nothing in `src/` builds an L2 bank
- **Severity:** low (latent)
- **Class:** `training.retrieval.query_bank.QueryBank.neighbor_weighted_mean` ([query_bank.py:240-244](../../src/training/retrieval/query_bank.py#L240-L244)); the L2 index is created at [query_bank.py:90-93](../../src/training/retrieval/query_bank.py#L90-L93)
- **Found:** 2026-09-24

## What

`build_query_bank` accepts `index_type="flat_l2"` (also reachable from
`scripts/retrieval/build_query_bank.py --index-type flat_l2`). For that index
FAISS returns **squared L2 distances**, where smaller means closer, and
`KNNResult.scores` says so ("cosine sim (flat_ip) or L2 distance (flat_l2)").
`neighbor_weighted_mean(weighting="similarity")` then uses
`np.clip(res.scores, 0, None)` as the neighbour weight, documented as
`max(cos_i, 0)`. With distances, the farthest neighbour gets the largest weight
and an exact match (distance 0) gets weight 0.

`build_knn_imputed_store` defaults to `weighting="similarity"`, so an imputed
store built off an L2 bank would be dominated by the *least* similar
neighbours.

## Repro

Bank of `near=[1,0]`, `far=[10,0]` (L2), query `[1,0]`, store vectors
`near->[1,0]`, `far->[0,1]`:

```
flat_l2 similarity-weighted mean = [0. 1.]   # all weight on the far neighbour
```

## Suggested fix

Convert L2 scores to a similarity before weighting (e.g. `1/(1+d)` or
`exp(-d)`), or refuse `weighting="similarity"` unless `self.normalized`
(inner-product index) and say so in the error.
