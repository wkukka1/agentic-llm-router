# KNNROUTER-001: routing only query ids missing from the store crashes inside scikit-learn

- **Status:** reproduced
- **Severity:** low-medium
- **Class:** `KNNRouter.predict_scores` ([routers.py:423-429](../../../src/router/routing/routers.py#L423-L429)); root cause in `knn_router_matrix` ([baselines_infer.py:69-72](../../../src/router/nirt/baselines_infer.py#L69-L72))
- **Found:** 2026-09-19

## What

`knn_router_matrix` drops ids that are not in the store (`ids = [q ... if q in q_store._index]`),
which is fine for a partial miss: `RouterModel.aligned_scores` reindexes to the requested
ids, the missing rows become NaN, and `RoutingResult.fallback` flags them. When **every**
id is missing, `ids` is empty and `nn.kneighbors` receives a `(0, d)` array.

## Failure scenario

Repro (`known_bugs/_repros/repro_knn.py`):

```
known ids  : (1, 2)
unknown id : ValueError - Found array with 0 sample(s) (shape=(0, 4)) while a minimum of 1
             is required by NearestNeighbors.
```

`KNNRouter.route(["not-in-store"])` raises an opaque scikit-learn error instead of returning
a flagged fallback row, and a batch of one unknown id behaves differently from a batch of
one unknown id plus one known id.

## Suggested fix

Return an empty frame (`pd.DataFrame(index=[], columns=model_ids)`) when `ids` is empty, so
the base class produces its NaN fallback rows.
