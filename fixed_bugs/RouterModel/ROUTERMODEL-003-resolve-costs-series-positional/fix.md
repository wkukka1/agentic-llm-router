# ROUTERMODEL-003 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-20, committed in `cfb7007`
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`_resolve_costs` converts a labelled `pd.Series` to a dict first, so it is aligned by model id like any other mapping instead of by position.

Regression test `tests/router/routing/test_routing_interface.py::test_series_model_costs_are_aligned_by_label_not_position`

## Fix suggested when the bug was filed

Handle `pd.Series` explicitly: `costs = costs.to_dict()` (or `.reindex(model_ids)`) before
the Mapping branch.

## Regression tests

`tests/router/routing/test_routing_interface.py:81` `test_series_model_costs_are_aligned_by_label_not_position`

```python
def test_series_model_costs_are_aligned_by_label_not_position():
    """ROUTERMODEL-003: a labelled ``pd.Series`` in a different order than
    ``model_ids`` must be aligned by index, like a dict."""
    true_df, _ = _mats()
    shuffled = pd.Series({"GPT-C": 0.03, "GPT-A": 0.01, "GPT-B": 0.02})
    res = MatrixRouter(true_df).route(["q001"], lam=1.0, model_costs=shuffled)
    np.testing.assert_allclose(res.model_costs, [0.01, 0.02, 0.03])
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/routing/base.py`

```diff
--- a/src/router/routing/base.py
+++ b/src/router/routing/base.py
@@ -134,6 +134,8 @@ def _resolve_costs(
     which ``argmax`` would pick whenever ``lam > 0``."""
     if costs is None:
         return None
+    if isinstance(costs, pd.Series):    # labelled: align by index like a Mapping, not by position
+        costs = costs.to_dict()
     if isinstance(costs, Mapping):
         missing = [m for m in model_ids if m not in costs]
         if missing:
```
