# MOD-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-25, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`compare_routers` and `evaluate_router` share `_require_pool_covered`; a router whose pool lacks a `true_df` model now raises. Test: `test_compare_routers_raises_when_a_routers_pool_is_missing_a_model`.

## Fix suggested when the bug was filed

Apply the same missing-column check `evaluate_router` uses (factor it into a
shared helper both functions call) before the `reindex` in `compare_routers`.

## Regression tests

`tests/router/routing/test_routing_interface.py:275` `test_compare_routers_raises_when_a_routers_pool_is_missing_a_model`

```python
def test_compare_routers_raises_when_a_routers_pool_is_missing_a_model():
    """MOD-001 (evaluation.routing.oracle): compare_routers must guard the
    same reindex-introduces-an-all-NaN-column failure mode evaluate_router
    already guards (test_evaluate_raises_when_outcomes_lack_a_pool_model)."""
    true_df, cost_df = _mats()  # columns GPT-A, GPT-B, GPT-C
    partial_pool_router = MatrixRouter(true_df[["GPT-A", "GPT-C"]], name="partial")

    with pytest.raises(ValueError, match="GPT-B"):
        compare_routers([partial_pool_router], true_df, cost_df)
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/evaluation/routing/oracle.py`

```diff
--- a/src/evaluation/routing/oracle.py
+++ b/src/evaluation/routing/oracle.py
@@ -57,6 +57,18 @@ if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
 _TOL = 1e-9
 
 
+def _require_pool_covered(label: str, have: Sequence[str], need: Sequence[str], *, reason: str) -> None:
+    """Raise if ``need`` isn't fully covered by ``have`` -- reindexing to a
+    model that's missing would add an all-NaN column and silently turn every
+    regret / hit rate into NaN. Shared by :func:`evaluate_router` (checks
+    ``true_df``/``cost_df``'s columns against the router's pool) and
+    :func:`compare_routers` (checks each router's own pool against the
+    comparison's fixed ``model_ids`` -- the opposite direction, MOD-001)."""
+    missing = sorted(set(need) - set(have))
+    if missing:
+        raise ValueError(f"{label} has no column for {missing}; {reason}")
+
+
 def evaluate_router(
     router,
     true_df: pd.DataFrame,
@@ -82,14 +94,11 @@ def evaluate_router(
     does. Pass ``model_costs=`` explicitly (via ``**kwargs``) to override.
     """
     model_ids = list(router.model_ids)
-    # reindexing to a model the outcome matrix lacks would add an all-NaN column
-    # and silently turn every regret / hit rate into NaN
     for name, frame in (("true_df", true_df), ("cost_df", cost_df)):
-        missing = sorted(set(model_ids) - set(frame.columns)) if frame is not None else []
-        if missing:
-            raise ValueError(
-                f"{name} has no column for router pool models {missing}; "
-                "restrict the router's pool or supply their outcomes"
+        if frame is not None:
+            _require_pool_covered(
+                name, frame.columns, model_ids,
+                reason="restrict the router's pool or supply their outcomes",
             )
     scores = router.aligned_scores(list(true_df.index))
     true = true_df.reindex(columns=model_ids).to_numpy(np.float64)
@@ -675,6 +684,10 @@ def compare_routers(
                 f"compare_routers: duplicate router name {r.name!r}; give each "
                 "router passed in a distinct .name"
             )
+        _require_pool_covered(
+            f"router {r.name!r}'s pool", r.model_ids, model_ids,
+            reason="restrict true_df/cost_df to the router's pool or give it those models",
+        )
         mat = r.aligned_scores(query_ids).reindex(columns=model_ids)
         preds[r.name] = mat.to_numpy(float)
 
```
