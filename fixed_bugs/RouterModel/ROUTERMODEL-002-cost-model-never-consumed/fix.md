# ROUTERMODEL-002 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`_lazy_model_costs` (the shared `default_model_costs` body for `NIRTRouter`/`KNNRouter`/`MLPRouter`) now reads a per-model-id USD/query prior from `router.cost_model.output_token_priors` when `cost_model` is set, falling back to the existing train-split-mean path exactly as before when it is `None`. Depends on [ROUTERMODEL-004](../ROUTERMODEL-004-subclass-init-drops-cost-model-and-artifact/bug.md) (fixed first, same pass) so `cost_model` is reachable from direct construction, not only through `RouterModelFactory`. Covered by `tests/router/routing/test_router_model_subclass_init.py::test_default_model_costs_uses_the_injected_cost_model_when_set`.

## Fix suggested when the bug was filed

Either consume it (use `self.cost_model` in `default_model_costs` when set, falling back to
the train mean), or remove the parameter and the docstring claim until it is wired.

## Regression tests

`tests/router/routing/test_router_model_subclass_init.py:34` `test_default_model_costs_uses_the_injected_cost_model_when_set`

```python
def test_default_model_costs_uses_the_injected_cost_model_when_set():
    """ROUTERMODEL-002: a CostModel passed at construction must change the
    router's cost vector, not be silently ignored in favour of the
    train-mean cost."""
    cost_model = CostModel(output_token_priors={"a": 0.02, "b": 0.05})
    r = KNNRouter(data=object(), model_ids=_IDS, cost_model=cost_model)
    costs = r.default_model_costs
    assert list(costs) == [0.02, 0.05]
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/routing/routers.py`

```diff
--- a/src/router/routing/routers.py
+++ b/src/router/routing/routers.py
@@ -81,9 +85,18 @@ def _mean_train_costs(data, model_ids: Sequence[str]) -> Optional[np.ndarray]:
 
 
 def _lazy_model_costs(router) -> Optional[np.ndarray]:
-    """Shared ``default_model_costs`` body for a data-backed router: lazily
-    compute and cache the train-split mean cost vector on ``router._cost_cache``
-    (set to ``_UNSET`` in ``__init__``)."""
+    """Shared ``default_model_costs`` body for a data-backed router. When the
+    router carries an injected ``cost_model`` (``RouterModel.cost_model``,
+    ROUTERMODEL-002), read one USD/query prior per pool model from its
+    ``output_token_priors`` -- the same per-model-id, USD-scale shape this
+    property already returns, so this doesn't change what any caller compares
+    it against. Falls back to the lazily-cached train-split mean cost
+    (``router._cost_cache``, set to ``_UNSET`` in ``__init__``) when there is
+    no cost_model, exactly as before."""
+    cost_model = getattr(router, "cost_model", None)
+    if cost_model is not None:
+        priors = cost_model.output_token_priors
+        return np.array([float(priors.get(m, 0.0)) for m in router._model_ids], dtype=np.float64)
     if router._cost_cache is _UNSET:
         router._cost_cache = _mean_train_costs(router._data, router._model_ids)
     return router._cost_cache
```
