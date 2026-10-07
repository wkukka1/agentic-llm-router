# ROUTERMODEL-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-20, committed in `cfb7007`
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`__call__ = route` is replaced by a real `__call__` that calls `self.route(...)`, so `router(...)` honours a subclass's `route` override.

Regression test `tests/router/routing/test_routing_interface.py::test_call_dispatches_to_a_subclass_route_override`

## Fix suggested when the bug was filed

```python
def __call__(self, *args, **kwargs):
    return self.route(*args, **kwargs)
```

## Regression tests

`tests/router/routing/test_routing_interface.py:68` `test_call_dispatches_to_a_subclass_route_override`

```python
def test_call_dispatches_to_a_subclass_route_override():
    """ROUTERMODEL-001: ``__call__ = route`` froze the base function, so
    ``router(...)`` bypassed a subclass's own ``route()``."""
    true_df, _ = _mats()

    class Overriding(MatrixRouter):
        def route(self, *args, **kwargs):
            raise RuntimeError("subclass route() called")

    with pytest.raises(RuntimeError, match="subclass route"):
        Overriding(true_df)(["q001"])
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/routing/base.py`

```diff
--- a/src/router/routing/base.py
+++ b/src/router/routing/base.py
@@ -341,7 +347,10 @@ class RouterModel(abc.ABC):
             lam=lam, model_costs=model_costs, eligible=eligible,
         )
 
-    __call__ = route
+    def __call__(self, *args, **kwargs) -> RoutingResult:
+        # not ``__call__ = route``: that binds the base function, so a subclass
+        # overriding route() would be bypassed by ``router(...)``
+        return self.route(*args, **kwargs)
 
     def route_text(
         self,
```
