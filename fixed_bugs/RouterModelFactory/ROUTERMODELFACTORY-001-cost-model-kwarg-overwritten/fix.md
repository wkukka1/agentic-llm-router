# ROUTERMODELFACTORY-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-20, committed in `cfb7007`
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`RouterModelFactory.create` assigns the positional `store_or_cost_model` to `cost_model` only when no `cost_model=` keyword was given, so an explicit keyword is no longer overwritten.

Regression test `tests/router/models/test_factory.py::test_factory_create_forwards_the_cost_model_keyword`

## Fix suggested when the bug was filed

```python
artifact = artifact_or_id
if cost_model is None:
    cost_model = store_or_cost_model
```

or split the two overloads into separate methods, which also removes the ambiguity.

## Regression tests

`tests/router/models/test_factory.py:58` `test_factory_create_forwards_the_cost_model_keyword`

```python
def test_factory_create_forwards_the_cost_model_keyword():
    """ROUTERMODELFACTORY-001: ``create(artifact, cost_model=cm)`` overwrote the
    keyword with the (None) positional slot, so ``from_artifact`` got no cost model."""
    class Recorder:
        @classmethod
        def from_artifact(cls, artifact, cost_model=None):
            return cost_model

    factory = RouterModelFactory()
    factory.register("recorder", Recorder)
    artifact = SimpleNamespace(router_model_name="recorder")
    cost_model = CostModel()
    assert factory.create(artifact, cost_model=cost_model) is cost_model
    assert factory.create(artifact, cost_model) is cost_model    # positional still works
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/models/registry.py`

```diff
--- a/src/router/models/registry.py
+++ b/src/router/models/registry.py
@@ -47,7 +47,8 @@ class RouterModelFactory:
             artifact = store.load(artifact_or_id)
         else:
             artifact = artifact_or_id
-            cost_model = store_or_cost_model
+            if cost_model is None:      # don't clobber an explicit ``cost_model=`` keyword
+                cost_model = store_or_cost_model
 
         model_class = self._registry.get(artifact.router_model_name)
         if model_class is None:
```
