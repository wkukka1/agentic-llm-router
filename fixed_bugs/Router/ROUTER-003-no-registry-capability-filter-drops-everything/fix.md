# ROUTER-003 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`_StubProfile.capabilities` is now `None` (*unknown*) instead of `[]` (*known, has none*); `RouterModel.filter_candidates` only applies the capability check when `capabilities is not None`, so an unknown-capability stub candidate passes through rather than being dropped. `Router.route` also warns (matching the existing `RC-01` `max_latency` precedent) when `required_capabilities` is set with no `llm_registry`, since the constraint genuinely can't be checked in that mode. Covered by `tests/router/test_router_pipeline.py::test_required_capability_with_no_registry_warns_and_keeps_candidates`.

## Fix suggested when the bug was filed

Either raise/warn when `required_capabilities` is set and `llm_registry is
None` (capabilities aren't known in that mode, so the constraint can't be
checked), or give `_StubProfile` an explicit "capabilities unknown" marker
that `filter_candidates` treats differently from "capabilities known and
empty".

## Regression tests

`tests/router/test_router_pipeline.py:93` `test_required_capability_with_no_registry_warns_and_keeps_candidates`

```python
def test_required_capability_with_no_registry_warns_and_keeps_candidates():
    """ROUTER-003: no-registry mode can't know any candidate's capabilities.
    Silently emptying every candidate (the old behaviour) must become a
    warning plus keeping the pool, not a silent, unexplained empty result."""
    constraints = RoutingConstraints(required_capabilities=["vision"])
    pipeline = Router(FakeTextRouter(), llm_registry=None)
    with pytest.warns(UserWarning, match="required_capabilities"):
        decision = pipeline.route(
            RoutingRequest(request_id="r4", prompt="anything", constraints=constraints)
        )
    assert len(decision.ranked_models) > 0
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/router.py`

```diff
--- a/src/router/router.py
+++ b/src/router/router.py
@@ -126,6 +129,13 @@ class Router:
         if self.llm_registry is None:
             # no registry at all -> the router's bare ids are the candidates. An
             # *empty* registry is a real "no candidates" and must not fall back.
+            if constraints.required_capabilities:
+                warnings.warn(
+                    "RoutingConstraints.required_capabilities is set, but no LLMRegistry "
+                    "is configured -- candidate capabilities are unknown in this mode, so "
+                    "the constraint can't be checked and is not applied (ROUTER-003)",
+                    stacklevel=2,
+                )
             context.candidates = [_StubProfile(m) for m in self.router_model.model_ids]
         filtered = self.router_model.filter_candidates(context.candidates, request.constraints)
 
@@ -186,5 +215,8 @@ class _StubProfile:
 
     def __init__(self, model_id: str):
         self.model_id = model_id
-        self.capabilities: list[str] = []
+        # None (unknown), not [] (known -- has none): no-registry mode can't
+        # tell either apart, and filter_candidates must not treat "unknown"
+        # as "known-empty" (ROUTER-003).
+        self.capabilities: Optional[list[str]] = None
         self.output_cost_per_token = 0.0
```

`src/router/routing/base.py`

```diff
--- a/src/router/routing/base.py
+++ b/src/router/routing/base.py
@@ -217,7 +219,11 @@ class RouterModel(abc.ABC):
                     continue
                 # SCORE_WITH_PRIOR: kept. RouterModel.route can't score a column outside the pool;
                 # Router.route gives it the pool-mean prediction with confidence 0.
-            if required and not required.issubset(set(getattr(c, "capabilities", []) or [])):
+            capabilities = getattr(c, "capabilities", None)
+            # None means "unknown" (e.g. a no-registry _StubProfile, ROUTER-003) --
+            # can't be checked, so it passes rather than being treated as "known,
+            # has none" (an explicit [] still filters, same as before).
+            if required and capabilities is not None and not required.issubset(set(capabilities)):
                 continue
             out.append(c)
         return out
```
