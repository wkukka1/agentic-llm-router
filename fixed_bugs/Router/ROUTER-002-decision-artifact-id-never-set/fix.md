# ROUTER-002 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-20, committed in `cfb7007`; partly uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`Router.route` fills `RoutingDecision.artifact_id` from `router_model.artifact.artifact_id` when the routing policy left it unset.

Regression test `tests/router/test_router_pipeline.py::test_decision_records_the_artifact_that_produced_it`

## Fix suggested when the bug was filed

In `Router.route`, after `decide(...)`, set
`decision.artifact_id = getattr(self.router_model.artifact, "artifact_id", None)`.

## Regression tests

`tests/router/test_router_pipeline.py:187` `test_decision_records_the_artifact_that_produced_it`

```python
def test_decision_records_the_artifact_that_produced_it():
    """ROUTER-002: RoutingDecision.artifact_id was never set."""
    artifact = SimpleNamespace(artifact_id="run-42")
    router_model = FakeTextRouter()
    router_model.artifact = artifact
    decision = Router(router_model, llm_registry=_registry()).route(
        RoutingRequest(request_id="r10", prompt="anything"))
    assert decision.artifact_id == "run-42"
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/router.py`

```diff
--- a/src/router/router.py
+++ b/src/router/router.py
@@ -86,8 +86,11 @@ class Router:
         ``expected_cost`` is the router's own per-model cost vector
         (:attr:`~router.routing.base.RouterModel.default_model_costs`, train-mean
         USD/query) when it has one -- the same scale ``RouterModel.route(lam=...)``
-        trades against -- and falls back to the profile's
-        ``output_cost_per_token`` only when the router carries no cost signal.
+        trades against. A kept candidate outside the pool is priced at the pool
+        maximum on that same scale; the profile's ``output_cost_per_token`` is
+        used only when the router carries no cost signal at all.
+        ``RoutingDecision.artifact_id`` is the router model's artifact id, if it
+        was built from one.
 
         Non-finite scores are dropped (a model with no prediction must never
         win). Candidates kept by ``UnsupportedCandidatePolicy.SCORE_WITH_PRIOR``
@@ -150,16 +164,31 @@ class Router:
                 continue
             if not math.isfinite(value):
                 continue
+            if model_id in cost_of:
+                expected_cost = cost_of[model_id]
+            elif off_pool_cost is not None:
+                expected_cost = off_pool_cost
+            else:
+                # ``output_cost_per_token`` is ``None`` for a profile with no
+                # configured price (unknown, not free) -- fall back to 0.0
+                # rather than propagating ``None`` into ``ModelScore``.
+                expected_cost = getattr(cand, "output_cost_per_token", 0.0)
+                if expected_cost is None:
+                    expected_cost = 0.0
             scores.append(ModelScore(
                 model=cand,
                 score=value,
                 expected_quality=value,
-                expected_cost=cost_of.get(model_id, getattr(cand, "output_cost_per_token", 0.0)),
+                expected_cost=expected_cost,
                 confidence=confidence,
                 supported=supported,
             ))
         scores = _apply_hard_limits(scores, request.constraints)
-        return self.routing_policy.decide(context, scores)
+        decision = self.routing_policy.decide(context, scores)
+        if decision.artifact_id is None:
+            artifact = getattr(self.router_model, "artifact", None)
+            decision.artifact_id = getattr(artifact, "artifact_id", None)
+        return decision
 
 
 def _apply_hard_limits(scores: list[ModelScore], constraints) -> list[ModelScore]:
```
