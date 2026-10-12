# ROUTER-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-20, committed in `cfb7007`; partly uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`Router.route` prices a kept candidate that is outside the router's pool at the pool's maximum per-query cost (`off_pool_cost`), so it stays on the same USD/query scale as the pool. The profile's per-token price is used only when the router has no cost vector at all.

Regression test `tests/router/test_router_pipeline.py::test_unsupported_candidate_is_costed_on_the_pool_scale`

## Fix suggested when the bug was filed

When `pool_costs` is not `None`, price a candidate outside the pool with a pool-scale
value, for example the pool maximum (the same pessimistic choice `_mean_train_costs`
makes). Use the profile's per-token price only when the router has no cost vector at all.

## Regression tests

`tests/router/test_router_pipeline.py:153` `test_unsupported_candidate_is_costed_on_the_pool_scale`

```python
def test_unsupported_candidate_is_costed_on_the_pool_scale():
    """ROUTER-001: an off-pool candidate kept by SCORE_WITH_PRIOR used to get its
    per-*token* price (6e-5) next to per-*query* costs, so a model the router has no
    evidence for looked ~1000x cheaper than everything else and won."""
    reg = LLMRegistry()
    for mid, out_cost in [("big", 6e-5), ("small", 6e-6), ("unseen", 6e-5)]:
        reg.register(LLMProfile(name=mid, provider="x", model_id=mid,
                                output_cost_per_token=out_cost))
    rm = _CostedTextRouter()
    rm.unsupported_policy = UnsupportedCandidatePolicy.SCORE_WITH_PRIOR

    decision = Router(rm, llm_registry=reg).route(RoutingRequest(
        request_id="r8", prompt="hi",
        constraints=RoutingConstraints(objective=OptimizationObjective(cost_weight=10.0)),
    ))

    by_id = {s.model.model_id: s for s in decision.ranked_models}
    assert by_id["unseen"].supported is False
    assert by_id["unseen"].expected_cost == pytest.approx(0.010)   # pool max, not 6e-5
    assert decision.ranked_models[0].model.model_id != "unseen"
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/router.py`

```diff
--- a/src/router/router.py
+++ b/src/router/router.py
@@ -136,6 +146,10 @@ class Router:
         pool_costs = self.router_model.default_model_costs
         cost_of = (dict(zip(self.router_model.model_ids, map(float, pool_costs)))
                    if pool_costs is not None else {})
+        # a candidate outside the pool has no entry in ``cost_of``; price it at the
+        # pool max (pessimistic, same choice as ``_mean_train_costs``) so it stays on
+        # the router's USD/query scale -- the profile's per-token price is ~1000x smaller
+        off_pool_cost = max(cost_of.values()) if cost_of else None
         with_prior = (self.router_model.unsupported_policy
                       is UnsupportedCandidatePolicy.SCORE_WITH_PRIOR)
 
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
