# LLMCLIENT-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, committed in `cfb7007`; partly uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`LLMProfile.input_cost_per_token`/`output_cost_per_token` now default to `None` (unknown) instead of `0.0`; `LLMClient.price()` returns `None` when the price for a side that was actually used (a nonzero token count) is unset. `router.py`'s off-pool fallback (`getattr(cand, "output_cost_per_token", 0.0)`) was adjusted to coalesce the now-possible `None` back to `0.0` so it keeps its existing pessimistic-fallback behaviour rather than propagating `None` into `ModelScore.expected_cost`. `CostModel.predict_cost` was left as-is: it already fails loudly (`TypeError`) if handed a profile with an unset price it needs, which is the documented acceptable outcome, and nothing in `src/` calls it with such a profile today (see `ROUTERMODEL-002`, cost_model is not yet consumed on the live routing path). Covered by `tests/router/llm/test_client.py::test_unpriced_profile_makes_price_unknown_not_zero` and `test_price_only_requires_the_side_actually_used`.

## Fix suggested when the bug was filed

Default the two prices to `None` (unknown), have `price()` return `None` when the price it needs
is `None`, and let `CostModel` fail loudly or use a stated fallback.

## Regression tests

`tests/router/llm/test_client.py:16` `test_unpriced_profile_makes_price_unknown_not_zero`

```python
def test_unpriced_profile_makes_price_unknown_not_zero():
    """LLMCLIENT-001: a profile whose prices were never set must price a call
    ``None`` (unknown), not ``0.0`` (indistinguishable from genuinely free)."""
    client = LLMClient(adapter=None, profile=_profile())
    assert client.price(1000, 1000) is None
```

`tests/router/llm/test_client.py:23` `test_price_only_requires_the_side_actually_used`

```python
def test_price_only_requires_the_side_actually_used():
    """A profile priced on only one side (e.g. output-only) must still price
    a call that uses only that side -- LLMCLIENTFACTORY-001's repro needs
    ``price(0, 1000)`` to resolve from ``output_cost_per_token`` alone."""
    client = LLMClient(adapter=None, profile=_profile(output_cost_per_token=5e-6))
    assert client.price(0, 1000) == 5e-6 * 1000
    assert client.price(1000, 0) is None  # needs the unset input price
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/llm/client.py`

```diff
--- a/src/router/llm/client.py
+++ b/src/router/llm/client.py
@@ -57,11 +62,17 @@ class LLMClient:
 
     def price(self, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
         """``in * input_cost_per_token + out * output_cost_per_token``, or
-        ``None`` without a profile or token counts."""
+        ``None`` without a profile or token counts, or when the price for a
+        side that was actually used (a nonzero token count) is unset -- an
+        unpriced profile must not be indistinguishable from a free one."""
         if self.profile is None or (input_tokens is None and output_tokens is None):
             return None
-        return float((input_tokens or 0) * self.profile.input_cost_per_token
-                     + (output_tokens or 0) * self.profile.output_cost_per_token)
+        in_price = self.profile.input_cost_per_token
+        out_price = self.profile.output_cost_per_token
+        if (input_tokens and in_price is None) or (output_tokens and out_price is None):
+            return None
+        return float((input_tokens or 0) * (in_price or 0.0)
+                     + (output_tokens or 0) * (out_price or 0.0))
 
 
 #: serving provider -> ``"module:Class"``; imported lazily so importing the
```

`src/router/llm/profile.py`

```diff
--- a/src/router/llm/profile.py
+++ b/src/router/llm/profile.py
@@ -13,14 +13,19 @@ class LLMProfile:
     """``provider`` may be the model's *author* (as ``configs/model_registry.yaml``
     records it: ``meta``, ``mistralai`` ...); ``serving_provider`` is who is
     called (``openai``, ``together``, an OpenAI-compatible endpoint ...) and is
-    what :class:`~router.llm.client.LLMClientFactory` dispatches on when set."""
+    what :class:`~router.llm.client.LLMClientFactory` dispatches on when set.
+
+    ``input_cost_per_token``/``output_cost_per_token`` default to ``None``
+    (*unset/unknown*), not ``0.0`` -- a genuinely free model should set its
+    price to ``0.0`` explicitly; leaving it unset must not be silently priced
+    at zero (see :meth:`~router.llm.client.LLMClient.price`)."""
 
     name: str
     provider: str
     model_id: str
     version: str = ""
-    input_cost_per_token: float = 0.0
-    output_cost_per_token: float = 0.0
+    input_cost_per_token: Optional[float] = None
+    output_cost_per_token: Optional[float] = None
     context_window: int = 0
     capabilities: list[str] = field(default_factory=list)
     serving_provider: Optional[str] = None
```

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
