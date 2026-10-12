# AGENTICROUTER-002 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

> **Update 2026-10-03:** a real call whose cost can't be priced is now
> charged its pre-call *estimate* (`SubCall.meta["cost_source"] == "estimate"`,
> `AgenticRouter._leaf`) instead of making the total unknown. A cost still
> becomes `None` only when there is no estimate either. The rule this fix
> established, that an unpriced call is never reported as free, still holds.

## What changed

([result.py:38-53](../../../src/router/agentic/result.py#L38-L53), [router.py](../../../src/router/agentic/router.py); regression tests `test_unpriced_successful_call_makes_cost_unknown_not_zero`, `test_one_unpriced_leaf_makes_the_whole_orchestrated_total_unknown`, `test_failed_leaf_still_allows_a_known_total` in `tests/router/agentic/test_agentic_router.py`)

`AgenticResult.cost` is now `Optional[float] = None`. A new `result.total_cost()` helper sums
a set of leaf `SubCall`s and returns `None` the moment it hits a leaf with `cost is None` and
`error is None` (a call that genuinely completed but couldn't be priced) -- but a **failed**
leaf (`error` is set) contributes `0`, not `None`: it didn't complete, so nothing was billed,
and treating it as "unknown" would needlessly poison an otherwise fully-priced total (this
distinction is what an existing regression test, `test_failed_subcall_keeps_paid_siblings`,
already required — a failed sibling must not block a known total for the rest). `_solve`'s
branch-cost computation and both of `run()`'s `AgenticResult(...)` constructions now go
through `root.cost` directly (no more `or 0.0`), so an unpriced call is `None` all the way out
to the caller instead of silently reading as free.

## Fix suggested when the bug was filed

Make `AgenticResult.cost` `Optional[float]`: `None` when any leaf is unknown (or a separate
`cost_complete: bool`), and sum only known leaves otherwise.

## Regression tests

`tests/router/agentic/test_agentic_router.py:388` `test_unpriced_successful_call_makes_cost_unknown_not_zero`

```python
def test_unpriced_successful_call_makes_cost_unknown_not_zero():
    """AGENTICROUTER-002: a call that actually completed but whose client
    couldn't price it must make AgenticResult.cost None -- not silently 0.0,
    which would misreport a real, billed call as free."""
    reg = ClientRegistry({m: LLMClient(CallableAdapter(m, lambda p, **kw: LLMResponse(p, cost=None)))
                          for m in POOL})
    res = _agent_with(reg).run("What is the capital of France?")
    assert res.mode == "single"
    assert res.cost is None
```

`tests/router/agentic/test_agentic_router.py:399` `test_one_unpriced_leaf_makes_the_whole_orchestrated_total_unknown`

```python
def test_one_unpriced_leaf_makes_the_whole_orchestrated_total_unknown():
    """AGENTICROUTER-002: an orchestrated run with one known-cost leaf and one
    unpriced (but successful) leaf must report an unknown total, not silently
    sum the known leaf and treat the unpriced one as free."""
    def maybe_priced(p, **kw):
        return LLMResponse(p, cost=None if "second" in p else 1.0)

    reg = ClientRegistry({m: LLMClient(CallableAdapter(m, maybe_priced)) for m in POOL})
    res = _agent_with(reg).run("first part here and then second part here")
    assert res.mode == "orchestrated"
    assert res.cost is None
```

`tests/router/agentic/test_agentic_router.py:412` `test_failed_leaf_still_allows_a_known_total`

```python
def test_failed_leaf_still_allows_a_known_total():
    """A failed call (error set) is $0, not unknown -- it didn't complete, so
    nothing was billed. Distinguishes 'failed' from 'succeeded but unpriced'
    (both used to collapse to the same misleading 0.0 total)."""
    def boom(p, **kw):
        if "second" in p:
            raise TimeoutError("provider timeout")
        return LLMResponse(p, cost=1.0)

    reg = ClientRegistry({m: LLMClient(CallableAdapter(m, boom)) for m in POOL})
    res = _agent_with(reg).run("first part here and then second part here")
    assert res.mode == "orchestrated" and len(res.errors()) == 1
    assert res.cost == pytest.approx(1.0)
```

`tests/router/test_review_fixes.py:306` `test_failed_subcall_keeps_paid_siblings`

```python
def test_failed_subcall_keeps_paid_siblings():
    """RA-05"""
    def boom(p, **kw):
        if "second" in p:
            raise TimeoutError("provider timeout")
        return LLMResponse("ok", cost=1.0)

    reg = ClientRegistry({m: LLMClient(CallableAdapter(m, boom)) for m in POOL})
    res = AgenticRouter(_TextRouter([0.9, 0.9, 0.9]), reg).run("first part here and then second part here")
    assert res.mode == "orchestrated" and len(res.errors()) == 1
    assert res.cost == pytest.approx(1.0)
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/agentic/result.py`

```diff
--- a/src/router/agentic/result.py
+++ b/src/router/agentic/result.py
@@ -3,9 +3,9 @@
 from __future__ import annotations
 
 from dataclasses import dataclass, field
-from typing import Optional
+from typing import Iterable, Optional
 
-__all__ = ["SubCall", "AgenticResult"]
+__all__ = ["SubCall", "AgenticResult", "total_cost"]
 
 
 @dataclass
@@ -35,6 +35,25 @@ class SubCall:
         return (n for n in self.walk() if not n.children)
 
 
+def total_cost(nodes: Iterable[SubCall]) -> Optional[float]:
+    """Sum ``nodes``' costs (pass ``.leaves()`` -- a branch's own ``cost``
+    already includes its descendants, so summing branches double-counts).
+
+    A failed node (``error`` set) contributes ``0`` -- it didn't complete, so
+    nothing was billed. A *successful* node with ``cost is None`` (a real
+    call whose client couldn't price it, e.g. no profile configured) makes
+    the total ``None`` -- unresolved, not free (AGENTICROUTER-002: `None`
+    must not collapse to `0.0`, or an unpriced run silently reports as
+    free)."""
+    total = 0.0
+    for n in nodes:
+        if n.cost is not None:
+            total += n.cost
+        elif n.error is None:
+            return None
+    return total
+
+
 @dataclass
 class AgenticResult:
     prompt: str
@@ -45,7 +64,7 @@ class AgenticResult:
     predicted_quality: Optional[float] = None
     steps: list[SubCall] = field(default_factory=list)
     orchestrator: Optional[str] = None
-    cost: float = 0.0
+    cost: Optional[float] = None
 
     def models_used(self) -> list[str]:
         """Models actually invoked. In orchestrated mode ``selected_model_id``
```

`src/router/agentic/router.py`

```diff
--- a/src/router/agentic/router.py
+++ b/src/router/agentic/router.py
@@ -28,18 +28,29 @@ Failure handling: a model call that raises inside an orchestration is recorded
 as a failed :class:`SubCall` (``error`` set, see ``AgenticResult.errors()``) so
 the sub-answers already paid for survive; a failing *root* call still raises.
 ``max_calls`` caps the model calls one :meth:`run` may make across the whole
-recursion tree (``max_depth`` alone bounds depth, not fan-out).
+recursion tree (``max_depth`` alone bounds depth, not fan-out) -- routed calls
+*and* the helper LLMs (``LLMTriage``, ``LLMDecomposer``, ``LLMSynthesizer``,
+and the ``LangChainToolOrchestrator`` tool-agent's own reasoning steps) all
+count against it. The one gap: a tool-agent's reasoning steps are counted but
+not directly enforced mid-reasoning (there's no ``SubCall`` to fail there);
+the limit still bites the next routed call once exhausted.
+
+``run()`` is safe to call concurrently on one shared ``AgenticRouter``
+instance -- each call's routing cache and call count live in a context-local
+state (thread- or asyncio-task-isolated), not on the instance.
 """
 
 from __future__ import annotations
 
+import contextvars
+from dataclasses import dataclass, field
 from typing import Callable, Optional
 
 from ..routing.base import RouterModel, RoutingResult
 from .llm_clients import ClientRegistry, LLMClient
 from .decompose import NaiveDecomposer, concat_synthesizer
 from .orchestrator import RecursiveOrchestrator
-from .result import AgenticResult, SubCall
+from .result import AgenticResult, SubCall, total_cost
 from .triage import HeuristicTriage, TriageDecision, best_from_result
 
 __all__ = ["AgenticRouter"]
@@ -135,8 +181,10 @@ class AgenticRouter:
                 prompt=prompt, mode="single", answer=root.answer,
                 triage_reason=reason,
                 selected_model_id=root.model_id,
-                predicted_quality=decision.predicted_quality,
-                steps=[root], cost=root.cost or 0.0,
+                # the model that actually answered's own score, not the pool
+                # best decision.predicted_quality carries (AGENTICROUTER-001)
+                predicted_quality=root.predicted_quality,
+                steps=[root], cost=root.cost,
             )
         return AgenticResult(
             prompt=prompt, mode="orchestrated", answer=root.answer,
@@ -144,7 +192,7 @@ class AgenticRouter:
             selected_model_id=decision.selected_model_id,
             predicted_quality=decision.predicted_quality,
             steps=root.children, orchestrator=getattr(self.orchestrator, "name", "?"),
-            cost=root.cost or 0.0,
+            cost=root.cost,
         )
 
     # -- recursion unit ------------------------------------------------
@@ -173,7 +221,7 @@ class AgenticRouter:
             predicted_quality=decision.predicted_quality, answer=answer,
             mode="orchestrated", depth=depth, children=children,
             # leaves only: a branch child's cost already includes its descendants
-            cost=sum((n.cost or 0.0) for c in children for n in c.leaves()),
+            cost=total_cost(n for c in children for n in c.leaves()),
         )
 
     @staticmethod
```
