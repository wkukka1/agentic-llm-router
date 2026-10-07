# AGENTICROUTER-004 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

([router.py:63-91](../../../src/router/agentic/router.py#L63-L91); regression test `test_reentrant_run_on_the_same_instance_keeps_its_own_call_budget` in `tests/router/agentic/test_agentic_router.py`)

Per-run state (`decisions`, the call counter, `max_calls`) now lives in a `_RunState` object
held by a module-level `contextvars.ContextVar`, not on the `AgenticRouter` instance. `run()`
sets a fresh `_RunState` at the start (`token = _current_run.set(...)`) and restores the
previous value in a `finally` (`_current_run.reset(token)`) rather than clearing instance
attributes. Threads and asyncio tasks each get their own copy of the context automatically, so
two overlapping `run()` calls -- on the same instance or on a `_shallow_copy()` -- no longer
share a triage cache or a call counter; a re-entrant `run()` (e.g. a custom triage/tool
callback that calls back into the same agent) is also correctly isolated, restoring the outer
call's state when the inner one finishes rather than leaking into it. `_solve`/
`_answer_directly`/`_call_model` and the helper LLM `on_call` hooks (AGENTICROUTER-003) all
read the active state via a new `_run_state()` accessor, which lazily creates an ephemeral
one for calls made outside `run()` (e.g. driving `_call_model` directly, as the LangChain tool
tests do) rather than requiring every call site to thread a `run` parameter through by hand.

## Fix suggested when the bug was filed

Keep the per-run state in a small `_Run` object created in `run()` and passed down
(`_solve`, `_answer_directly`, `_call_model`, and the orchestrators via `agent`), or document
one-instance-per-request.

## Regression tests

`tests/router/agentic/test_agentic_router.py:460` `test_reentrant_run_on_the_same_instance_keeps_its_own_call_budget`

```python
def test_reentrant_run_on_the_same_instance_keeps_its_own_call_budget():
    """AGENTICROUTER-004: per-run state used to live on the AgenticRouter
    instance, so a run() nested inside another (e.g. a custom triage/tool
    calling back into the same agent) leaked its call count and triage cache
    into the outer run. A nested run() on the same instance must not spend
    the outer call's budget."""
    reg = ClientRegistry.echo(POOL)
    seen: dict = {}

    def nested_triage(prompt, *, model_id, predicted_quality, scores=None):
        if "outer" not in seen:
            seen["outer"] = True
            # re-enter run() on the SAME instance before the outer call
            # finishes -- simulates two overlapping run() calls without
            # needing real threads.
            inner = agent.run("a fully unrelated inner prompt")
            assert inner.mode == "single"
        return TriageDecision("single", model_id, predicted_quality, "forced", scores or {})

    agent = AgenticRouter(FakeTextRouter(), reg, triage=nested_triage, max_calls=1)
    res = agent.run("the outer prompt")
    assert res.mode == "single"
    assert res.selected_model_id in POOL
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/agentic/router.py`

```diff
--- a/src/router/agentic/router.py
+++ b/src/router/agentic/router.py
@@ -49,6 +60,37 @@ Synthesizer = Callable[..., str]
 Triage = Callable[..., TriageDecision]
 
 
+@dataclass
+class _RunState:
+    """One :meth:`AgenticRouter.run` call's state: routes already computed
+    (so a sub-task triage isn't repeated) and the shared call budget.
+
+    Lives in a :class:`contextvars.ContextVar`, not on the ``AgenticRouter``
+    instance (AGENTICROUTER-004) -- each ``run()`` call (its own thread,
+    asyncio task, or a re-entrant nested call) gets an isolated state, so
+    concurrent runs can no longer wipe each other's cached triage decisions
+    or share a call counter."""
+
+    decisions: dict[str, TriageDecision] = field(default_factory=dict)
+    calls: int = 0
+    max_calls: Optional[int] = None
+
+    def use_call(self, n: int = 1, *, enforce: bool = True) -> None:
+        """Count ``n`` model call(s) against the budget. Every LLM
+        invocation a run makes -- routed, triage, decompose, synthesize --
+        goes through here, not just the routed calls (AGENTICROUTER-003).
+        ``enforce=False`` counts without raising, for call sites that can't
+        safely raise mid-call (see ``LangChainToolOrchestrator``)."""
+        if enforce and self.max_calls is not None and self.calls + n > self.max_calls:
+            raise RuntimeError(f"max_calls={self.max_calls} model calls exhausted for this run")
+        self.calls += n
+
+
+_current_run: "contextvars.ContextVar[Optional[_RunState]]" = contextvars.ContextVar(
+    "agentic_router_current_run", default=None
+)
+
+
 class AgenticRouter:
     def __init__(
         self,
@@ -78,9 +120,13 @@ class AgenticRouter:
         self.max_depth = int(max_depth)
         self.max_calls = None if max_calls is None else int(max_calls)
         self._resolver = query_resolver
-        # per-run state (reset by run()): routes already computed, calls made
-        self._decisions: dict[str, TriageDecision] = {}
-        self._calls = 0
+        # AGENTICROUTER-003: route each helper's own LLM call through the same
+        # per-run budget a routed call goes through (see _RunState.use_call).
+        # Harmless no-op for helpers that don't expose ``on_call`` (the
+        # heuristic/naive defaults, or a user-supplied plain callable).
+        for helper in (self.triage, self.decomposer, self.synthesizer):
+            if hasattr(helper, "on_call"):
+                helper.on_call = self._use_call
 
     # ------------------------------------------------------------------ #
     # routing a single prompt                                            #
@@ -112,7 +158,7 @@ class AgenticRouter:
         rr = self.route_decision(prompt)
         model_id, q, scores = best_from_result(rr)
         decision = self.triage(prompt, model_id=model_id, predicted_quality=q, scores=scores)
-        self._decisions[prompt] = decision
+        self._run_state().decisions[prompt] = decision
         return decision
 
     # ------------------------------------------------------------------ #
@@ -120,12 +166,12 @@ class AgenticRouter:
     # ------------------------------------------------------------------ #
     def run(self, prompt: str) -> AgenticResult:
         """Triage ``prompt`` and either answer it directly or orchestrate it."""
-        self._decisions, self._calls = {}, 0
+        token = _current_run.set(_RunState(max_calls=self.max_calls))
         try:
             decision = self.triage_prompt(prompt)
             root = self._solve(prompt, 0, decision=decision)
         finally:
-            self._decisions = {}
+            _current_run.reset(token)
 
         if root.mode == "single":
             reason = decision.reason
@@ -187,7 +235,7 @@ class AgenticRouter:
     ) -> SubCall:
         """Route ``prompt`` (if the model was not already chosen) and invoke it."""
         if model_id is None:
-            known = self._decisions.get(prompt)
+            known = self._run_state().decisions.get(prompt)
             if known is not None:          # triage already routed this prompt
                 model_id, predicted_quality = known.selected_model_id, self._quality_of(known)
             else:
@@ -219,6 +265,24 @@ class AgenticRouter:
             answer=resp.content, mode="single", depth=depth, cost=resp.cost,
         )
 
+    # -- per-run state (AGENTICROUTER-004: context-local, not instance state) --
+    def _run_state(self) -> _RunState:
+        """The active run's state. Falls back to a fresh, context-local one
+        for calls made outside :meth:`run` (e.g. driving ``_solve``/
+        ``_call_model`` directly, as the LangChain tool tests do)."""
+        rs = _current_run.get()
+        if rs is None:
+            rs = _RunState(max_calls=self.max_calls)
+            _current_run.set(rs)
+        return rs
+
+    def _use_call(self, n: int = 1, *, enforce: bool = True) -> None:
+        """Count ``n`` model call(s) against this run's shared budget. Every
+        LLM invocation -- routed (``_call_model``) and helper (triage,
+        decompose, synthesize, the LangChain tool-agent's own reasoning) --
+        goes through here (AGENTICROUTER-003)."""
+        self._run_state().use_call(n, enforce=enforce)
+
     # ------------------------------------------------------------------ #
     # convenience                                                        #
     # ------------------------------------------------------------------ #
```
