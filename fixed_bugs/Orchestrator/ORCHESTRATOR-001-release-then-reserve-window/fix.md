# ORCHESTRATOR-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-10-02, uncommitted
- **Diff base:** `95c4f04` to the working tree on 2026-10-02

## What changed

([budget.py](../../../src/router/execution/budget.py), [orchestrator.py](../../../src/router/execution/orchestrator.py))

New `BudgetLedger.resize(task_id, amount) -> bool`: under the ledger lock it replaces the task's hold (none counts as 0) with `amount`, and returns `False` -- leaving the old hold untouched -- when `amount` exceeds the free budget plus that hold. `Orchestrator.run` now calls `resize(key, plan.budget_estimate())` instead of `release(key)` + `reserve(...)`, so a child's spawn-time allocation turns into its plan reservation in one atomic step. On `over_budget` the leftover hold is released explicitly before terminating (previously the earlier `release` had already dropped it). The repro now prints `child needs 5.0 of its 6.0 allocation -> None | status: completed`.

## Fix suggested when the bug was filed

Add a ledger operation that atomically resizes an existing hold (`resize(task_id, new_amount) -> bool`, releasing the old hold only if the new amount fits), and use it in `run()`.

## Regression tests

(`tests/router/execution/test_execution_fixes.py`)

```python
def test_resize_swaps_a_hold_in_place():
    ledger = BudgetLedger("root", 10.0)
    assert ledger.reserve("a", 6.0)
    assert ledger.resize("a", 5.0) and ledger.reserved == pytest.approx(5.0)
    assert ledger.resize("a", 9.0) and ledger.reserved == pytest.approx(9.0)   # grows into free budget


def test_resize_that_does_not_fit_leaves_the_hold_untouched():
    ledger = BudgetLedger("root", 10.0)
    ledger.reserve("a", 6.0)
    ledger.reserve("b", 3.0)
    assert not ledger.resize("a", 8.0)                 # only 1.0 free beyond a's own 6.0
    assert ledger.reserved == pytest.approx(9.0)


def test_child_keeps_its_allocation_while_resizing_to_the_plan():
    """The repro: total 10, child allocated 6, plan needs 5. A sibling that
    tries to grab budget the moment the child's hold is released must not be
    able to take the child's allocation -- run() swaps the hold atomically."""
    ledger = BudgetLedger("root", 10.0)
    orch = _Fixed(OrchestratorConfig("o", "0", llm=None, ledger=ledger,
                                     limits=ExecutionLimits(max_cost=10.0)))
    orch.task = AgentTask("root", None, "root", "parent prompt")
    child = AgentTask("child", "root", "root", "child prompt", depth=1, allocated_budget=6.0)
    child_orch = orch.spawn(child)

    sibling = {}
    real_release = ledger.release

    def release_then_sibling_reserves(task_id):
        real_release(task_id)
        if task_id == "child":
            sibling["got"] = ledger.reserve("sibling", 8.0)

    ledger.release = release_then_sibling_reserves
    res = child_orch.run(child)
    assert res.success and child.status is TaskStatus.COMPLETED
    assert ledger.spent == pytest.approx(5.0)


def test_over_budget_plan_drops_the_spawn_hold():
    ledger = BudgetLedger("root", 10.0)

    class _Big(_Fixed):
        def plan(self, task):
            return Plan("p", task.task_id, steps=[PlanStep("s", "x", None, estimated_cost=50.0)])

    orch = _Big(OrchestratorConfig("o", "0", llm=None, ledger=ledger, limits=ExecutionLimits()))
    orch.task = AgentTask("root", None, "root", "p")
    child = AgentTask("child", "root", "root", "c", depth=1, allocated_budget=6.0)
    res = orch.spawn(child).run(child)
    assert res.error_code == "over_budget" and ledger.reserved == 0.0
```

## Diff

```diff
diff --git a/src/router/execution/budget.py b/src/router/execution/budget.py
index ef5d9ba..0025945 100644
--- a/src/router/execution/budget.py
+++ b/src/router/execution/budget.py
@@ -83,4 +83,18 @@ class BudgetLedger:
             return True
 
+    def resize(self, task_id: str, amount: float) -> bool:
+        """Atomically replace ``task_id``'s hold (none counts as ``0``) with
+        ``amount``. ``False`` -- and the existing hold left exactly as it was --
+        when ``amount`` exceeds what is free plus that hold. One locked step,
+        so no sibling can take the old hold in between (ORCHESTRATOR-001: a
+        separate ``release`` + ``reserve`` left that window open)."""
+        amount = _amount(amount, "resize amount")
+        with self._lock:
+            current = self._reservations.get(task_id, 0.0)
+            if amount > self.remaining() + current:
+                return False
+            self._reservations[task_id] = amount
+            return True
+
     def commit(self, task_id: str, actual: float) -> None:
         actual = _amount(actual, "committed cost")
diff --git a/src/router/execution/orchestrator.py b/src/router/execution/orchestrator.py
index ffdefc4..636d988 100644
--- a/src/router/execution/orchestrator.py
+++ b/src/router/execution/orchestrator.py
@@ -104,6 +104,9 @@ class Orchestrator(abc.ABC):
             while True:
                 plan.validate()
-                self.ledger.release(key)       # drop any spawn-time allocation hold
-                if not self.ledger.reserve(key, plan.budget_estimate()):
+                # swap any spawn-time allocation hold for the plan's estimate in
+                # one locked step -- release-then-reserve let a sibling take the
+                # allocation in between (ORCHESTRATOR-001)
+                if not self.ledger.resize(key, plan.budget_estimate()):
+                    self.ledger.release(key)
                     return self._finish(task, TaskStatus.TERMINATED,
                                         ExecutionResult(error_code="over_budget"))
```
