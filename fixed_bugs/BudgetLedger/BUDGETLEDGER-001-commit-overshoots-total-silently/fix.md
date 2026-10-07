# BUDGETLEDGER-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

Latent -- nothing constructs a ledger outside tests yet; regression test `tests/router/test_review_fixes.py::test_commit_past_total_warns_and_reports_overspent`

`commit()` now `warnings.warn`s when the new `spent` exceeds `total`, naming
the root task id and the overrun amount. Added an `overspent` property
(`max(0.0, spent - total)`) for callers that want the number without parsing
a warning. Updated the class docstring to say the `spent + reserved <= total`
invariant holds for reservations, not for commits.

## Fix suggested when the bug was filed

Record the overrun (an `overspent` property, or a `warnings.warn` when `spent > total`), and
correct the docstring to say the invariant holds for reservations, not for commits.

## Regression tests

`tests/router/test_review_fixes.py:397` `test_commit_past_total_warns_and_reports_overspent`

```python
def test_commit_past_total_warns_and_reports_overspent():
    """BUDGETLEDGER-001: commit() must not push spent past total in silence."""
    ledger = BudgetLedger("root", 10.0)
    ledger.reserve("t", 4.0)
    with pytest.warns(UserWarning, match="exceeds total"):
        ledger.commit("t", 15.0)
    assert ledger.spent == 15.0
    assert ledger.overspent == pytest.approx(5.0)
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/execution/budget.py`

```diff
--- a/src/router/execution/budget.py
+++ b/src/router/execution/budget.py
@@ -45,6 +45,12 @@ def _amount(value: float, what: str) -> float:
 class BudgetLedger:
     """Tracks ``spent + reserved <= total`` for one root task's tree.
 
+    That invariant holds for *reservations*: ``reserve`` refuses to hold more
+    than :meth:`remaining`. It does not hold for ``commit``: real spend is
+    recorded even if it overshoots the reservation that held it, so
+    ``spent`` (and thus ``spent + reserved``) can exceed ``total`` -- see
+    :attr:`overspent`.
+
     ``reserve`` is a hold (e.g. a plan's estimated cost) that must be
     ``commit``\\ ted (replaced with the actual amount) or ``release``\\ d
     (freed, on failure/skip) -- never left dangling, or ``remaining()``
@@ -87,7 +93,20 @@ class BudgetLedger:
                 )
             self._reservations.pop(task_id, None)
             self.spent += actual
+            if self.spent > self.total:
+                warnings.warn(
+                    f"BudgetLedger.commit: spent {self.spent!r} exceeds total "
+                    f"{self.total!r} for root task {self.root_task_id!r} "
+                    f"(over by {self.spent - self.total!r})",
+                    stacklevel=2,
+                )
 
     def release(self, task_id: str) -> None:
         with self._lock:
             self._reservations.pop(task_id, None)
+
+    @property
+    def overspent(self) -> float:
+        """How far ``spent`` exceeds ``total`` -- ``0.0`` while within budget."""
+        with self._lock:
+            return max(0.0, self.spent - self.total)
```
