# PLANFORECASTER-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-10-02, uncommitted
- **Diff base:** `95c4f04` to the working tree on 2026-10-02

## What changed

([forecast.py](../../../src/router/execution/forecast.py))

`score_plan` still refreshes the plan's estimates and utility on every call, but only advances the status `DRAFT -> EVALUATED`; a plan that is `APPROVED`, `EXECUTING`, `COMPLETED` or `FAILED` keeps its status. Docstring updated to match.

## Fix suggested when the bug was filed

Only advance `DRAFT -> EVALUATED`; leave later states alone. The estimates can still be refreshed.

## Regression tests

(`tests/router/execution/test_execution_fixes.py`)

```python
@pytest.mark.parametrize("status", [PlanStatus.APPROVED, PlanStatus.EXECUTING,
                                    PlanStatus.COMPLETED, PlanStatus.FAILED])
def test_rescoring_a_later_stage_plan_keeps_its_status(status):
    plan = Plan("p", "t", status=status)
    _ConstForecaster().score_plan(plan, OptimizationObjective())
    assert plan.status is status and plan.estimated_cost == 1.0      # estimates still refreshed


def test_scoring_a_draft_marks_it_evaluated():
    plan = Plan("p", "t")
    _ConstForecaster().score_plan(plan, OptimizationObjective())
    assert plan.status is PlanStatus.EVALUATED
```

## Diff

```diff
diff --git a/src/router/execution/forecast.py b/src/router/execution/forecast.py
index bccc7ee..a291d38 100644
--- a/src/router/execution/forecast.py
+++ b/src/router/execution/forecast.py
@@ -71,6 +71,6 @@ class PlanForecaster(abc.ABC):
         a plan's admission score and a model's routing score never drift apart.
 
-        Writes the estimates, utility and ``EVALUATED`` status back onto
-        ``plan``. ``within_budget`` is checked against ``ledger.remaining()``
+        Writes the estimates and utility back onto ``plan``, and moves a
+        ``DRAFT`` plan to ``EVALUATED`` (a later status is left alone). ``within_budget`` is checked against ``ledger.remaining()``
         and ``limits`` when given."""
         cost = self.forecast_cost(plan)
@@ -93,5 +93,9 @@ class PlanForecaster(abc.ABC):
         plan.estimated_latency = latency
         plan.utility = utility
-        plan.status = PlanStatus.EVALUATED
+        # only advance DRAFT -> EVALUATED: re-scoring an approved / executing /
+        # finished plan refreshes its estimates but must not move it backwards
+        # through the lifecycle (PLANFORECASTER-001)
+        if plan.status is PlanStatus.DRAFT:
+            plan.status = PlanStatus.EVALUATED
         return PlanScore(quality=quality, cost=cost.total_cost, latency=latency,
                          utility=utility, within_budget=within)
```
