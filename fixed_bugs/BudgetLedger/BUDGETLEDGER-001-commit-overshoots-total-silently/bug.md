# BUDGETLEDGER-001: `commit` can push `spent` past `total` with no signal

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** low
- **Class:** `BudgetLedger.commit` ([budget.py:79-89](../../../src/router/execution/budget.py#L79-L89))
- **Found:** 2026-09-19

## What

The class docstring says the ledger "tracks `spent + reserved <= total`". `commit(task_id,
actual)` validates only that `actual` is finite and non-negative, then adds it to `spent`.
`Orchestrator.run` passes `result.actual_cost` straight through.

## Failure scenario

Repro (`known_bugs/_repros/repro_serving_cost.py`, section `BUDGETLEDGER-001`): total 10.0,
reserve 4.0, commit 15.0:

```
total=10.0 spent=15.0 remaining=-5.0
```

Real spend cannot be refused after the fact, so accepting it is reasonable. What is missing is
any indication: no warning, no `overspent` flag. The only downstream effect is that later
`reserve` calls fail, so the tree stops, but the caller never learns the cap was exceeded or by
how much.
