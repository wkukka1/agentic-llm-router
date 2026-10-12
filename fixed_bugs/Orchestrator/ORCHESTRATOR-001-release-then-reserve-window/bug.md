# ORCHESTRATOR-001: `run()` gives up a child's allocated budget before re-reserving it

- **Status:** fixed 2026-10-02 (see [fix.md](fix.md))
- **Severity:** medium once wired (siblings share one ledger across threads by design)
- **Class:** `Orchestrator.run` ([execution/orchestrator.py:106-109](../../../src/router/execution/orchestrator.py#L106-L109))
- **Found:** 2026-09-19

## What

`spawn()` reserves the child's `allocated_budget` under its `task_id`. When the child runs,
`run()` calls `self.ledger.release(key)` "to drop any spawn-time allocation hold" and *then*
`self.ledger.reserve(key, plan.budget_estimate())`. The two ledger calls are individually
locked but not atomic together. Between them the child's budget is free, and a sibling
(which the ledger is explicitly shared with, across threads) can reserve it.

## Failure scenario

Repro (`known_bugs/_repros/repro_orchestrator.py`): total 10.0, child allocated 6.0, plan
needs 5.0. A sibling reserves 8.0 in the window:

```
after spawn: child holds 6.0, remaining = 4.0
  window: sibling reserve(8.0) -> True
child needs 5.0 of its 6.0 allocation -> over_budget | status: terminated
```

The child is terminated as `over_budget` although it was granted more than it needed.
Allocation would not be a guarantee.

## Suggested fix

Add a ledger operation that atomically resizes an existing hold
(`resize(task_id, new_amount) -> bool`, releasing the old hold only if the new amount fits),
and use it in `run()`.
