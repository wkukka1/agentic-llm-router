# PLANFORECASTER-001: `score_plan` unconditionally sets the plan back to `EVALUATED`

- **Status:** fixed 2026-10-02 (see [fix.md](fix.md))
- **Severity:** low
- **Class:** `PlanForecaster.score_plan` ([forecast.py:95](../../../src/router/execution/forecast.py#L95))
- **Found:** 2026-09-19

## What

`plan.status = PlanStatus.EVALUATED` runs on every call. The lifecycle is
`DRAFT -> EVALUATED -> APPROVED -> EXECUTING -> COMPLETED/FAILED`, so re-scoring a plan that
has already been approved or is executing moves it backwards.

## Failure scenario

An admission step approves a plan (`APPROVED`); a monitoring hook re-scores it with updated
prices to check `within_budget`; the plan is silently `EVALUATED` again and the approval is
lost. Re-scoring an `EXECUTING` plan likewise clears the state `Orchestrator.run` set.

## Suggested fix

Only advance `DRAFT -> EVALUATED`; leave later states alone. The estimates can still be
refreshed.
