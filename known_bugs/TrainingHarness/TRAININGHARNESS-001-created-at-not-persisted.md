# TRAININGHARNESS-001: `created_at` is stamped only on the in-memory artifact

- **Status:** confirmed (by reading; no runtime repro)
- **Severity:** low
- **Class:** `TrainingHarness.train` ([pipeline.py:141-144](../../src/training/pipeline.py#L141-L144))
- **Found:** 2026-09-19 (first reported by the earlier diff review; re-checked top-down)

## What

`NIRTTrainer` and `MLPRouterTrainer` persist the artifact through `LocalArtifactStore`
*inside* `trainer.train()` and return `store.load(...)`. The harness then does
`artifact.created_at = now()` on that returned object. The on-disk `run.json` was already
written with `created_at=""` and is never updated.

## Failure scenario

`TrainingRun.created_at` holds a timestamp; `LocalArtifactStore.load(run_id)` later returns
`created_at=""`. The two records of the same run disagree, and the persisted one, which is
what serving loads, has no creation time.

## Suggested fix

Have the trainer stamp `created_at` before it saves (the trainers own persistence), or have
the harness re-save through the store after stamping. The harness docstring already says
it cannot generically re-save, so the first option is the cleaner one.
