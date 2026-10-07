# TRAININGHARNESS-004: `TrainingJob.run_id` is never read; the harness uses `config.run_name` instead

- **Status:** confirmed (traced; code_review finding TP-02)
- **Severity:** low-medium
- **Class:** `TrainingHarness` ([training/pipeline.py:136-148](../../src/training/pipeline.py#L136-L148))
- **Found:** 2026-09-22

## What

`TrainingJob` has its own `run_id: str` field
([pipeline.py:80](../../src/training/pipeline.py#L80)).
`TrainingHarness.run(job)` immediately discards it —
`return self.train(job.trainer, job.dataset, job.config)`
([pipeline.py:137](../../src/training/pipeline.py#L137)) — and the returned
`TrainingRun.run_id` is set from `config.run_name`
([pipeline.py:148](../../src/training/pipeline.py#L148)), not `job.run_id`.

## Failure scenario

A caller that sets `TrainingJob(run_id="experiment-42", ..., config=TrainingConfig(run_name="experiment-42-attempt-3"))`
expecting `job.run_id` to identify the run gets back a `TrainingRun` labeled
`"experiment-42-attempt-3"` instead — silently, whenever the two ids diverge.
If nothing in the codebase ever constructs a `TrainingJob` with
`run_id != config.run_name`, this is latent rather than triggered.

## Suggested fix

Either drop `TrainingJob.run_id` (redundant with `config.run_name`) or make
`TrainingHarness` use it and drop the redundant `config.run_name` id instead
— pick one source of truth for the run identifier.
