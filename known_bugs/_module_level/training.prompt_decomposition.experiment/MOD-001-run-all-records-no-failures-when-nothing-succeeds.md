# MOD-001: `run_all` writes no `failures.json` / leaderboard when no experiment succeeds

- **Status:** reproduced (`known_bugs/_repros/repro_prompt_decomposition_training.py`, section 2)
- **Severity:** low
- **Where:** `run_all` ([experiment.py:242-252](../../../src/training/prompt_decomposition/experiment.py#L242-L252))
- **Found:** 2026-09-25

## What

The summary is only written under `if results:`. `failures` is only ever
persisted by `_write_summary`, and `render_markdown` has an explicit
"_no successful runs_" plus "## Failures" rendering path for exactly this
case, but it is unreachable: with no successes, `run_all` never calls
`_write_summary`.

## Failure scenario

A sweep where every config fails (a wrong `data.variant`, a missing built
split, a bad model name, or the dataset-builder defect in
`training.prompt_decomposition.data.dataset` MOD-001 cascading into
`load_splits`) leaves the output directory empty. The failure text exists only
in the log stream (`log.exception`).
Repro: one config with a nonexistent variant, `run_all(...)` returns
an empty frame and `out_dir` contains no files.

A mixed sweep is fine, because a later success triggers a write that includes
the earlier failures.

## Suggested fix

Call `_write_summary` after every config (drop the `if results:` guard;
`_summary_frame` and `render_markdown` already handle the empty case).
