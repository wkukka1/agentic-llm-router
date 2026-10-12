# NIRTTRAINER-001: `phase0_cfg` is unreachable through the trainer wrapper, and its path logic is duplicated by hand

- **Status:** confirmed (traced; code_review finding TN-03)
- **Severity:** low-medium
- **Class:** `NIRTTrainer` ([training/nirt/trainer.py:33-49](../../../src/training/nirt/trainer.py#L33-L49))
- **Found:** 2026-09-22

## What

`fit()` accepts a `phase0_cfg=None` parameter that lets `_resolve_runs_dir`
([train.py:797-807](../../../src/training/nirt/train.py#L797-L807)) resolve a
relative `runs_dir` against that config's root instead of the repo root.
`NIRTTrainer.train()` calls `fit(nirt_cfg, ..., runs_dir=self.runs_dir, ...)`
without a `phase0_cfg` — there's no parameter on `NIRTTrainer.train()` to
receive one — so `fit()` always resolves with `phase0_cfg=None` on this path.

Separately, when `self.runs_dir is None`, `NIRTTrainer.train()` re-derives
the same fallback inline (`resolve_path(nirt_cfg.get("runs_dir", ...))`,
[trainer.py:44-48](../../../src/training/nirt/trainer.py#L44-L48)) instead of
calling `_resolve_runs_dir` directly. It happens to agree with `fit()`'s own
resolution today (both fall back to plain `resolve_path` since neither ever
has a real `phase0_cfg`), but the agreement is coincidental — the two
implementations of "resolve `runs_dir`" now have to be kept in sync by hand.

## Failure scenario

Latent today (no call site passes `phase0_cfg`, so both paths degrade to the
same fallback). It becomes a real, silent path mismatch the moment
`_resolve_runs_dir`'s fallback logic changes, or a caller wants
phase0-config-relative `runs_dir` resolution through the `RouterModelTrainer`
interface — `NIRTTrainer` has no way to provide it, unlike calling `fit()`
directly.

## Suggested fix

Add an optional `phase0_cfg` passthrough to `NIRTTrainer.__init__`/`.train()`
and forward it to `fit()`; have the post-`fit()` store-load path call
`_resolve_runs_dir(nirt_cfg, phase0_cfg)` directly instead of duplicating its
body.
