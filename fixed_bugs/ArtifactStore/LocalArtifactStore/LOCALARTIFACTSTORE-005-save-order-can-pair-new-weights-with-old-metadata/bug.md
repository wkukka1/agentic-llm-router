# LOCALARTIFACTSTORE-005: re-saving over an existing artifact is not atomic as a pair

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** low
- **Class:** `LocalArtifactStore.save` ([store.py:145-162](../../../../src/router/models/store.py#L145-L162))
- **Found:** 2026-09-19

## What

Each file is replaced atomically, but the pair is not. The comment says writing `run.json`
last means "a reader never sees a complete `model.pt` paired with a missing/stale
`run.json`". That covers the first save of a new id. It does not cover a **re-save over an
existing artifact**: `model.pt` is replaced first, so for the window between the two
`os.replace` calls a concurrent `load` gets the new weights with the old `run.json`
(`content_hash`, `hyperparameters`, `training_run_id` from the previous save).

A crash between the two replaces leaves the same mismatch permanently.

## Failure scenario

A serving process reloading an artifact while a trainer re-saves under the same id
builds a router from new weights but records the old hyperparameters and hash.
