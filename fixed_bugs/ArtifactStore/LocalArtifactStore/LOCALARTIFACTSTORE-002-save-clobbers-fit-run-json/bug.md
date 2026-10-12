# LOCALARTIFACTSTORE-002: `save` overwrites the training history that `fit()` wrote to `run.json`

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** medium (silent loss of training provenance)
- **Class:** `LocalArtifactStore.save` ([store.py:146-162](../../../../src/router/models/store.py#L146-L162))
- **Found:** 2026-09-19

## What

The module docstring says the store's "leaner `run.json` sidecar ... never conflicts with
the richer one `training.nirt.train.fit` writes". They are the **same file**:
`fit()` writes `<runs_dir>/<name>/run.json` with `git_sha`, `history`, `best_epoch`,
`val_metrics`, `baselines`
([train.py:771-789](../../../../src/training/nirt/train.py#L771-L789)), and `save` writes
`<root>/<artifact_id>/run.json` with only the typed artifact fields. `load_run` not
reading `run.json` only means loading is unaffected; it does nothing to protect the file
on `save`.

## Failure scenario

Repro (`known_bugs/_repros/repro_store.py`, section `LOCALARTIFACTSTORE-002`): a run
directory with a `fit()`-style `run.json` (`git_sha`, `best_epoch`, `history`,
`val_metrics`), then `store.save(artifact)` for the same id. Afterwards `run.json` has
only `artifact_id, content_hash, created_at, hyperparameters, router_model_name,
router_model_version, schema_version, training_dataset_version, training_run_id`. The
history, validation metrics, best epoch and git sha are gone.

Anything that re-saves a loaded NIRT artifact in place (a copy, a version bump, a
metadata fix) destroys the run's only record.
