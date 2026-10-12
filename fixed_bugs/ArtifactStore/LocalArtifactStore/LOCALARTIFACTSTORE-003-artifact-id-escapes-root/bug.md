# LOCALARTIFACTSTORE-003: `artifact_id` is joined onto `root_path` without validation

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** low (local tool; `artifact_id` normally comes from `TrainingConfig.run_name`)
- **Class:** `LocalArtifactStore` ([store.py:61](../../../../src/router/models/store.py#L61), [66](../../../../src/router/models/store.py#L66), [143](../../../../src/router/models/store.py#L143))
- **Found:** 2026-09-19

## What

`self.root_path / artifact_id` is used in `exists`, `load` and `save` with no check.
`Path` joins `"../x"` literally, so it leaves the root; an absolute id replaces the root
entirely.

## Failure scenario

Repro (`known_bugs/_repros/repro_store.py`, section `LOCALARTIFACTSTORE-003`):
`store.save(artifact)` with `artifact_id="../escaped"` writes `escaped/model.pt` next to
the runs directory, not inside it (`outside root: True`). A `run_name` from a config or
CLI argument can therefore write or, with `load`, read outside the runs directory. `load`
only unpickles with `weights_only=True`, so the exposure is file placement, not code
execution.
