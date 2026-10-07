# ROUTERMODELARTIFACT-001: `content_hash` is a required field that nothing ever computes

- **Status:** confirmed (grep over `src/`)
- **Severity:** low-medium (the identity/integrity guarantee the type advertises does not exist)
- **Class:** `RouterModelArtifact` ([artifacts.py:163](../../src/router/models/artifacts.py#L163))
- **Found:** 2026-09-19

## What

The module docstring says the artifact is "enough to resolve a live decision back to an
exact checkpoint". `content_hash` is the only field that could pin the *bytes*, and it is
mandatory (no default). But every writer passes `""`:

- [mlp_router_trainer.py:62](../../src/training/trainers/mlp_router_trainer.py#L62) `content_hash=""`
- [baseline/trainer.py:59](../../src/training/nirt/baseline/trainer.py#L59) `content_hash=""`
- `LocalArtifactStore.save` writes whatever it is given; `_load_nirt` reads it back from a
  `run.json` that `fit()` never populates with one, so it defaults to `""`.

No code computes a hash of `model.pt`, and nothing verifies one on load.

## Failure scenario

Two different checkpoints saved under the same `artifact_id` (a re-run overwriting a run
directory) are indistinguishable by their artifact records. A decision that references
`artifact_id` cannot be tied to the exact weights that produced it, and a truncated or
swapped `model.pt` loads without any integrity check.

## Suggested fix

Compute a SHA-256 of the serialized payload in `LocalArtifactStore.save` (and fill it in on
`_load_*` from the file if `run.json` has none), then verify it on `load`. Give the field
a default of `""` in the dataclass so trainers are not forced to pass a placeholder.
