# LOCALARTIFACTSTORE-004: `load` deserializes `model.pt` two to three times

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** low (performance and peak memory)
- **Class:** `LocalArtifactStore.load` ([store.py:63-70](../../../../src/router/models/store.py#L63-L70))
- **Found:** 2026-09-19

## What

`load` calls `torch.load` on the whole `model.pt` only to read the `"format"` tag. It then
dispatches to `_load_mlp`, which calls `torch.load` again, or to `_load_nirt`, which goes
through `load_run`, which calls `torch.load` again.

## Failure scenario

Every load decodes the full state dict twice. `NIRTTrainer` and `MLPRouterTrainer` both call
`store.load()` immediately after training, so each training run pays the extra read and the
transient memory for the checkpoint size.
