# LOCALARTIFACTSTORE-001: an MLP artifact loses `router_model_version` and `training_dataset_version` on reload

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** low-medium
- **Class:** `LocalArtifactStore._load_mlp` ([store.py:109-120](../../../../src/router/models/store.py#L109-L120))
- **Found:** 2026-09-19

## What

`save` writes both fields to `run.json` ([store.py:152-153](../../../../src/router/models/store.py#L152-L153)).
`_load_nirt` reads them back, but `_load_mlp` builds the `RouterModelArtifact` without
`router_model_version` or `training_dataset_version`, so they come back as `""`.

## Failure scenario

Repro (`known_bugs/_repros/repro_store.py`, section `LOCALARTIFACTSTORE-001`):

```
router_model_version       saved='v2'      loaded=''   LOST
training_dataset_version   saved='ds-7'    loaded=''   LOST
(content_hash, schema_version, training_run_id, created_at, hyperparameters: preserved)
```

Any MLP artifact reloaded from disk loses its model version and dataset version, which
are the provenance fields the artifact exists to carry.
