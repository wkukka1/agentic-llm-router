# MLPROUTERTRAINER-001: the saved artifact carries no hyperparameters

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** medium
- **Class:** `MLPRouterTrainer` ([mlp_router_trainer.py:59-66](../../../../src/training/trainers/mlp_router_trainer.py#L59-L66))
- **Found:** 2026-09-19 (first reported by the earlier diff review; re-checked top-down)
- **Serving-side effect:** [MLPROUTER-001](../../../RouterModel/MLPRouter/MLPROUTER-001-from-artifact-loses-pathway-and-data-config/bug.md)

## What

`train()` fits with `kw = dict(config.hyperparameters)`, which may include `pathway`,
`query_pathway` and `query_features`. It then saves a `RouterModelArtifact` without passing
`hyperparameters=`, and `MLPArtifactPayload` records only `in_dim`, `hidden` and `dropout`.
The stored artifact (and `run.json`) has no record of the pathway or features the MLP was
trained with.

## Failure scenario

A run trained with a non-default pathway can't be reproduced or served correctly from its
artifact; see the linked serving-side bug.
