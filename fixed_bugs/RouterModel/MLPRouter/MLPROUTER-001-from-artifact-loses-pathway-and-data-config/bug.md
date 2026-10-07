# MLPROUTER-001: `MLPRouter.from_artifact` rebuilds with default pathway and default data config

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** medium
- **Class:** `MLPRouter` ([routers.py:459-492](../../../../src/router/routing/routers.py#L459-L492))
- **Found:** 2026-09-19 (first reported by the earlier diff review; re-checked top-down)
- **Root cause on the training side:** [MLPROUTERTRAINER-001](../../../RouterModelTrainer/MLPRouterTrainer/MLPROUTERTRAINER-001-hyperparameters-not-persisted/bug.md)

## What

`from_artifact` calls `cls(model, payload.model_ids, data=data, name=...)`. That means:

1. `pathway` is the constructor default `"irt"`. `MLPArtifactPayload` records only
   `in_dim`/`hidden`/`dropout`, so the serving side cannot know which pathway the MLP was
   trained on.
2. `data=None` goes to `load_training_data_for_router(None)`, which loads
   `configs/phase0.yaml`. `NIRTRouter.from_artifact` instead reads `phase0_config` out of
   the stored config; `MLPRouter` has nothing stored to read.
3. `MLPRouter` has no `query_pathway` / `query_features` parameters at all, although
   `fit_mlp_router` accepts both. A model trained with either can't be served faithfully.

## Failure scenario

Train with `pathway="bert"`; save; reload through `RouterModelFactory`. The router encodes
queries from the `"irt"` store: either a wrong-width input (`load_state_dict`/matmul
error) or, if the widths happen to match, scores from the wrong embedding space with no
warning.
