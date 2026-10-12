# MLPROUTERTRAINER-004: Architecture introspection assumes a fixed `nn.Sequential` layout

- **Status:** confirmed (traced; code_review finding TT-03)
- **Severity:** low-medium (latent — breaks silently or loudly only if the model shape changes)
- **Class:** `MLPRouterTrainer` ([training/trainers/mlp_router_trainer.py:54-56](../../../src/training/trainers/mlp_router_trainer.py#L54-L56))
- **Found:** 2026-09-22

## What

```python
in_dim=int(model[0].in_features),
hidden=int(model[0].out_features),
dropout=float(model[2].p),
```

`MLPArtifactPayload`'s hyperparameters are read back by indexing the trained
`nn.Sequential` at fixed positions (`model[0]` = first `Linear`, `model[2]` =
the `Dropout`). This only works because `fit_mlp_router`'s current
architecture happens to be `[Linear, Activation, Dropout, ...]` in that exact
order.

## Failure scenario

Any change to `fit_mlp_router`'s layer ordering (adding a normalization
layer before the first `Linear`, reordering activation/dropout, adding
another `Linear` at the front) either raises an `AttributeError` on the
wrong layer type or — worse — silently reads the wrong layer's dimensions
into the persisted artifact, with no type check to catch it.

## Suggested fix

Have `fit_mlp_router` return its hyperparameters explicitly (the values it
already knows when constructing the model) instead of asking the trainer to
reverse-engineer them from the module's positional structure.
