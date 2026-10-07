# MLPROUTERTRAINER-003: `config.seed` only fills in a gap instead of overriding hyperparameters

- **Status:** fixed 2026-09-25, see [fix.md](fix.md)
- **Severity:** medium
- **Class:** `MLPRouterTrainer` ([training/trainers/mlp_router_trainer.py:41-42](../../../../src/training/trainers/mlp_router_trainer.py#L41-L42))
- **Found:** 2026-09-22

## What

```python
kw = dict(config.hyperparameters)
kw.setdefault("seed", config.seed)
```

`setdefault` means an explicit `"seed"` already present in
`config.hyperparameters` wins over `config.seed`. The sibling trainers
(`NIRTTrainer`, `BaselineNIRTTrainer`, via the shared
`translate_training_config`) do the opposite — `cfg["seed"] = config.seed`
unconditionally, so `TrainingConfig.seed` is always the authoritative value.
`MLPRouterTrainer` inverts that precedence relative to its siblings.

## Failure scenario

A caller sets `TrainingConfig(seed=123, hyperparameters={"seed": 999, ...})`
expecting `seed` (the dedicated, documented field) to win, as it does for
every other trainer. `MLPRouterTrainer` silently trains with `999` instead.
Reproducibility tooling that always sets `TrainingConfig.seed` and leaves
`hyperparameters` as a passthrough dict gets inconsistent behavior depending
on which trainer is used.
