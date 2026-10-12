# MLPROUTERTRAINER-002: `TrainingConfig.preprocessing` is silently ignored

- **Status:** confirmed (by reading)
- **Severity:** low
- **Class:** `MLPRouterTrainer.train` ([mlp_router_trainer.py:41-48](../../../src/training/trainers/mlp_router_trainer.py#L41-L48))
- **Found:** 2026-09-19

## What

`translate_training_config` merges `config.preprocessing` into `cfg["data"]` for the two NIRT
trainers. `MLPRouterTrainer.train` builds `kw` from `hyperparameters`, `seed`, `max_epochs`
and `early_stopping_patience` only. `preprocessing` is never read, and no error or warning
is raised.

## Failure scenario

A caller sets `TrainingConfig(preprocessing={"pathway": "bert"})` and runs the MLP trainer
through the same harness as the others. The setting has no effect: the MLP trains on the
default `pathway="irt"`. The same config works for `NIRTTrainer`, so the difference is
invisible.

## Suggested fix

Map the keys `fit_mlp_router` understands (`pathway`, `query_pathway`, `query_features`)
from `config.preprocessing` into `kw`, and raise on keys it does not understand.
