# TRAININGHARNESS-002: training metrics come from the last epoch, validation metrics from the best epoch

- **Status:** confirmed (by reading `_extract_metrics` and both `fit()` functions)
- **Severity:** low
- **Class:** `TrainingHarness` helper `_extract_metrics` ([pipeline.py:129-131](../../src/training/pipeline.py#L129-L131))
- **Found:** 2026-09-19 (first reported by the earlier diff review; re-checked top-down)

## What

`_extract_metrics` takes `history[-1]` for `TrainingMetrics`. In both wrapped `fit()`
functions the model is restored to the **best-epoch** weights after the loop
([nirt/train.py:721-722](../../src/training/nirt/train.py#L721-L722),
[baseline/train.py:219-220](../../src/training/nirt/baseline/train.py#L219-L220)), and
`val_metrics` is computed from those restored weights. So `validation_metrics` describes the
returned model, but `training_metrics.loss` is the train loss of the last epoch, which
early stopping ran `patience` epochs past the best.

## Failure scenario

A run with `best_epoch=12` that stopped at epoch 18 reports the validation score of epoch
12's weights next to the train loss of epoch 18. The train/validation gap read off a
`TrainingRun` is measured across two different models.

## Suggested fix

Select the history row whose `epoch == last_result.best_epoch` (both result types carry
`best_epoch`), falling back to `history[-1]` only when it is `-1`.
