# TRAININGHARNESS-003: metrics a trainer does not produce default to `0.0`, which is the best possible value

- **Status:** fixed 2026-09-25, see [fix.md](fix.md)
- **Severity:** medium (misleading results, no error)
- **Class:** `TrainingMetrics` / `ValidationMetrics` ([pipeline.py:56-75](../../../src/training/pipeline.py#L56-L75)) and `_extract_metrics`
- **Found:** 2026-09-19

## What

Every metric field defaults to `0.0`, and `_extract_metrics` leaves fields at the default when
the trainer's `last_result` lacks them. For `loss`, `bce` and `regret`, lower is better and
`0.0` is perfect. `MLPRouterTrainer.last_result` is a bare `(model, model_ids)` tuple with
no metrics at all, so a `TrainingRun` for an MLP reports `loss=0.0, bce=0.0, regret=0.0`
for both training and validation, which reads as a flawless model.

`ValidationMetrics.loss` is also never filled for the NIRT trainers: `val_metrics` has no
`loss`/`train_loss` key, so it stays `0.0`, and so does `regret` for `BaselineNIRTTrainer`
(not in its `val_metrics`).

## Failure scenario

A leaderboard or comparison over `TrainingRun`s ranks an unevaluated MLP run above every
evaluated NIRT run on `bce` and `regret`. Nothing distinguishes "not measured" from
"measured zero".
