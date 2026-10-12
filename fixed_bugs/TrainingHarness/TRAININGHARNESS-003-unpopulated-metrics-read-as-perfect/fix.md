# TRAININGHARNESS-003 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-25, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`TrainingMetrics`/`ValidationMetrics` metric fields default to `None` (not reported) instead of `0.0`; `training_time` unchanged. Tests: `tests/training/test_pipeline.py::test_unreported_metrics_default_to_none_not_zero`, updated `test_harness_train_runs_mlp_router`.

## Fix suggested when the bug was filed

Default the metric fields to `Optional[float] = None` (or `float("nan")`) and leave them
unset when the trainer does not report them.

## Regression tests

`tests/training/test_pipeline.py:114` `test_unreported_metrics_default_to_none_not_zero`

```python
def test_unreported_metrics_default_to_none_not_zero(tmp_path):
    """TRAININGHARNESS-003: a trainer whose last_result has no metrics at all
    (MLPRouterTrainer's is a bare (model, model_ids) tuple) must leave those
    fields unset, not silently read as a perfect 0.0 score."""
    d = FakeTrainingData(make_split_obs(seed=25), query_dim=16)
    harness = TrainingHarness()
    trainer = MLPRouterTrainer(runs_dir=tmp_path)
    config = TrainingConfig(
        run_name="harness-mlp-none", hyperparameters={"hidden": 16, "epochs": 2, "pathway": "irt"},
    )
    run = harness.train(trainer, d, config)

    assert run.validation_metrics.bce is None
    assert run.validation_metrics.regret is None
    assert run.training_metrics.loss is None
    # training_time is harness-set regardless of the trainer, so it's a real
    # number, not part of this "unreported -> None" behaviour
    assert run.training_metrics.training_time > 0
```

`tests/training/test_pipeline.py:86` `test_harness_train_runs_mlp_router`

```python
def test_harness_train_runs_mlp_router(tmp_path):
    d = FakeTrainingData(make_split_obs(seed=23), query_dim=16)

    harness = TrainingHarness()
    trainer = MLPRouterTrainer(runs_dir=tmp_path)
    config = TrainingConfig(
        run_name="harness-mlp", hyperparameters={"hidden": 16, "epochs": 3, "pathway": "irt"},
    )
    run = harness.train(trainer, d, config)

    assert run.trainer_name == "mlp"
    assert run.artifact.router_model_name == "mlp"
    # MLPRouterTrainer.last_result has no val_metrics/history -> the metric
    # stays unset (None), not a fake 0.0 "perfect score" (TRAININGHARNESS-003)
    assert run.validation_metrics.bce is None
    assert run.training_metrics.training_time > 0

    # the artifact-driven load path this trainer exists to enable (TT-01) --
    # must not raise TypeError("no from_artifact")
    from router.models.registry import RouterModelFactory
    from router.routing.routers import MLPRouter

    factory = RouterModelFactory()
    factory.register("mlp", MLPRouter)
    router = factory.create(run.artifact, CostModel())
    assert isinstance(router, MLPRouter)
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/training/pipeline.py`

```diff
--- a/src/training/pipeline.py
+++ b/src/training/pipeline.py
@@ -55,23 +56,23 @@ from .trainers.base import RouterModelTrainer, TrainingConfig
 
 @dataclass
 class TrainingMetrics:
-    loss: float = 0.0
-    accuracy: float = 0.0
-    bce: float = 0.0
-    auc: float = 0.0
-    spearman: float = 0.0
-    regret: float = 0.0
+    loss: Optional[float] = None
+    accuracy: Optional[float] = None
+    bce: Optional[float] = None
+    auc: Optional[float] = None
+    spearman: Optional[float] = None
+    regret: Optional[float] = None
     training_time: float = 0.0
 
 
 @dataclass
 class ValidationMetrics:
-    loss: float = 0.0
-    accuracy: float = 0.0
-    bce: float = 0.0
-    auc: float = 0.0
-    spearman: float = 0.0
-    regret: float = 0.0
+    loss: Optional[float] = None
+    accuracy: Optional[float] = None
+    bce: Optional[float] = None
+    auc: Optional[float] = None
+    spearman: Optional[float] = None
+    regret: Optional[float] = None
     selected_parameters: dict[str, Any] = field(default_factory=dict)
 
 
@@ -116,13 +117,13 @@ def _metrics_kwargs(d: dict[str, Any]) -> dict[str, float]:
 
 def _extract_metrics(last_result: Any) -> tuple[TrainingMetrics, ValidationMetrics]:
     """Best-effort: works for anything exposing ``val_metrics``/``history``
-    dicts (``RunResult``, ``BaselineRun``); returns zero-valued defaults for
-    anything else (e.g. ``MLPRouterTrainer``'s bare tuple).
+    dicts (``RunResult``, ``BaselineRun``); leaves every field ``None``
+    ("not reported") for anything else (e.g. ``MLPRouterTrainer``'s bare tuple).
 
     ``history`` rows only ever carry ``train_loss`` (matched onto
     ``TrainingMetrics.loss``) -- neither wrapped trainer's per-epoch loop
-    computes train-split accuracy/bce/auc/spearman, so those four fields stay
-    at their ``0.0`` default regardless of trainer; only the validation split
+    computes train-split accuracy/bce/auc/spearman, so those four fields are
+    left ``None`` ("not reported") regardless of trainer; only the validation split
     (``val_metrics``) has them."""
     val_metrics = getattr(last_result, "val_metrics", None) or {}
     validation = ValidationMetrics(**_metrics_kwargs(val_metrics))
```
