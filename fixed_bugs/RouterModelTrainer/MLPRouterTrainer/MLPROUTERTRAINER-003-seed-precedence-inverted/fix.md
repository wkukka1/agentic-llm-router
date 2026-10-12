# MLPROUTERTRAINER-003 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-25, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`MLPRouterTrainer.train` sets `kw["seed"] = config.seed` unconditionally, like the sibling trainers. Test: `test_config_seed_always_wins_over_a_conflicting_hyperparameters_seed`.

## Fix suggested when the bug was filed

Match the sibling trainers: `kw["seed"] = config.seed` unconditionally
(or document why `MLPRouterTrainer` intentionally lets a manual
`hyperparameters["seed"]` win, if that's actually wanted).

## Regression tests

`tests/training/trainers/test_mlp_router_trainer.py:113` `test_config_seed_always_wins_over_a_conflicting_hyperparameters_seed`

```python
def test_config_seed_always_wins_over_a_conflicting_hyperparameters_seed(tmp_path, monkeypatch):
    """MLPROUTERTRAINER-003: seed precedence must match the sibling trainers
    (NIRTTrainer/BaselineNIRTTrainer) -- TrainingConfig.seed wins even when
    hyperparameters also sets "seed" to something else."""
    import training.trainers.mlp_router_trainer as mod

    captured = {}
    real_fit = mod.fit_mlp_router

    def spy(data, **kw):
        captured.update(kw)
        return real_fit(data, **kw)

    monkeypatch.setattr(mod, "fit_mlp_router", spy)

    d = FakeTrainingData(make_split_obs(seed=18), query_dim=16)
    trainer = MLPRouterTrainer(runs_dir=tmp_path)
    config = TrainingConfig(
        run_name="mlp-run-seed", seed=123,
        hyperparameters={"hidden": 16, "epochs": 2, "seed": 999},
    )
    trainer.train(d, config)

    assert captured["seed"] == 123
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/training/trainers/mlp_router_trainer.py`

```diff
--- a/src/training/trainers/mlp_router_trainer.py
+++ b/src/training/trainers/mlp_router_trainer.py
@@ -39,7 +39,7 @@ class MLPRouterTrainer(RouterModelTrainer):
         """``dataset`` is the ``data`` object ``fit_mlp_router`` needs (a
         ``TrainingData`` facade or equivalent)."""
         kw = dict(config.hyperparameters)
-        kw.setdefault("seed", config.seed)
+        kw["seed"] = config.seed
         if config.max_epochs is not None:
             kw["epochs"] = config.max_epochs
         if config.early_stopping_patience is not None:
```
