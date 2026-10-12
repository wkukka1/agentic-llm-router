# MOD-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-25, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`_resolve_runs_dir` takes an explicit `runs_dir` override, so `fit()` joins a relative `runs_dir` against the phase-0 root (or `REPO_ROOT`); `NIRTTrainer` now uses the same call so save and load agree. Tests: `tests/training/nirt/test_train_runs_dir.py`, `test_trainer.py::test_relative_runs_dir_saves_and_loads_consistently`.

## Fix suggested when the bug was filed

Make NIRT's `fit()` join an explicit relative `runs_dir` against
`phase0_cfg`'s root the same way the baseline's `fit()` does for `out_dir`
(or, if CWD-relative is intentional for NIRT, document the asymmetry and why
the two aren't meant to match).

## Regression tests

`tests/training/nirt/test_trainer.py:75` `test_relative_runs_dir_saves_and_loads_consistently`

```python
def test_relative_runs_dir_saves_and_loads_consistently(tmp_path, monkeypatch):
    """MOD-001: when NIRTTrainer is given an explicit relative runs_dir,
    fit() and LocalArtifactStore must resolve it the same way (against
    REPO_ROOT), not CWD. This test verifies that save/load are consistent."""
    import router.config as router_config

    # Monkeypatch REPO_ROOT so the relative path resolves to tmp_path
    monkeypatch.setattr(router_config, "REPO_ROOT", tmp_path)

    train_obs, val_obs, q_store, m_store = make_synthetic_irt(n_queries=200, seed=16)
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)

    # Use a relative runs_dir
    trainer = NIRTTrainer(runs_dir="rel_runs")
    config = TrainingConfig(run_name="trainer-rel-dir", hyperparameters=nirt_cfg())
    artifact = trainer.train((train_ds, val_ds), config)

    # If save/load paths don't match, load() would fail or load a stale run
    assert artifact.artifact_id == "trainer-rel-dir"
    expected = tmp_path / "rel_runs" / "trainer-rel-dir"
    assert expected.exists()
    assert trainer.last_result.path == expected
```

All of `tests/training/nirt/test_train_runs_dir.py` (new module).

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/training/nirt/train.py`

```diff
--- a/src/training/nirt/train.py
+++ b/src/training/nirt/train.py
@@ -741,7 +741,7 @@ def fit(
     )
 
     if save:
-        base = Path(runs_dir) if runs_dir is not None else _resolve_runs_dir(nirt_cfg, phase0_cfg)
+        base = _resolve_runs_dir(nirt_cfg, phase0_cfg, runs_dir)
         out = base / name
         out.mkdir(parents=True, exist_ok=True)
         # record which phase-0 config (paths, stores, pool) the run was trained
@@ -794,12 +794,17 @@ def fit(
     return result
 
 
-def _resolve_runs_dir(nirt_cfg: dict, phase0_cfg) -> Path:
+def _resolve_runs_dir(nirt_cfg: dict, phase0_cfg, runs_dir: "str | os.PathLike | None" = None) -> Path:
     """Absolute, else joined to ``phase0_cfg``'s root (if given) or
     :data:`router.config.REPO_ROOT`. Both branches ultimately delegate to
     :func:`router.config.resolve_path` (XD-11) -- ``Config.resolve`` already
-    handles the absolute-path case, so it's checked only once."""
-    rel = nirt_cfg.get("runs_dir", DEFAULT_NIRT_RUNS_DIR)
+    handles the absolute-path case, so it's checked only once. ``runs_dir``,
+    when given, overrides ``nirt_cfg["runs_dir"]`` as the value to resolve --
+    this is what makes an *explicit* relative ``runs_dir`` join against
+    ``phase0_cfg``'s root the same way the baseline trainer's explicit
+    relative ``out_dir`` does, instead of resolving against the process's CWD
+    (MOD-001, training.nirt)."""
+    rel = runs_dir if runs_dir is not None else nirt_cfg.get("runs_dir", DEFAULT_NIRT_RUNS_DIR)
     if phase0_cfg is not None and hasattr(phase0_cfg, "resolve"):
         return phase0_cfg.resolve(rel)
     from router.config import resolve_path
```

`src/training/nirt/trainer.py`

```diff
--- a/src/training/nirt/trainer.py
+++ b/src/training/nirt/trainer.py
@@ -19,7 +19,7 @@ from router.models.artifacts import RouterModelArtifact
 from router.models.store import LocalArtifactStore
 
 from ..trainers.base import RouterModelTrainer, TrainingConfig, translate_training_config
-from .train import RunResult, fit
+from .train import RunResult, fit, _resolve_runs_dir
 
 
 class NIRTTrainer(RouterModelTrainer):
@@ -38,12 +38,8 @@ class NIRTTrainer(RouterModelTrainer):
             nirt_cfg, datasets=dataset, name=config.run_name,
             runs_dir=self.runs_dir, save=True, verbose=False,
         )
-        # LocalArtifactStore needs a concrete path; resolve runs_dir=None the
-        # exact same way fit() itself just did (_resolve_runs_dir ->
-        # resolve_path), including honoring a nirt_cfg["runs_dir"] override.
-        runs_dir = self.runs_dir
-        if runs_dir is None:
-            from router.config import DEFAULT_NIRT_RUNS_DIR, resolve_path
-
-            runs_dir = resolve_path(nirt_cfg.get("runs_dir", DEFAULT_NIRT_RUNS_DIR))
+        # LocalArtifactStore needs a concrete path; resolve the same way fit()
+        # itself just did (_resolve_runs_dir with phase0_cfg=None), including
+        # honoring both an explicit runs_dir and a nirt_cfg["runs_dir"] override.
+        runs_dir = _resolve_runs_dir(nirt_cfg, None, self.runs_dir)
         return LocalArtifactStore(runs_dir).load(config.run_name)
```
