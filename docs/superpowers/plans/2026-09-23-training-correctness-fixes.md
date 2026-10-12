# Training-side correctness fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three open, `med`-severity `known_bugs/` defects in the `training/` package: `TrainingHarness`'s metric dataclasses default unreported metrics to `0.0` (the *best* possible value for loss/BCE/regret, so an unevaluated run reads as flawless), `MLPRouterTrainer` inverts seed precedence relative to its sibling trainers, and `training.nirt.train.fit`'s explicit relative `runs_dir` resolves against the process CWD instead of the phase-0 config root the way the baseline trainer's `out_dir` does.

**Architecture:** Three independent, single-file patches: `training/pipeline.py` (`TrainingMetrics`/`ValidationMetrics`), `training/trainers/mlp_router_trainer.py` (`MLPRouterTrainer.train`), `training/nirt/train.py` (`_resolve_runs_dir`/`fit`). No dependency between them — any order.

**Tech Stack:** Python 3.12, pytest, PyTorch (`pytest.importorskip("torch")`, already required by the affected test files).

**Spec:** `fixed_bugs/TrainingHarness/TRAININGHARNESS-003-unpopulated-metrics-read-as-perfect/bug.md`, `fixed_bugs/RouterModelTrainer/MLPRouterTrainer/MLPROUTERTRAINER-003-seed-precedence-inverted/bug.md`, `fixed_bugs/_module_level/training.nirt/MOD-001-runs-dir-resolution-asymmetric-vs-baseline/bug.md`.

## Global Constraints

- No `src/` code outside `training/pipeline.py` reads `TrainingMetrics`/`ValidationMetrics` fields today (verified: `grep -rn "training_metrics\.\|validation_metrics\." src/` finds only `training/pipeline.py`'s own construction) — the `Optional[float] = None` default change in Task 1 is safe with no other production caller to update. `training_time` (`TrainingMetrics` only) is always harness-set (`training.training_time = elapsed`, `training/pipeline.py:146`) and stays a plain `float = 0.0`, not `Optional`.
- `tests/training/test_pipeline.py::test_harness_train_runs_mlp_router` (line ~100) currently asserts `run.validation_metrics.bce == 0.0` with the comment "MLPRouterTrainer.last_result has no val_metrics/history -> zero defaults, not a crash" — that assertion encodes today's *bug*, not a requirement. Task 1 must update it to `is None` with a corrected comment, not just leave it to start failing.
- `MLPRouterTrainer`'s existing test `tests/training/trainers/test_mlp_router_trainer.py::test_typed_config_fields_override_hyperparameters` (`captured["seed"] == 3` with `TrainingConfig(seed=3, ...)` and no `"seed"` key in `hyperparameters`) must keep passing unchanged after Task 2 — it doesn't exercise the precedence-inversion case (no conflicting `hyperparameters["seed"]`), so `kw["seed"] = config.seed` and the old `kw.setdefault(...)` behave identically for it.
- Task 3 must not change `training.nirt.train.fit`'s *default* `runs_dir=None` behavior (already correct, already resolves against `phase0_cfg` via `_resolve_runs_dir`) — only the explicit-relative-path branch changes.

## Review Focus

- A metric a trainer *does* report (e.g. NIRT's validation `bce`, already exercised by `tests/training/test_pipeline.py::test_harness_train_runs_nirt_and_loads_through_the_factory`) must still come through as a real float, not `None` — the fix only changes the *default* for fields the trainer's `last_result` has no key for. (Task 1)
- `TrainingConfig.seed`'s docstring/existing contract (the "dedicated, documented field" the bug file references) must win even when `hyperparameters` also sets `"seed"` — add a test with both set to different values, not just one or the other. (Task 2)
- An **absolute** explicit `runs_dir` must still be used as-is (unchanged from today) — only a *relative* explicit `runs_dir` should start resolving against `phase0_cfg`'s root instead of the CWD. (Task 3)
- A `runs_dir=None` default call (the common case) must resolve identically before and after the fix — `_resolve_runs_dir`'s existing default-branch tests (if any) or `training.nirt` runs relying on `configs/phase0.yaml`'s ambient root must not shift. (Task 3)
- `phase0_cfg=None` (no phase-0 config object at all) must still work for an explicit relative `runs_dir`, falling back to `router.config.resolve_path`'s own default root (`REPO_ROOT`) the same way `_resolve_runs_dir`'s existing `else` branch already does for the default-`runs_dir` case. (Task 3)

---

## Task 1: Unreported training/validation metrics default to `None`, not `0.0`

**Files:**
- Modify: `src/training/pipeline.py` (`TrainingMetrics`, `ValidationMetrics`, ~L56-75)
- Test: `tests/training/test_pipeline.py`

**Interfaces:**
- Consumes: `_metrics_kwargs(d: dict) -> dict[str, float]` (`training/pipeline.py:100-114`, unchanged — it already only includes keys actually present in `d`).
- Produces: `TrainingMetrics.loss/accuracy/bce/auc/spearman/regret: Optional[float] = None` (were `float = 0.0`); `ValidationMetrics.loss/accuracy/bce/auc/spearman/regret: Optional[float] = None` (same). `TrainingMetrics.training_time` and `ValidationMetrics.selected_parameters` are unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/training/test_pipeline.py`:

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

Also update the existing `test_harness_train_runs_mlp_router` (same file, ~line 100):

```python
    # MLPRouterTrainer.last_result has no val_metrics/history -> the metric
    # stays unset (None), not a fake 0.0 "perfect score" (TRAININGHARNESS-003)
    assert run.validation_metrics.bce is None
```

(replacing the current `assert run.validation_metrics.bce == 0.0` line and its old comment).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_pipeline.py::test_unreported_metrics_default_to_none_not_zero tests/training/test_pipeline.py::test_harness_train_runs_mlp_router -v`
Expected: `test_unreported_metrics_default_to_none_not_zero` FAILS (`run.validation_metrics.bce == 0.0`, not `None`). `test_harness_train_runs_mlp_router` FAILS too, since you already edited its assertion to `is None` in Step 1 against the still-unfixed dataclass.

- [ ] **Step 3: Implement**

In `src/training/pipeline.py`:

```python
@dataclass
class TrainingMetrics:
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    bce: Optional[float] = None
    auc: Optional[float] = None
    spearman: Optional[float] = None
    regret: Optional[float] = None
    training_time: float = 0.0


@dataclass
class ValidationMetrics:
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    bce: Optional[float] = None
    auc: Optional[float] = None
    spearman: Optional[float] = None
    regret: Optional[float] = None
    selected_parameters: dict[str, Any] = field(default_factory=dict)
```

(`Optional` is already imported at the top of `training/pipeline.py` — `from typing import Any, Optional`.)

Also update the module docstring's "Metrics translation is necessarily best-effort" paragraph (~L8-17) to say fields are left `None`, not "left at their zero defaults", so the documented behavior matches:

```
``TrainingMetrics``/``ValidationMetrics`` below are populated from whatever
keys are present and left ``None`` otherwise -- not every trainer has an
equivalent of every field (none compute ``regret`` today; that lives in
``evaluation.routing.oracle``, a separate, label-needing concern). ``None``
means "not reported by this trainer", never "reported as zero"
(TRAININGHARNESS-003).
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_pipeline.py -v`
Expected: PASS, all tests in the file, including `test_harness_train_runs_nirt_and_loads_through_the_factory` (its `np.isfinite(run.validation_metrics.bce) and run.validation_metrics.bce != 0.0` assertion still holds — NIRT's `last_result.val_metrics` really does carry `bce`, so that field is still populated as a real float, not `None`).

- [ ] **Step 5: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/training/pipeline.py tests/training/test_pipeline.py
git commit -m "fix(training): unreported TrainingMetrics/ValidationMetrics fields default to None, not 0.0 (TRAININGHARNESS-003)"
```

---

## Task 2: `MLPRouterTrainer`'s `config.seed` always wins over `hyperparameters["seed"]`

**Files:**
- Modify: `src/training/trainers/mlp_router_trainer.py` (`MLPRouterTrainer.train`, ~L38-42)
- Test: `tests/training/trainers/test_mlp_router_trainer.py`

**Interfaces:**
- Consumes: `TrainingConfig.seed: int` (`training/trainers/base.py`, unchanged).
- Produces: `MLPRouterTrainer.train`'s `kw["seed"]` is always `config.seed`, matching `NIRTTrainer`/`BaselineNIRTTrainer`'s shared `translate_training_config` precedence (`cfg["seed"] = config.seed` unconditionally) instead of `kw.setdefault("seed", config.seed)` (which let an explicit `hyperparameters["seed"]` win).

- [ ] **Step 1: Write the failing test**

Add to `tests/training/trainers/test_mlp_router_trainer.py`, next to `test_typed_config_fields_override_hyperparameters`:

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

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/training/trainers/test_mlp_router_trainer.py::test_config_seed_always_wins_over_a_conflicting_hyperparameters_seed -v`
Expected: FAIL — `captured["seed"] == 999`, not `123`.

- [ ] **Step 3: Implement**

In `src/training/trainers/mlp_router_trainer.py`, `train()`:

```python
        kw = dict(config.hyperparameters)
        kw["seed"] = config.seed
```

(replacing `kw.setdefault("seed", config.seed)`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/trainers/test_mlp_router_trainer.py -v`
Expected: PASS, all tests in the file, including the pre-existing `test_typed_config_fields_override_hyperparameters` (`captured["seed"] == 3` — no conflicting `hyperparameters["seed"]` there, so both the old and new precedence give the same result).

- [ ] **Step 5: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/training/trainers/mlp_router_trainer.py tests/training/trainers/test_mlp_router_trainer.py
git commit -m "fix(training): MLPRouterTrainer.config.seed always wins over hyperparameters[\"seed\"] (MLPROUTERTRAINER-003)"
```

---

## Task 3: `training.nirt.train.fit`'s explicit relative `runs_dir` resolves against `phase0_cfg`'s root

**Files:**
- Modify: `src/training/nirt/train.py` (`_resolve_runs_dir` ~L797-807, its call site ~L744)
- Test: new file `tests/training/nirt/test_train_runs_dir.py`

**Interfaces:**
- Consumes: `router.config.Config.resolve(rel) -> Path` (`router/config.py:91-92`, unchanged — "absolute, else joined to `self._root`").
- Produces: `_resolve_runs_dir(nirt_cfg: dict, phase0_cfg, runs_dir: str | os.PathLike | None = None) -> Path` gains a third parameter: when given, it overrides `nirt_cfg.get("runs_dir", ...)` as the value to resolve, going through the exact same `phase0_cfg.resolve(...)` (or `resolve_path` fallback) as the default-branch already does. `fit()`'s call site becomes unconditional: `base = _resolve_runs_dir(nirt_cfg, phase0_cfg, runs_dir)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/training/nirt/test_train_runs_dir.py`:

```python
"""MOD-001 (training.nirt): an explicit relative runs_dir must resolve the
same way the baseline trainer's explicit relative out_dir does -- against
phase0_cfg's root, not the process's current working directory."""

from __future__ import annotations

from router.config import Config
from training.nirt.train import _resolve_runs_dir


def test_explicit_relative_runs_dir_joins_against_phase0_root(tmp_path):
    phase0_cfg = Config({}, root=tmp_path)

    resolved = _resolve_runs_dir({}, phase0_cfg, runs_dir="my_runs")

    assert resolved == tmp_path / "my_runs"


def test_explicit_absolute_runs_dir_is_left_untouched(tmp_path):
    absolute = tmp_path / "abs_runs"
    phase0_cfg = Config({}, root=tmp_path / "elsewhere")

    resolved = _resolve_runs_dir({}, phase0_cfg, runs_dir=absolute)

    assert resolved == absolute


def test_no_explicit_runs_dir_falls_back_to_nirt_cfg_unchanged(tmp_path):
    """The pre-existing default-branch behaviour must not shift."""
    phase0_cfg = Config({}, root=tmp_path)

    resolved = _resolve_runs_dir({"runs_dir": "cfg_runs"}, phase0_cfg)

    assert resolved == tmp_path / "cfg_runs"


def test_explicit_relative_runs_dir_with_no_phase0_cfg_falls_back_to_repo_root():
    from router.config import REPO_ROOT

    resolved = _resolve_runs_dir({}, None, runs_dir="my_runs")

    assert resolved == REPO_ROOT / "my_runs"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/nirt/test_train_runs_dir.py -v`
Expected: `test_explicit_relative_runs_dir_joins_against_phase0_root` FAILS with `TypeError: _resolve_runs_dir() got an unexpected keyword argument 'runs_dir'`. The other three currently-passing-by-accident or not-yet-callable tests fail the same way (the signature doesn't accept `runs_dir` yet at all).

- [ ] **Step 3: Implement**

In `src/training/nirt/train.py`, `_resolve_runs_dir`:

```python
def _resolve_runs_dir(nirt_cfg: dict, phase0_cfg, runs_dir: "str | os.PathLike | None" = None) -> Path:
    """Absolute, else joined to ``phase0_cfg``'s root (if given) or
    :data:`router.config.REPO_ROOT`. Both branches ultimately delegate to
    :func:`router.config.resolve_path` (XD-11) -- ``Config.resolve`` already
    handles the absolute-path case, so it's checked only once. ``runs_dir``,
    when given, overrides ``nirt_cfg["runs_dir"]`` as the value to resolve --
    this is what makes an *explicit* relative ``runs_dir`` join against
    ``phase0_cfg``'s root the same way the baseline trainer's explicit
    relative ``out_dir`` does, instead of resolving against the process's CWD
    (MOD-001, training.nirt)."""
    rel = runs_dir if runs_dir is not None else nirt_cfg.get("runs_dir", DEFAULT_NIRT_RUNS_DIR)
    if phase0_cfg is not None and hasattr(phase0_cfg, "resolve"):
        return phase0_cfg.resolve(rel)
    from router.config import resolve_path

    return resolve_path(rel)
```

(`training/nirt/train.py` already does `import os` at the top — the `os.PathLike` type hint needs nothing further.)

In `fit()` (~L744), replace the conditional call site:

```python
    if save:
        base = _resolve_runs_dir(nirt_cfg, phase0_cfg, runs_dir)
```

(replacing `base = Path(runs_dir) if runs_dir is not None else _resolve_runs_dir(nirt_cfg, phase0_cfg)`; drop the now-unused `Path(runs_dir)` branch — `_resolve_runs_dir` handles every case).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/nirt/test_train_runs_dir.py -v`
Expected: PASS, all four tests.

- [ ] **Step 5: Run the training.nirt suite plus the full suite (fit()'s call site changed, so confirm nothing that constructs a NIRTTrainer/calls fit() with an explicit runs_dir regresses)**

Run: `.venv/Scripts/python.exe -m pytest tests/training/nirt tests/training/trainers -v`
Expected: PASS, all tests, including `tests/training/nirt/test_trainer.py::test_runs_dir_none_falls_back_to_the_default_runs_dir` (the `runs_dir=None` path is untouched by this fix).

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/training/nirt/train.py tests/training/nirt/test_train_runs_dir.py
git commit -m "fix(training): nirt.train.fit resolves an explicit relative runs_dir against phase0_cfg's root (MOD-001 training.nirt)"
```

---

## Final step: update `known_bugs/`

Update status to `fixed` in:
- `fixed_bugs/TrainingHarness/TRAININGHARNESS-003-unpopulated-metrics-read-as-perfect/bug.md`
- `fixed_bugs/RouterModelTrainer/MLPRouterTrainer/MLPROUTERTRAINER-003-seed-precedence-inverted/bug.md`
- `fixed_bugs/_module_level/training.nirt/MOD-001-runs-dir-resolution-asymmetric-vs-baseline/bug.md`

And their rows in `known_bugs/README.md`'s `## Bugs` table.
