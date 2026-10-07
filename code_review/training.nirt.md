# Code review: training.nirt

**Files reviewed:** 2 (145 lines): `src/training/nirt/trainer.py` (65), `src/training/nirt/baseline/trainer.py` (80) · **Context-only:** `src/training/nirt/train.py`, `src/training/nirt/baseline/train.py`, `src/training/nirt/baseline/*.py`, `src/training/trainers/base.py`, `src/training/pipeline.py`, `src/router/models/{store,artifacts}.py`, `docs/nirt_model.md`, `docs/baseline_control_arm.md`
**Summary:** 5 findings: 2 correctness, 3 design · 1 confirmed, 4 plausible

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| TN-01 | `src/training/nirt/trainer.py:32-38`, `src/training/nirt/baseline/trainer.py:32-44` | correctness | medium | confirmed |
| TN-02 | `src/training/nirt/trainer.py:26-38`, `src/training/nirt/baseline/trainer.py:32-44` | design | low | confirmed |
| TN-03 | `src/training/nirt/trainer.py:57-64` | design | low | plausible |
| TN-04 | `src/training/nirt/baseline/trainer.py:59-61` | correctness | low | plausible |
| TN-05 | `src/training/nirt/trainer.py:65`, `src/training/nirt/baseline/trainer.py:72-80` | design | low | plausible |

---

## src/training/nirt/trainer.py / src/training/nirt/baseline/trainer.py

### TN-01: config translation crashes on an explicit `null` config section `:32-38` / `:32-44`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
# trainer.py / baseline/trainer.py (identical in both)
cfg = json.loads(json.dumps(config.hyperparameters))
cfg["seed"] = config.seed
if config.max_epochs is not None:
    cfg.setdefault("train", {})["epochs"] = config.max_epochs
if config.early_stopping_patience is not None:
    cfg.setdefault("train", {})["patience"] = config.early_stopping_patience
if config.preprocessing:
    cfg.setdefault("data", {}).update(config.preprocessing)
```

**What breaks:** `dict.setdefault(key, {})` only supplies the default when `key` is *absent* — if `config.hyperparameters` already contains `"train": None` or `"data": None` (a real, anticipated shape: both wrapped `fit()` functions defensively unwrap it themselves, e.g. `training/nirt/train.py:408-409` does `tcfg = nirt_cfg.get("train", {}) or {}` / `dcfg = nirt_cfg.get("data", {}) or {}`, and YAML commonly produces `null` for an empty block like `train:` with no children), `setdefault` returns `None` unchanged and the next line does `None["epochs"] = ...` / `None.update(...)`, raising `TypeError`/`AttributeError` instead of the config values being applied. This triggers whenever a caller passes a config with an explicit null `train`/`data` block *and* also sets `config.max_epochs`, `config.early_stopping_patience`, or `config.preprocessing` on the typed `TrainingConfig` — exactly the "typed fields override/fill in hyperparameters" use case both wrapper docstrings advertise as the intended calling convention.
**Evidence:** Confirmed by reading both `fit()` implementations, which both explicitly guard the same sections with `... or {}` precisely because an explicit `None` value is expected to occur (`training/nirt/train.py:408-409`, and the equivalent `resp = cfg.get("response", {}) or {}` pattern in `training/nirt/baseline/train.py`). The wrapper's `setdefault(...)` does not carry the same guard, so it is strictly less defensive than the code it wraps despite operating earlier and unconditionally (every `train()` call), whereas the sibling latent case inside `fit()` itself (`nirt_cfg.setdefault("data", {})["phase0_config"] = ...` at `train.py:758`) only fires conditionally under `save and phase0_cfg.source_path is not None`.
**Direction:** Match the wrapped functions' own guard: `cfg.setdefault("train", {})` → `(cfg.setdefault("train", cfg.get("train") or {}))` or simply normalize once up front, e.g. `cfg["train"] = cfg.get("train") or {}` / `cfg["data"] = cfg.get("data") or {}`, before touching sub-keys.

---

### TN-02: `_nirt_cfg_from_config`/`_phase1_cfg_from_config` are byte-identical, duplicated across two files `:26-38` / `:32-44`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
def _nirt_cfg_from_config(config: TrainingConfig) -> dict:
    cfg = json.loads(json.dumps(config.hyperparameters))
    cfg["seed"] = config.seed
    if config.max_epochs is not None:
        cfg.setdefault("train", {})["epochs"] = config.max_epochs
    if config.early_stopping_patience is not None:
        cfg.setdefault("train", {})["patience"] = config.early_stopping_patience
    if config.preprocessing:
        cfg.setdefault("data", {}).update(config.preprocessing)
    return cfg
```

**What breaks:** Nothing at runtime, but the two functions (`_nirt_cfg_from_config` in `trainer.py`, `_phase1_cfg_from_config` in `baseline/trainer.py`) have identical bodies (only the docstring wording differs). TN-01's bug, and any future fix to it, has to be applied twice; a future `TrainingConfig` field will silently only reach one of the two `fit()` wrappers if a contributor edits just one copy (which is exactly how the two currently share the bug in TN-01 — copy-pasted together).
**Evidence:** Direct text comparison of the two function bodies in the diff (both files, `git diff main...HEAD`).
**Direction:** Hoist the shared translation into one helper (e.g. `training/trainers/base.py` or a small shared module both `training/nirt/trainer.py` and `training/nirt/baseline/trainer.py` import), since `TrainingConfig` itself is already the common, shared contract both are translating from.

---

### TN-03: manual re-derivation of `_resolve_runs_dir`'s fallback path instead of reuse `src/training/nirt/trainer.py:57-64`
**Category:** design · **Severity:** low · **Confidence:** plausible

```python
runs_dir = self.runs_dir
if runs_dir is None:
    from router.config import DEFAULT_NIRT_RUNS_DIR, resolve_path

    runs_dir = resolve_path(nirt_cfg.get("runs_dir", DEFAULT_NIRT_RUNS_DIR))
return LocalArtifactStore(runs_dir).load(config.run_name)
```

**What breaks:** Currently nothing — this happens to match `train.py`'s own `_resolve_runs_dir(nirt_cfg, phase0_cfg)` fallback branch (`train.py:797-806`) exactly, because `NIRTTrainer.train()` never forwards a `phase0_cfg` to `fit()`, so `_resolve_runs_dir` always takes its `resolve_path(rel)` branch, never the `phase0_cfg.resolve(rel)` branch. It is covered by `tests/training/nirt/test_trainer.py::test_runs_dir_none_falls_back_to_the_default_runs_dir`, which passes. But it is a private helper (`_resolve_runs_dir`, underscore-prefixed) reimplemented inline rather than called, with a comment asserting the two stay "the exact same way" — that assertion silently stops being true the moment `NIRTTrainer` (or `fit`'s default `phase0_cfg` handling) changes to honor a non-default `phase0_cfg`, since `_resolve_runs_dir` has a second branch (`phase0_cfg.resolve(rel)`) this copy never replicates.
**Evidence:** Compared `trainer.py:57-64` against `train.py:797-806`'s two branches; `NIRTTrainer.train()` (`trainer.py:53-56`) has no `phase0_cfg` parameter to forward, so the divergent branch is currently unreachable from this wrapper, not incorrect today.
**Direction:** If `_resolve_runs_dir` is meant to be reused rather than duplicated, drop the leading underscore (or add a small public wrapper) and import it directly instead of re-deriving the fallback by hand.

---

### TN-04: relative `runs_dir` resolves against different roots in `BaselineNIRTTrainer` vs `NIRTTrainer` `src/training/nirt/baseline/trainer.py:59-61`
**Category:** correctness · **Severity:** low · **Confidence:** plausible

```python
out_dir = str(Path(self.runs_dir) / config.run_name) if self.runs_dir else None
run = fit(phase1_cfg, arrays=dataset, out_dir=out_dir, save=self.runs_dir is not None,
          verbose=False)
```

**What breaks:** When `self.runs_dir` is a *relative* path, `BaselineNIRTTrainer` passes that relative `out_dir` straight into `training/nirt/baseline/train.py`'s `fit()`, which resolves a non-absolute `out_dir` against the phase-0 `Config`'s root (`out = out if out.is_absolute() else p0.root / out`, `baseline/train.py:230-231`) — not CWD. `NIRTTrainer` (`trainer.py`), by contrast, passes its relative `runs_dir` into `training/nirt/train.py`'s `fit()`, which uses `Path(runs_dir)` completely as given with no root-joining when `runs_dir is not None` (`train.py:744`). So the same `runs_dir="some/relative/path"` constructor argument lands the on-disk checkpoint in a different place depending on which of the two trainer classes it's given to (`p0.root/some/relative/path/<run>` for baseline vs `<cwd>/some/relative/path/<run>` for NIRT), whenever the process CWD and the phase-0 config root differ.
**Evidence:** Traced both `fit()` bodies' out-dir/runs-dir resolution logic (`baseline/train.py:229-231` vs `train.py:743-744`). The existing test `test_train_writes_to_disk_when_a_runs_dir_is_given` only exercises `runs_dir=tmp_path` (always absolute), so this asymmetry is untested. It affects only the debug/provenance checkpoint on disk — `BaselineNIRTTrainer` never reads it back (`self.last_result`/the returned artifact are built purely in-memory from `run`), so the returned `RouterModelArtifact` itself is unaffected; only where the side-effect file lands can surprise a caller relying on `runs_dir/run_name`.
**Direction:** Either resolve `out_dir` the same way `NIRTTrainer` does (pass through as-is) or explicitly document/route it through `resolve_path` before joining, so a relative `runs_dir` behaves identically across both trainer classes.

---

### TN-05: artifacts built by both trainers always carry empty provenance fields `src/training/nirt/trainer.py:65`, `src/training/nirt/baseline/trainer.py:72-80`
**Category:** design · **Severity:** low · **Confidence:** plausible

**What breaks:** `RouterModelArtifact` is documented (`router/models/artifacts.py:1-10`) as carrying "enough to resolve a live decision back to an exact checkpoint, training dataset version, and hyperparameters." Neither trainer populates that: `BaselineNIRTTrainer.train()` constructs `RouterModelArtifact(..., content_hash="", ...)` directly with `training_dataset_version`/`created_at` left at their dataclass defaults (`""`); `NIRTTrainer.train()` gets its artifact back via `LocalArtifactStore.load()`, which reads `content_hash`/`schema_version`/`training_dataset_version`/`created_at` out of `run.json` (`store.py:82-97`) — but the `run.json` actually on disk was written by `fit()`'s own save path (`train.py:771-789`), whose keys (`name, git_sha, elapsed_sec, config, n_models, model_index, best_epoch, history, val_metrics, baselines`) never include any of those fields, so they silently fall back to `meta.get(..., "")`/`""` every time, for every run.
**Evidence:** Traced `LocalArtifactStore._load_nirt`'s `meta.get(...)` calls against the actual keys `fit()` writes to `run.json` at `train.py:771-789` — no overlap for `content_hash`/`schema_version`/`training_dataset_version`/`created_at`/`router_model_version`. This mirrors `MLPRouterTrainer`'s own `content_hash=""` (`training/trainers/mlp_router_trainer.py:62`), so it looks like an accepted, consistent gap across all three new trainer wraps rather than something unique to this package — flagging for visibility since it weakens the stated purpose of `RouterModelArtifact`, not as a novel bug.
**Direction:** If real provenance tracking (content hash, dataset version) is wanted before these artifacts are relied on for routing/serving decisions, it needs to be computed explicitly (e.g. hash the saved `model.pt`) rather than assumed to come along for free from `fit()`'s existing `run.json`.
