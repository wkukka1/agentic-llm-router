# Code review: training

**Files reviewed:** 1 (134 lines): pipeline.py · **Context-only:** cli.py, phase0.py, trainers/base.py, trainers/mlp_router_trainer.py, nirt/trainer.py, nirt/baseline/trainer.py, router/models/artifacts.py, router/models/store.py
**Summary:** 6 findings: 3 correctness, 2 design, 1 convention · 6 confirmed, 0 plausible

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| TP-01 | `src/training/pipeline.py:85-111` | correctness | high | confirmed |
| TP-02 | `src/training/pipeline.py:114-116` | correctness | medium | confirmed |
| TP-03 | `src/training/pipeline.py:118-134` | correctness | medium | confirmed |
| TP-04 | `src/training/pipeline.py:63-69` | design | low | confirmed |
| TP-05 | `src/training/pipeline.py:8-17` | convention | low | confirmed |
| TP-06 | `src/training/pipeline.py:34` | convention | low | confirmed |

---

## src/training/pipeline.py

### TP-01: `TrainingMetrics` (train-side loss/accuracy/bce/auc/spearman) is always 0.0 for every real trainer `:85-111`
**Category:** correctness · **Severity:** high · **Confidence:** confirmed

```python
102: def _extract_metrics(last_result: Any) -> tuple[TrainingMetrics, ValidationMetrics]:
106:     val_metrics = getattr(last_result, "val_metrics", None) or {}
107:     validation = ValidationMetrics(**_metrics_kwargs(val_metrics))
108:     history = getattr(last_result, "history", None) or []
109:     train_dict = history[-1] if history else {}
110:     training = TrainingMetrics(**_metrics_kwargs(train_dict))
```

`_metrics_kwargs` looks for bare keys `"loss"`, `"accuracy"`, `"bce"`, `"auc"`,
`"spearman_r"/"spearman"` in `train_dict`. But both wrapped trainers build
their `history` rows with different names:
- `training/nirt/train.py:655`: `row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in vm.items()}}`
- `training/nirt/baseline/train.py:202`: `history.append({"epoch": epoch, "train_loss": train_loss, "val_nll": val_nll, **{f"val_{k}": v for k, v in vm.items() ...}})`

Every metric in the last history row is either `train_loss` (not `loss`) or
prefixed `val_*` (not matched by `_metrics_kwargs` at all — those are only
ever read from `val_metrics`, not `history`). No row from either trainer
ever contains a bare `"loss"`, `"accuracy"`, `"bce"`, `"auc"`, or
`"spearman_r"` key, and neither training loop computes train-side accuracy/
bce/auc/spearman at all (only `train_loss` is tracked per epoch).

**What breaks:** `TrainingRun.training_metrics` is `TrainingMetrics(loss=0.0,
accuracy=0.0, bce=0.0, auc=0.0, spearman=0.0, regret=0.0,
training_time=<real>)` for every NIRT and baseline run, unconditionally —
not a degraded-data edge case but the deterministic outcome on the primary
path for the two trainers `TrainingHarness` is mainly built for.
`tests/training/test_pipeline.py` doesn't catch this: it only asserts
`training_metrics.training_time > 0`, never that `loss`/`bce` are non-zero.
Any future consumer of `TrainingRun` that reports or compares train loss
(logging, run-selection, a dashboard) silently sees "0.0" for every run.
**Evidence:** traced `history.append(...)` call sites in both `fit()`
implementations and the `_metrics_kwargs` source-key list side by side.
**Direction:** add `"train_loss"` as a source for `"loss"` (matching
`("loss", ("loss", "train_loss"))`), and drop/rename the train-side
accuracy/bce/auc/spearman fields (or clearly document them as
NIRT/baseline-unsupported) since neither trainer's per-epoch loop computes
them for the training split at all.

---

### TP-02: `TrainingJob.run_id` is silently discarded — `TrainingRun.run_id` always comes from `config.run_name` `:114-116`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
114: class TrainingHarness:
115:     def run(self, job: TrainingJob) -> TrainingRun:
116:         return self.train(job.trainer, job.dataset, job.config)
```

`docs/architecture.md:860-865` models `TrainingJob` with its own `+String
runId` distinct from `config`, exactly so a caller can identify a job
independently of the trainer's run name. `TrainingHarness.run()` never reads
`job.run_id`; `train()` always sets `TrainingRun.run_id = config.run_name`
(pipeline.py:125). If a caller ever sets `job.run_id != job.config.run_name`
(e.g. reusing one `TrainingConfig` across several jobs, or namespacing job
ids separately from artifact/run names), the returned `TrainingRun.run_id`
silently reports the wrong id. The only current test
(`test_harness_run_accepts_a_training_job`) happens to set both to the same
string, masking this.
**Evidence:** `job.run_id` never referenced anywhere in pipeline.py; grepped
the whole file for `run_id` usages.
**Direction:** either use `job.run_id` (falling back to `config.run_name`)
when building the `TrainingRun` from `run()`, or drop the field from
`TrainingJob` if `config.run_name` is meant to be the sole identifier.

---

### TP-03: `TrainingRun.created_at` / `.dataset_version` are always `""` for every trainer `:118-134`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
132:             dataset_version=artifact.training_dataset_version,
133:             created_at=artifact.created_at,
```

These are read off `artifact.training_dataset_version`/`artifact.created_at`,
but no trainer path ever populates them:
- `NIRTTrainer`/`MLPRouterTrainer` go through `LocalArtifactStore`, whose
  `run.json` sidecar only carries `created_at`/`training_dataset_version` if
  the `RouterModelArtifact` passed to `save()` already had them set — and
  none of the three trainers ever set them (`MLPRouterTrainer` constructs
  the artifact with neither kwarg; `NIRTTrainer` doesn't call `save()` at
  all, it reads back `fit()`'s own richer `run.json`, which has no
  `created_at`/`training_dataset_version`/`content_hash` keys — see
  `training/nirt/train.py:770-786` vs. `router/models/store.py`'s
  `_read_run_json().get(..., "")` defaults).
- `BaselineNIRTTrainer` builds `RouterModelArtifact` directly
  (`nirt/baseline/trainer.py:72-80`) with neither kwarg either.

**What breaks:** `RouterModelArtifact`'s own docstring says it's meant to be
"enough to resolve a live decision back to an exact checkpoint, training
dataset version, and hyperparameters" — but `TrainingRun.dataset_version`/
`.created_at`, the two provenance fields the harness explicitly promotes to
top-level `TrainingRun` fields, are dead placeholders that always read `""`
regardless of which trainer ran or when.
**Evidence:** traced `LocalArtifactStore.save`/`load`/`_read_run_json` and
each trainer's `RouterModelArtifact(...)` construction call; none pass
`created_at=`/`training_dataset_version=`.
**Direction:** either have each trainer stamp `created_at`
(`datetime.utcnow().isoformat()`) and `training_dataset_version` (whatever
dataset identity is available) onto the artifact before persisting/
returning it, or have `TrainingHarness.train()` stamp `created_at` itself
(it already has `t0`) rather than relying on a field that's currently never
populated end-to-end.

---

### TP-04: `TrainingJob.output_path` is a dead field `:63-69`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
63: @dataclass
64: class TrainingJob:
65:     run_id: str
66:     trainer: RouterModelTrainer
67:     dataset: Any
68:     config: TrainingConfig
69:     output_path: str = ""
```

`output_path` is never read anywhere in `pipeline.py` (or by
`TrainingHarness.run`/`train`). The module docstring explains *why* the
harness doesn't do a save step (each trainer persists internally, format
knowledge lives with the trainer), which is a reasonable call — but it
doesn't mention that the `TrainingJob.output_path` field this same rationale
makes irrelevant is still present and silently inert. A caller who sets it
expecting it to control where the artifact lands gets no effect.
**Evidence:** grepped `output_path` across `src/training/pipeline.py`; only
the field declaration matches.
**Direction:** drop the field (cleanest, given the documented no-save-step
decision) or wire it through to the trainer's `runs_dir` where applicable,
and note the decision in the docstring alongside the existing "no separate
save step" paragraph.

---

### TP-05: Module docstring's "none compute regret today" is contradicted by `NIRTTrainer`'s underlying `fit()` `:8-17`
**Category:** convention · **Severity:** low · **Confidence:** confirmed

```python
15: left at their zero defaults otherwise -- not every trainer has an equivalent
16: of every field (none compute ``regret`` today; that lives in
17: ``evaluation.routing.oracle``, a separate, label-needing concern).
```

`training/nirt/train.py:728` unconditionally sets
`val_metrics["regret"] = _val_regret(va, val_prob)` before returning
`RunResult`, so `NIRTTrainer.last_result.val_metrics["regret"]` *is*
present and non-zero, and `_metrics_kwargs` picks it up into
`ValidationMetrics.regret` via the existing `("regret", ("regret",))`
mapping. Only the baseline trainer (`nirt/baseline/train.py`, which tracks
`val_nll` instead) has no regret. The docstring's blanket "none compute
regret today" is inaccurate for the NIRT path and could mislead a reader
into thinking `ValidationMetrics.regret` is always 0 for every trainer.
**Evidence:** read `_val_regret`'s definition/call site and the final
`val_metrics` assembly in `training/nirt/train.py:726-728`.
**Direction:** narrow the docstring to "the baseline trainer doesn't compute
regret" (or similar), since NIRT already does.

---

### TP-06: Unused `Optional` import `:34`
**Category:** convention · **Severity:** low · **Confidence:** confirmed

```python
34: from typing import Any, Optional
```

`Optional` is never used anywhere else in `pipeline.py` (all optional-typed
fields — e.g. `RouterModelTrainer.last_result` — live in `trainers/base.py`,
not here). Flagged by any standard unused-import lint (F401).
**Evidence:** grepped `Optional` in the file; only the import line matches.
**Direction:** drop `Optional` from the import.
