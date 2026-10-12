# Code review: router.models

**Files reviewed:** 4 (373 lines): `src/router/models/__init__.py`, `src/router/models/artifacts.py` (173 lines), `src/router/models/registry.py` (63 lines), `src/router/models/store.py` (156 lines) · **Context-only:** `src/router/nirt/checkpoint.py`, `src/router/routing/base.py`, `src/router/routing/routers.py`, `src/training/nirt/trainer.py`, `src/training/nirt/baseline/trainer.py`, `src/training/trainers/mlp_router_trainer.py`, `src/training/trainers/mlp_router.py`, `src/training/pipeline.py`, `docs/architecture.md`

**Summary:** 7 findings: 4 correctness, 2 design, 1 performance · 5 confirmed, 2 plausible

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| RM-01 | `src/router/models/store.py:120-147` | correctness | high | plausible |
| RM-02 | `src/router/models/registry.py:36-63` | correctness | medium | confirmed |
| RM-03 | `src/training/nirt/trainer.py:41-65`, `src/training/pipeline.py:124-134` | correctness | medium | confirmed |
| RM-04 | `src/router/models/store.py:61-68, 100-104` | performance | medium | confirmed |
| RM-05 | `src/router/models/store.py:120-124` | design | medium | confirmed |
| RM-06 | `src/router/models/store.py:149-156` | correctness | low | plausible |
| RM-07 | `src/router/models/artifacts.py` (whole file) | design | low | confirmed |

---

## src/router/models/store.py

### RM-01: `save()` writes `model.pt` and `run.json` non-atomically — a killed write leaves a "loadable" artifact with silently empty/stale metadata, or an overwrite corrupts the previous good checkpoint `:120-147`
**Category:** correctness · **Severity:** high · **Confidence:** plausible

```python
127:        out = self.root_path / artifact.artifact_id
128:        out.mkdir(parents=True, exist_ok=True)
129:        torch.save(artifact.payload.to_state(), out / "model.pt")
130:        (out / "run.json").write_text(
...
147:        return str(out)
```
and the read side:
```python
58:    def exists(self, artifact_id: str) -> bool:
59:        return (self.root_path / artifact_id / "model.pt").exists()
...
149:    def _read_run_json(self, artifact_id: str) -> dict[str, Any]:
150:        path = self.root_path / artifact_id / "run.json"
151:        if not path.exists():
152:            return {}
153:        try:
154:            return json.loads(path.read_text(encoding="utf-8"))
155:        except (json.JSONDecodeError, OSError):
156:            return {}
```
**What breaks:** `save()` writes two files directly to their final paths, in sequence, neither atomically nor together. Three crash windows:
1. Process killed after `torch.save(model.pt)` completes but before `run.json` is written (or while `run.json` is being written). `load()` then succeeds — `exists()` is true, `load_run`/`torch.load` succeeds — but `_read_run_json` returns `{}` (file missing) or `{}` (caught `JSONDecodeError` on a half-written file), so the artifact silently comes back with `content_hash=""`, `schema_version="0"`, `router_model_version=""`, `training_dataset_version=""`, `created_at=""`. No exception, no warning — exactly the class of bug flagged in `router.embeddings` ("a killed rebuild leaves a store that loads with new ids, zero vectors and the old complete manifest"): here it's "loads with a valid model and a blank identity."
2. Process killed mid-`torch.save(model.pt)` on an **overwrite** of an existing `artifact_id` (re-running training under the same run name, or `MLPRouterTrainer`/`BaselineNIRTTrainer` re-saving). `torch.save` writes to the destination path directly (no temp file), so a partial write clobbers the previously-good `model.pt` in place — the prior valid checkpoint is lost, not just the new one.
3. Two callers calling `save()` for the same `artifact_id` concurrently interleave their `model.pt` and `run.json` writes with no locking.

**Evidence:** Traced `save()`'s write order and `_read_run_json`'s fallback-to-`{}` on both missing-file and parse-error, against `load()`/`_load_nirt`/`_load_mlp`, which never check whether `run.json` existed at all before defaulting every metadata field.
**Direction:** Write `model.pt` (and `run.json`) to a temp path in the same directory and `os.replace()` into place, writing `run.json` last (or both, then rename the whole `out` directory into place from a `out.tmp` sibling) so a reader never observes a directory with a complete blob and a missing/stale sidecar. At minimum, make `_read_run_json` distinguish "no run.json" from "artifact never had one" so callers can detect the crash-mid-write case instead of silently defaulting.

---

## src/router/models/registry.py

### RM-02: `create(artifact, cost_model=...)` silently discards the `cost_model` keyword when `artifact_or_id` is not a string `:36-63`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
36:    def create(
37:        self,
38:        artifact_or_id: "RouterModelArtifact | str",
39:        store_or_cost_model: "ArtifactStore | CostModel | None" = None,
40:        cost_model: Optional["CostModel"] = None,
41:    ) -> "RouterModel":
42:        """``create(artifact, cost_model)`` or ``create(artifact_id, store, cost_model)``."""
43:        if isinstance(artifact_or_id, str):
44:            store = store_or_cost_model
45:            if store is None:
46:                raise TypeError("create(artifact_id, store, cost_model) needs a store")
47:            artifact = store.load(artifact_or_id)
48:        else:
49:            artifact = artifact_or_id
50:            cost_model = store_or_cost_model
51:```
**What breaks:** The two-overloads-in-one-method design keeps a third parameter, `cost_model`, on the signature purely for the `(artifact_id, store, cost_model)` call shape. But in the `else` branch (bare artifact object, the common case per the docstring's first form), line 50 unconditionally overwrites `cost_model` with `store_or_cost_model` — which defaults to `None`. Any caller who calls `factory.create(artifact, cost_model=my_cost_model)` (a plausible reading of the signature: pass the artifact, pass cost_model by keyword) gets `cost_model` silently reset to `None` — cost-aware routing gets built with no cost model and no error. Only `factory.create(artifact, my_cost_model)` (second positional arg) works, which every existing call site (tests, `NIRTRouter.from_artifact` usage) happens to use correctly, so this hasn't surfaced yet.
**Evidence:** Read the full branch logic; confirmed no call site currently uses the keyword form (`tests/router/models/test_factory.py` only passes `cost_model` positionally as the second argument).
**Direction:** Either drop the third `cost_model` parameter entirely (it's dead in the object-artifact branch as written) and document only the two positional call shapes, or fix the branch to prefer an explicitly-passed `cost_model` over `store_or_cost_model` (e.g. `cost_model = cost_model if cost_model is not None else store_or_cost_model`).

---

## src/training/nirt/trainer.py / src/training/pipeline.py

### RM-03: `NIRTTrainer`'s returned artifact — and therefore `TrainingRun.dataset_version`/`created_at` — is always empty, because it round-trips through `fit()`'s `run.json`, which never carries those keys `:41-65` / `:124-134`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

`NIRTTrainer.train()` (the trainer this package's own docstring calls "lowest-risk" and presumably the main production path) never calls `LocalArtifactStore.save()` — it lets `training.nirt.train.fit(..., save=True)` write `model.pt`/`run.json` itself, then does `LocalArtifactStore(runs_dir).load(config.run_name)`. `fit()`'s `run.json` writer (`src/training/nirt/train.py:771-789`) writes only `{name, git_sha, elapsed_sec, config, n_models, model_index, best_epoch, history, val_metrics, baselines}` — none of `content_hash`, `schema_version`, `router_model_version`, `training_dataset_version`, or `created_at`, the exact fields `store.py`'s own `_load_nirt`/`_read_run_json` populate from a `run.json` written by `LocalArtifactStore.save()`. Since `fit()`'s `run.json` never has those keys, every one of them silently defaults (`meta.get(..., "")`) on the artifact `NIRTTrainer` returns.

`TrainingHarness.train()` then does `dataset_version=artifact.training_dataset_version, created_at=artifact.created_at` (`src/training/pipeline.py:132-133`) — so for the NIRT path (and by the same logic, `BaselineNIRTTrainer`, which builds its artifact with no metadata fields set at all), `TrainingRun.dataset_version` and `.created_at` are **always** empty strings, never the actual values, with no error or warning anywhere in the chain. This directly undercuts `artifacts.py`'s own stated purpose: "enough to resolve a live decision back to an exact checkpoint, **training dataset version**, and hyperparameters."
**Evidence:** Read `training/nirt/train.py:771-789`'s `run.json` writer, `store.py::_load_nirt`/`_read_run_json`'s key vocabulary, and `pipeline.py:132-133`'s direct passthrough.
**Direction:** Either have `fit()` write the fields `LocalArtifactStore` expects (schema/content-hash/dataset-version) into `run.json`, or have `NIRTTrainer.train()` populate them explicitly on the artifact it returns (e.g. compute `training_dataset_version` from `config`/`dataset` before/after calling `store.load()`), rather than relying on a `run.json` shape that was never designed to satisfy this contract.

---

## src/router/models/store.py

### RM-04: `load()` reads the full checkpoint blob from disk twice — once to peek `"format"`, once more inside `_load_nirt`/`_load_mlp` `:61-68, 100-104`
**Category:** performance · **Severity:** medium · **Confidence:** confirmed

```python
61:    def load(self, artifact_id: str) -> RouterModelArtifact:
62:        import torch
63:
64:        blob_path = self.root_path / artifact_id / "model.pt"
65:        fmt = torch.load(blob_path, map_location="cpu", weights_only=True).get("format")
66:        if fmt == "mlp":
67:            return self._load_mlp(artifact_id)
68:        return self._load_nirt(artifact_id)
```
`_load_nirt` then calls `load_run()` (which does its own `torch.load(path, ...)` of the same file), and `_load_mlp` does `torch.load(self.root_path / artifact_id / "model.pt", ...)` again independently. Every `load()` call deserializes the full state dict (all model tensors) from disk twice to learn a single string key.
**What breaks:** Not incorrect, just wasteful — on a large NIRT/MLP checkpoint this doubles both I/O and CPU-side unpickling cost on every artifact load (`RouterModelFactory.create`, tests, any serving-path cold start).
**Direction:** Pass the already-loaded `blob` dict into `_load_mlp`/an inlined NIRT path instead of re-reading, or read only `run.json`'s (or a lightweight header's) format tag instead of the full tensor blob when one exists; only fall back to peeking the blob for legacy runs with no `run.json`.

---

### RM-05: `ArtifactPayload` is advertised as a generic, shape-agnostic extension point, but `save()` hardcodes the two concrete subclasses it accepts `:120-124`
**Category:** design · **Severity:** medium · **Confidence:** confirmed

```python
120:    def save(self, artifact: RouterModelArtifact) -> str:
121:        if not isinstance(artifact.payload, (NIRTArtifactPayload, MLPArtifactPayload)):
122:            raise NotImplementedError(
123:                f"LocalArtifactStore.save does not support {type(artifact.payload).__name__}"
124:            )
```
`ArtifactPayload`'s docstring in `artifacts.py:28-34` says it exists so `ArtifactStore` can "(de)serialize any payload generically without knowing its shape" via `to_state`/`from_state`. In practice `save()` gates on an `isinstance` allowlist of two specific subclasses before ever calling the generic `to_state()`, so adding any new `ArtifactPayload` subclass (a fifth router kind) requires editing `store.py` again regardless of whether it correctly implements `to_state`/`from_state` — the abstraction is generic in shape but closed in practice, one `isinstance` check away from the "any payload" promise in its own docstring.
**Evidence:** Read `store.py:120-124` against `artifacts.py`'s `ArtifactPayload` docstring; the load side (`load()`) has the equivalent problem via its `fmt == "mlp"` / else dispatch, which is similarly closed to exactly two formats.
**Direction:** Either drop the "generic" framing from the docstring (it's accurate to say "we support NIRT and MLP payloads today, add an isinstance branch for new ones") or make the allowlist a registry (`{"nirt": NIRTArtifactPayload, "mlp": MLPArtifactPayload}`) a new payload type can register itself into, matching the pattern `RouterModelFactory`/`router.routing.registry` already use elsewhere in this package for the analogous "new kind shouldn't need editing existing code" problem.

---

### RM-06: `_read_run_json` swallows `OSError`/`JSONDecodeError` into an empty dict with no logging, indistinguishable from "never had metadata" `:149-156`
**Category:** correctness · **Severity:** low · **Confidence:** plausible

```python
153:        try:
154:            return json.loads(path.read_text(encoding="utf-8"))
155:        except (json.JSONDecodeError, OSError):
156:            return {}
```
**What breaks:** A corrupt/truncated `run.json` (partial write, disk error, concurrent writer) is treated identically to a legacy run that simply predates the sidecar — both produce the same silent `{}`. Combined with RM-01, this hides evidence of exactly the kind of on-disk corruption this review was asked to check for.
**Direction:** At least log a warning distinguishing "file present but unparseable" from "file absent," so a corrupted sidecar doesn't look identical to a normal legacy-checkpoint miss.

---

## src/router/models/artifacts.py

### RM-07: `content_hash` is a required, no-default field on `RouterModelArtifact` that no code anywhere computes `:159-172`
**Category:** design · **Severity:** low · **Confidence:** confirmed

Every construction site in the diff and its callers passes `content_hash=""` literally (`store.py:86,110`, `training/nirt/baseline/trainer.py:75`) or a hardcoded test literal (`tests/router/models/test_store.py:62`, `"abc123"`). No function in `router.models`, `training.nirt.trainer`, `training.nirt.baseline.trainer`, or `training.trainers.mlp_router_trainer` computes an actual hash of `payload`/`model.pt` contents. The field exists (and is documented via `ArtifactFormat`/the class diagram as load-bearing identity) but is dead weight today — nothing populates or verifies it, so it cannot yet serve its apparent purpose (detecting a swapped/corrupted checkpoint, which is exactly the failure mode RM-01 introduces).
**Direction:** Either compute a real hash (e.g. of the serialized `model.pt` bytes) in `LocalArtifactStore.save()` before writing `run.json`, or drop the field until something populates it — a permanently-empty "content hash" is worse than no field, since it reads as verified when it isn't.
