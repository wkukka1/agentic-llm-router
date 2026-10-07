# Code review: training.trainers

**Files reviewed:** 2 (132 lines): base.py (65), mlp_router_trainer.py (67) · **Context-only:** src/training/nirt/trainer.py, src/training/nirt/baseline/trainer.py, src/training/trainers/mlp_router.py, src/router/models/artifacts.py, src/router/models/store.py, src/training/pipeline.py, src/router/routing/routers.py, docs/architecture.md, tests/training/trainers/*, tests/training/test_pipeline.py
**Summary:** 5 findings: 2 correctness, 3 design · 4 confirmed, 1 plausible

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| TT-01 | `src/training/trainers/mlp_router_trainer.py:58-67` (+ `src/router/routing/routers.py:450-477`) | correctness | high | confirmed |
| TT-02 | `src/training/trainers/mlp_router_trainer.py:41-42` | correctness | medium | confirmed |
| TT-03 | `src/training/trainers/mlp_router_trainer.py:54-56` | design | medium | confirmed |
| TT-04 | `src/training/trainers/mlp_router_trainer.py:58-67` | design | low | plausible |
| TT-05 | `src/training/trainers/base.py` vs. all 3 subclasses | design | low | confirmed |

---

## src/training/trainers/mlp_router_trainer.py

### TT-01: `MLPRouterTrainer` persists an artifact no registered `RouterModel` can actually load back `:58-67`
**Category:** correctness · **Severity:** high · **Confidence:** confirmed

```python
58:        store = LocalArtifactStore(self.runs_dir)
59:        store.save(RouterModelArtifact(
60:            artifact_id=config.run_name,
61:            payload=payload,
...
64:            router_model_name="mlp",
65:            supported_model_ids=list(model_ids),
66:        ))
67:        return store.load(config.run_name)
```

**What breaks:** The artifact-driven consumption path this trainer is supposedly plugging into is `RouterModelFactory.create(artifact, cost_model)` (`src/router/models/registry.py`), which does `getattr(model_class, "from_artifact", None)` and raises `TypeError` if the registered class has none. `MLPRouter` (`src/router/routing/routers.py:450`, `kind = "mlp"`) has **no `from_artifact` classmethod** — only `NIRTRouter` (line 228) does. So the moment any caller does the natural thing — `factory.register("mlp", MLPRouter); factory.create(mlp_artifact, cost_model)`, exactly mirroring `tests/training/test_pipeline.py`'s NIRT test (`test_harness_train_runs_nirt_and_loads_through_the_factory`) — it crashes.

`tests/training/test_pipeline.py::test_harness_train_runs_mlp_router` and `tests/training/trainers/test_mlp_router_trainer.py::test_train_persists_and_reloads_a_working_router` both avoid this: they never call `RouterModelFactory`, and the latter manually reconstructs the model via `build_mlp_router(...) + load_state_dict(...)` instead of exercising any documented artifact→router path. That's a tell that the round trip was validated at the payload/store layer only, not end-to-end.

**Evidence:** `router.models.registry.RouterModelFactory.create` (`src/router/models/registry.py:44-60`); `MLPRouter`'s docstring itself points elsewhere — "Fit it with `training.trainers.mlp_router.fit_mlp_router_as_router`" (in-memory construction, bypassing the artifact/store entirely) — suggesting the store-backed path added by this trainer was never actually meant to be the way `MLPRouter` gets built, but `MLPRouterTrainer`'s whole purpose (per its own module docstring: "adds persistence... rather than adapting an existing convention") is to make the artifact/store path work for MLP the way it already does for NIRT.
**Direction:** Either add `MLPRouter.from_artifact(artifact, cost_model)` (mirroring `NIRTRouter.from_artifact`: rebuild via `build_mlp_router(payload.in_dim, len(payload.model_ids), payload.hidden, payload.dropout)` + `load_state_dict`), or update this trainer's/`MLPArtifactPayload`'s docstrings to state plainly that the persisted "mlp" artifact is not yet consumable via `RouterModelFactory` and name the intended loader.

---

### TT-02: `seed` hyperparameter silently overrides the typed `config.seed`, opposite of the sibling trainers `:41-42`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
41:        kw = dict(config.hyperparameters)
42:        kw.setdefault("seed", config.seed)
```

**What breaks:** `NIRTTrainer._nirt_cfg_from_config` and `BaselineNIRTTrainer._phase1_cfg_from_config` both do `cfg["seed"] = config.seed` — an unconditional assignment, so the typed `TrainingConfig.seed` field always wins over whatever `hyperparameters` carries. `MLPRouterTrainer` instead uses `kw.setdefault("seed", config.seed)`, which only applies `config.seed` when `"seed"` is **absent** from `hyperparameters` — if `hyperparameters` already has a `"seed"` key (a very plausible authoring pattern for MLP, since `fit_mlp_router`'s own kwargs — `hidden`, `epochs`, `lr`, `dropout`, `seed`, ... — are exactly the flat shape `hyperparameters` is documented to hold, unlike NIRT's nested `{"model":..., "train":..., "data":...}` shape), the caller's explicit `config.seed` is silently discarded in favor of the stale value baked into `hyperparameters`.

The existing test `tests/training/trainers/test_mlp_router_trainer.py::test_typed_config_fields_override_hyperparameters` is misleadingly named for this: it asserts `config.max_epochs` overrides `hyperparameters["epochs"]` correctly, but never puts `"seed"` in `hyperparameters` at all, so it exercises `setdefault`'s no-conflict branch and never catches the inversion.
**Evidence:** Direct comparison of `mlp_router_trainer.py:42` against `src/training/nirt/trainer.py:31` and `src/training/nirt/baseline/trainer.py:37`.
**Direction:** `kw["seed"] = config.seed` (unconditional), matching the other two trainers; add a regression test that puts `"seed"` in `hyperparameters` and asserts the typed field still wins.

---

### TT-03: Reimplements MLP-module introspection by hardcoded index instead of reusing the safer pattern already in `router.nirt.baselines_infer` `:54-56`
**Category:** design · **Severity:** medium · **Confidence:** confirmed

```python
54:            in_dim=int(model[0].in_features),
55:            hidden=int(model[0].out_features),
56:            dropout=float(model[2].p),
```

**What breaks:** `model` is the `nn.Sequential` returned by `build_mlp_router` (`src/router/nirt/baselines_infer.py:101-106`: `Linear, ReLU, Dropout, Linear`). This code reaches in by positional index (`model[0]`, `model[2]`) to recover the shape it needs to rebuild the module later. The same file that owns `build_mlp_router` already has a more robust pattern for exactly this kind of introspection — `mlp_router_matrix` (`baselines_infer.py:142`) finds the first `Linear` via `next(m for m in model.modules() if isinstance(m, torch.nn.Linear))` rather than assuming a fixed index. `mlp_router_trainer.py` doesn't reuse that helper and instead couples itself to `build_mlp_router`'s exact internal layer ordering across module/package boundaries (`training.trainers` reaching into a structure owned by `router.nirt.baselines_infer`); an ordering change there (e.g. inserting a `BatchNorm` or a second hidden layer) breaks this silently or with a confusing `AttributeError` (`.p` on a non-`Dropout` module) rather than a clear error at the actual point of incompatibility.
**Evidence:** `build_mlp_router` definition vs. `mlp_router_matrix`'s `isinstance`-based lookup, both in `src/router/nirt/baselines_infer.py`.
**Direction:** Add a small helper (in `baselines_infer.py`, next to `build_mlp_router`) that returns `(in_dim, hidden, dropout)` from a built model via `isinstance` checks over `model.modules()`, and have `mlp_router_trainer.py` call it instead of indexing.

---

### TT-04: `train()` re-reads the artifact it just wrote instead of returning what's already in memory `:58-67`
**Category:** design (efficiency) · **Severity:** low · **Confidence:** plausible

```python
58:        store = LocalArtifactStore(self.runs_dir)
59:        store.save(RouterModelArtifact(...))
...
67:        return store.load(config.run_name)
```

**What breaks:** `store.save(...)` is called with the fully-formed `RouterModelArtifact` (payload already built from the just-fit `model`/`model_ids` in memory), then `store.load(config.run_name)` immediately re-reads it off disk — a full `torch.load` deserialization of the `state_dict` that was just serialized, for a result that (per `LocalArtifactStore._load_mlp`) reconstructs an equivalent `MLPArtifactPayload` plus an essentially-empty `run.json` (`content_hash`/`schema_version`/`hyperparameters` all default to `""`/`{}` since `save()` only ever writes what the caller put on `artifact.hyperparameters`, which was never set here). Nothing meaningful is gained over returning the artifact object already constructed at line 59. This mirrors `NIRTTrainer`'s save-then-load pattern (there it's cheaper to justify: `fit(..., save=True)` writes the file and never keeps a typed `RouterModelArtifact` around), but `MLPRouterTrainer` already has the typed artifact in hand before calling `store.save`, making the extra disk round trip pure overhead — doubled I/O on every `train()` call, worse on slow/networked `runs_dir`.
**Evidence:** Compared against `LocalArtifactStore.save`/`_load_mlp` (`src/router/models/store.py:100-147`), which shows `_load_mlp` reads back only what `save()` just wrote.
**Direction:** Return the already-constructed artifact directly (`return store.load(...)` → keep a reference to what was passed to `save` and return that), unless there's a deliberate reason to force a round-trip integrity check — if so, say so in a comment, since it reads as an oversight otherwise.

---

## src/training/trainers/base.py

### TT-05: The ABC standardizes `train()`/`name`/`version`/`last_result` but not construction, so each of the three implementations has a different, uncoordinated `__init__` shape
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
54:class RouterModelTrainer(abc.ABC):
55:    name: str = "trainer"
56:    version: str = "0"
57:
58:    #: the wrapped fit() call's own result object from the most recent
59:    #: train() -- see the module docstring's second gap. None before train()
60:    #: is ever called.
61:    last_result: Optional[Any] = None
62:
63:    @abc.abstractmethod
64:    def train(self, dataset: Any, config: TrainingConfig) -> RouterModelArtifact:
65:        raise NotImplementedError
```

**What breaks:** Nothing today (no caller constructs a `RouterModelTrainer` generically by name — `TrainingHarness`/`TrainingJob` both take an already-built instance), but the base class defines no `__init__` at all, so there's no enforced constructor contract to go with the enforced `train()` contract. The three concrete trainers disagree on it: `NIRTTrainer(runs_dir: Optional[os.PathLike] = None)`, `BaselineNIRTTrainer(runs_dir: Optional[str] = None)`, `MLPRouterTrainer(runs_dir: os.PathLike)` (required, no default — noted deliberately in a comment as "unlike NIRTTrainer/BaselineNIRTTrainer's optional runs_dir"). `docs/architecture.md`'s own note ("One implementation per router model... concrete variants aren't modeled") explicitly defers this, so it's a known, accepted gap rather than an oversight — flagging because the day a `TrainerFactory`/registry (the trainer-side analogue of `RouterModelFactory`) gets added, keyed by `router_model_name` the way the docstring describes, it will need a uniform construction signature this ABC doesn't provide, and `MLPRouterTrainer`'s required arg will be the one that doesn't fit a generic `Trainer(**common_kwargs)` call.
**Evidence:** Constructor signatures compared across `base.py`, `mlp_router_trainer.py:32`, `src/training/nirt/trainer.py:45`, `src/training/nirt/baseline/trainer.py:51`.
**Direction:** No action needed now; if/when a trainer registry is built, either give `RouterModelTrainer.__init__(self, runs_dir=None)` a base implementation subclasses call via `super().__init__()`, or document that `runs_dir` optionality is intentionally per-trainer (MLP has no fallback persistence path, so it can't default to `None` the way NIRT/baseline do).
