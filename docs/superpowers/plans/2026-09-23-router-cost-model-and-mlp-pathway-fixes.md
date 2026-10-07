# Router cost-model wiring + MLPRouter pathway persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four open, `med`-severity `known_bugs/` defects on the live routing path: `RouterModel` subclasses can't accept `cost_model`/`artifact` at construction (ROUTERMODEL-004), `cost_model` is stored but never consulted for cost (ROUTERMODEL-002), no-registry mode + `required_capabilities` silently empties every routing decision (ROUTER-003), and a reloaded `MLPRouter` always serves with the default pathway instead of the one it was trained on (MLPROUTER-001, whose root cause is training-side: MLPROUTERTRAINER-001 never persists the pathway).

**Architecture:** Five small, dependency-ordered patches to existing classes — no new files, no new abstractions. Tasks 1-3 are independent of each other and touch `router/routing/base.py`, `router/routing/routers.py`, `router/router.py`. Tasks 4-5 are a matched pair (trainer persists what the router reload needs) and must land in order: 4 (`MLPRouterTrainer` persists `pathway`/`query_pathway`/`query_features`) before 5 (`MLPRouter` reads them back and gains the constructor parameters to use them).

**Tech Stack:** Python 3.12, pytest, dataclasses, numpy/pandas, PyTorch (Task 4-5 only, already an existing test dependency via `pytest.importorskip("torch")`).

**Spec:** `fixed_bugs/RouterModel/ROUTERMODEL-004-subclass-init-drops-cost-model-and-artifact/bug.md`, `fixed_bugs/RouterModel/ROUTERMODEL-002-cost-model-never-consumed/bug.md`, `fixed_bugs/Router/ROUTER-003-no-registry-capability-filter-drops-everything/bug.md`, `fixed_bugs/RouterModelTrainer/MLPRouterTrainer/MLPROUTERTRAINER-001-hyperparameters-not-persisted/bug.md`, `fixed_bugs/RouterModel/MLPRouter/MLPROUTER-001-from-artifact-loses-pathway-and-data-config/bug.md`. Each task cites the exact suggested fix from its bug file.

## Global Constraints

- Every fix must keep `MLPRouterTrainer.train()` able to load an **old** artifact saved before this change (payload fields added with backward-compatible defaults in `from_state`) — no migration script, no breaking existing saved runs.
- No behavior change to a router built the way it already works today (`from_run`/`from_artifact` with no `pathway` override, `Router.route` with a real `llm_registry`) — only the previously-broken paths change.
- Follow the codebase's existing warn-don't-raise precedent for unverifiable constraints (`router.py`'s `RC-01` `max_latency`/`latency_weight` warnings) when a constraint can't be checked, rather than inventing a new error type.
- `RouterModel` subclass constructors keep their existing positional/keyword shape for every parameter that already exists — `cost_model`/`artifact` are additive, keyword-only, defaulting to `None`.

## Review Focus

- A router subclass built the old way (`MatrixRouter(scores)`, no `cost_model=`) must still work identically — the new keyword-only parameters must not change any existing call site's behavior. (Task 1)
- `RouterModelFactory.create(artifact, cost_model)` for `NIRTRouter`/`MLPRouter` must still set `.cost_model`/`.artifact` on the returned instance exactly as it does today (currently done by post-construction assignment; Task 1 moves this into the constructor call, and a test must confirm the observable result is unchanged). (Task 1)
- `default_model_costs` must fall back to the train-mean cost exactly as before when `cost_model` is `None` (the overwhelmingly common case today) — only a non-`None` `cost_model` should change any router's cost numbers. (Task 2)
- `required_capabilities=[]` (the default — "no capability constraint") must still let every no-registry candidate through; only a *non-empty* `required_capabilities` combined with no registry should trigger the new warn-and-keep behavior. (Task 3)
- An `MLPArtifactPayload` saved by the *old* code (no `pathway`/`query_pathway`/`query_features` keys in its state dict) must still `from_state` successfully and default to `pathway="irt"`, matching today's hardcoded default — old runs must not start raising or silently mis-scoring. (Task 4)

---

## Task 1: `RouterModel` subclasses accept and forward `cost_model`/`artifact`

**Files:**
- Modify: `src/router/routing/routers.py` (`MatrixRouter.__init__` ~L151, `NIRTRouter.__init__` ~L184 and `.from_artifact` ~L228-263, `KNNRouter.__init__` ~L387, `MLPRouter.__init__` ~L459 and `.from_artifact` ~L468-492, `RandomRouter.__init__` ~L520)
- Test: `tests/router/routing/test_routing_interface.py` (add), `tests/router/models/test_factory.py` (extend existing NIRTRouter assertions)

**Interfaces:**
- Consumes: `RouterModel.__init__(self, model_ids, *, name=None, unsupported_policy=..., cost_model=None, artifact=None)` (already exists, `router/routing/base.py:176-190` — unchanged).
- Produces: every concrete `RouterModel` subclass constructor now accepts `cost_model: Optional["CostModel"] = None, artifact: Optional["RouterModelArtifact"] = None` as trailing keyword-only parameters and forwards them to `super().__init__(...)`. `NIRTRouter.from_artifact`/`MLPRouter.from_artifact` pass them straight into `cls(...)` instead of assigning `router.cost_model = cost_model` after construction.

- [ ] **Step 1: Write the failing tests**

In `tests/router/routing/test_routing_interface.py`, add (near the existing `MatrixRouter`/`RandomRouter` tests):

```python
def test_matrix_router_accepts_cost_model_and_artifact():
    true_df, _ = _mats()
    sentinel_cost_model, sentinel_artifact = object(), object()
    r = MatrixRouter(true_df, cost_model=sentinel_cost_model, artifact=sentinel_artifact)
    assert r.cost_model is sentinel_cost_model
    assert r.artifact is sentinel_artifact


def test_random_router_accepts_cost_model_and_artifact():
    sentinel_cost_model, sentinel_artifact = object(), object()
    r = RandomRouter(M, cost_model=sentinel_cost_model, artifact=sentinel_artifact)
    assert r.cost_model is sentinel_cost_model
    assert r.artifact is sentinel_artifact
```

In a new file `tests/router/routing/test_router_model_subclass_init.py`:

```python
"""ROUTERMODEL-004: every RouterModel subclass must accept cost_model=/artifact=
at direct construction, not just through RouterModelFactory."""

from __future__ import annotations

from router.routing.routers import KNNRouter, MLPRouter, NIRTRouter


def test_nirt_router_accepts_cost_model_and_artifact():
    sentinel_cost_model, sentinel_artifact = object(), object()
    r = NIRTRouter(object(), {"a": 0, "b": 1}, data=object(),
                   cost_model=sentinel_cost_model, artifact=sentinel_artifact)
    assert r.cost_model is sentinel_cost_model
    assert r.artifact is sentinel_artifact


def test_knn_router_accepts_cost_model_and_artifact():
    sentinel_cost_model, sentinel_artifact = object(), object()
    r = KNNRouter(data=object(), model_ids=["a", "b"],
                  cost_model=sentinel_cost_model, artifact=sentinel_artifact)
    assert r.cost_model is sentinel_cost_model
    assert r.artifact is sentinel_artifact


def test_mlp_router_accepts_cost_model_and_artifact():
    sentinel_cost_model, sentinel_artifact = object(), object()
    r = MLPRouter(object(), ["a", "b"], data=object(),
                  cost_model=sentinel_cost_model, artifact=sentinel_artifact)
    assert r.cost_model is sentinel_cost_model
    assert r.artifact is sentinel_artifact
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/router/routing/test_routing_interface.py tests/router/routing/test_router_model_subclass_init.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'cost_model'` for every new test.

- [ ] **Step 3: Add the keyword-only parameters and forward them**

In `src/router/routing/routers.py`:

`MatrixRouter.__init__` (~L151-160):
```python
    def __init__(self, scores: pd.DataFrame, *, name: Optional[str] = None,
                 model_costs: Optional[Sequence[float]] = None,
                 cost_model: Optional["CostModel"] = None,
                 artifact: Optional["RouterModelArtifact"] = None):
        super().__init__(list(scores.columns), name=name, cost_model=cost_model, artifact=artifact)
```
(body unchanged below).

`NIRTRouter.__init__` (~L184-206):
```python
    def __init__(
        self,
        model,
        model_index: dict,
        *,
        data=None,
        pathway: str = "irt",
        query_pathway: Optional[str] = None,
        query_features: Optional[str] = None,
        name: Optional[str] = None,
        cost_model: Optional["CostModel"] = None,
        artifact: Optional["RouterModelArtifact"] = None,
    ):
        super().__init__(sorted(model_index, key=model_index.get), name=name,
                         cost_model=cost_model, artifact=artifact)
```
(body unchanged below).

`NIRTRouter.from_artifact` (~L252-263): replace the trailing post-assignment with constructor args —
```python
        router = cls(
            model,
            payload.model_index,
            data=load_training_data_for_router(data, dcfg.get("phase0_config")),
            pathway=dcfg.get("pathway", "irt"),
            query_pathway=dcfg.get("query_pathway") or None,
            query_features=dcfg.get("query_features") or None,
            name=name or f"nirt:{artifact.artifact_id}",
            cost_model=cost_model,
            artifact=artifact,
        )
        return router
```

`KNNRouter.__init__` (~L387-400):
```python
    def __init__(
        self,
        data=None,
        *,
        k: int = 5,
        pathway: str = "retrieval",
        model_ids: Optional[Sequence[str]] = None,
        name: Optional[str] = None,
        cost_model: Optional["CostModel"] = None,
        artifact: Optional["RouterModelArtifact"] = None,
    ):
        d = load_training_data_for_router(data)
        if model_ids is None:
            obs = d.nirt_observations()
            model_ids = sorted(obs.loc[obs["split"] == "train", "model_id"].astype(str).unique())
        super().__init__(model_ids, name=name or f"knn(k={k})", cost_model=cost_model, artifact=artifact)
```

`MLPRouter.__init__` (~L459-465):
```python
    def __init__(self, model, model_ids: Sequence[str], *, data=None,
                 pathway: str = "irt", name: Optional[str] = None,
                 cost_model: Optional["CostModel"] = None,
                 artifact: Optional["RouterModelArtifact"] = None):
        super().__init__(model_ids, name=name or "mlp", cost_model=cost_model, artifact=artifact)
```

`MLPRouter.from_artifact` (~L487-492): replace the trailing post-assignment —
```python
        router = cls(
            model, payload.model_ids, data=data, name=name or f"mlp:{artifact.artifact_id}",
            cost_model=cost_model, artifact=artifact,
        )
        return router
```

`RandomRouter.__init__` (~L520-522):
```python
    def __init__(self, model_ids: Sequence[str], *, seed: int = 0, name: Optional[str] = None,
                 cost_model: Optional["CostModel"] = None,
                 artifact: Optional["RouterModelArtifact"] = None):
        super().__init__(model_ids, name=name or "random", cost_model=cost_model, artifact=artifact)
```

Add `"CostModel"`/`"RouterModelArtifact"` to the `TYPE_CHECKING` imports at the top of `routers.py` if not already present (check the existing `if TYPE_CHECKING:` block before adding — do not duplicate an import).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/routing/test_routing_interface.py tests/router/routing/test_router_model_subclass_init.py tests/router/models/test_factory.py tests/training/trainers/test_mlp_router_trainer.py -v`
Expected: PASS (including the pre-existing `test_factory_create_matches_from_run`'s `router.cost_model is cost_model` / `router.artifact is artifact` assertions, which must keep passing unchanged).

- [ ] **Step 5: Run the full test suite for regressions**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass, same count as baseline (853) plus the new tests.

- [ ] **Step 6: Commit**

```bash
git add src/router/routing/routers.py tests/router/routing/test_routing_interface.py tests/router/routing/test_router_model_subclass_init.py
git commit -m "fix(router): forward cost_model/artifact from every RouterModel subclass constructor (ROUTERMODEL-004)"
```

---

## Task 2: `default_model_costs` consumes `self.cost_model` when set

**Files:**
- Modify: `src/router/routing/routers.py` (`_lazy_model_costs`, ~L83-89)
- Test: `tests/router/models/test_factory.py` or a new focused test in `tests/router/routing/test_router_model_subclass_init.py`

**Interfaces:**
- Consumes: `router.cost_model: Optional[CostModel]` (Task 1's constructor param), `CostModel.predict_cost(model: LLMProfile, input_tokens: int, output_tokens: int) -> float` (`router/llm/cost.py:44-46`, unchanged) and `CostModel.output_token_priors`.
- Produces: `_lazy_model_costs(router)` returns a `cost_model`-derived vector (one float per `router._model_ids`, same order) when `router.cost_model is not None`, else the existing train-mean fallback (`_mean_train_costs`) exactly as before.

`CostModel.predict_cost` needs an `LLMProfile`-shaped object (`model_id`, `input_cost_per_token`, `output_cost_per_token`) per model, which `RouterModel` does not have (it only knows string `model_ids`). Rather than requiring a full `LLMProfile` per pool id (a bigger, separate wiring problem the bug file does not ask for), use `CostModel.output_token_priors` directly as a **per-model-id USD prior**, the same shape `default_model_costs` already returns (one float per pool model, USD/query scale) — this is the minimal reading of "consume it (use `self.cost_model` in `default_model_costs` when set)" from the bug's suggested fix, and requires no new type. Document the scale assumption in the code comment since it's a real judgment call.

- [ ] **Step 1: Write the failing test**

In `tests/router/routing/test_router_model_subclass_init.py`, add:

```python
from router.llm.cost import CostModel


def test_default_model_costs_uses_the_injected_cost_model_when_set():
    """ROUTERMODEL-002: a CostModel passed at construction must change the
    router's cost vector, not be silently ignored in favour of the
    train-mean cost."""
    cost_model = CostModel(output_token_priors={"a": 0.02, "b": 0.05})
    r = KNNRouter(data=object(), model_ids=["a", "b"], cost_model=cost_model)
    costs = r.default_model_costs
    assert list(costs) == [0.02, 0.05]


def test_default_model_costs_falls_back_without_a_cost_model():
    """Unchanged behaviour: no cost_model -> the existing train-mean path,
    not a crash and not all-zero."""
    r = KNNRouter(data=object(), model_ids=["a", "b"])
    assert r.cost_model is None
    # no crash constructing the router; the train-mean path is exercised
    # by test_factory.py's fitted-artifact tests, not re-asserted here.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/router/routing/test_router_model_subclass_init.py::test_default_model_costs_uses_the_injected_cost_model_when_set -v`
Expected: FAIL — `costs` comes back from the train-mean path (`_mean_train_costs`), not `[0.02, 0.05]` (likely errors first, since `data=object()` has no `nirt_observations()`; that's fine, it proves the cost_model isn't consulted before the train-mean path is attempted).

- [ ] **Step 3: Implement**

In `src/router/routing/routers.py`, replace `_lazy_model_costs`:

```python
def _lazy_model_costs(router) -> Optional[np.ndarray]:
    """Shared ``default_model_costs`` body for a data-backed router. When the
    router carries an injected ``cost_model`` (``RouterModel.cost_model``,
    ROUTERMODEL-002), read one USD/query prior per pool model from its
    ``output_token_priors`` -- the same per-model-id, USD-scale shape this
    property already returns, so this doesn't change what any caller compares
    it against. Falls back to the lazily-cached train-split mean cost
    (``router._cost_cache``, set to ``_UNSET`` in ``__init__``) when there is
    no cost_model, exactly as before."""
    cost_model = getattr(router, "cost_model", None)
    if cost_model is not None:
        priors = cost_model.output_token_priors
        return np.array([float(priors.get(m, 0.0)) for m in router._model_ids], dtype=np.float64)
    if router._cost_cache is _UNSET:
        router._cost_cache = _mean_train_costs(router._data, router._model_ids)
    return router._cost_cache
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/routing/test_router_model_subclass_init.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite (this touches a shared helper used by NIRTRouter/KNNRouter/MLPRouter)**

Run: `.venv/Scripts/python.exe -m pytest tests/router tests/training -q`
Expected: all pass, no change to any test that doesn't set `cost_model`.

- [ ] **Step 6: Commit**

```bash
git add src/router/routing/routers.py tests/router/routing/test_router_model_subclass_init.py
git commit -m "fix(router): default_model_costs reads the injected CostModel when set (ROUTERMODEL-002)"
```

---

## Task 3: no-registry mode + `required_capabilities` warns instead of silently routing on nothing

**Files:**
- Modify: `src/router/router.py` (`Router.route`, the `_StubProfile` construction around current L128-133, and the `_StubProfile` class at the bottom of the file)
- Test: `tests/router/test_router_pipeline.py`

**Interfaces:**
- Consumes: `RoutingConstraints.required_capabilities: list[str]` (`router/constraints.py:56`, unchanged), `RouterModel.filter_candidates` (`router/routing/base.py:201-225`, unchanged — the fix is entirely in `Router.route` and `_StubProfile`, not in `filter_candidates`).
- Produces: `Router(router_model, llm_registry=None).route(request)` with a non-empty `request.constraints.required_capabilities` now emits a `UserWarning` (matching the existing `RC-01` style two lines above it) and candidates are **not** dropped by the capability check in this mode — `_StubProfile.capabilities` becomes `None` (*unknown*, not *known-empty*), and `filter_candidates`'s capability check already only applies `required.issubset(set(getattr(c, "capabilities", []) or []))`; changing the stub to expose no usable `capabilities` at all requires one extra guard.

Because `filter_candidates` currently does `set(getattr(c, "capabilities", []) or [])` — which turns `None` into `[]` same as an empty list — the minimal correct fix has two parts: (a) `_StubProfile.capabilities = None` to make "unknown" observable, and (b) `filter_candidates` must skip the capability check (not just default to `[]`) when a candidate's `capabilities` is `None`. This is a change to shared logic in `router/routing/base.py`, so re-verify Task 1's tests still pass afterward (they don't touch `capabilities`, but `filter_candidates` is exercised by `test_router_scaffolding.py`).

- [ ] **Step 1: Write the failing test**

In `tests/router/test_router_pipeline.py`, add near `test_required_capability_filters_out_unsupported_candidates`:

```python
def test_required_capability_with_no_registry_warns_and_keeps_candidates():
    """ROUTER-003: no-registry mode can't know any candidate's capabilities.
    Silently emptying every candidate (the old behaviour) must become a
    warning plus keeping the pool, not a silent, unexplained empty result."""
    constraints = RoutingConstraints(required_capabilities=["vision"])
    pipeline = Router(FakeTextRouter(), llm_registry=None)
    with pytest.warns(UserWarning, match="required_capabilities"):
        decision = pipeline.route(
            RoutingRequest(request_id="r4", prompt="anything", constraints=constraints)
        )
    assert len(decision.ranked_models) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/router/test_router_pipeline.py::test_required_capability_with_no_registry_warns_and_keeps_candidates -v`
Expected: FAIL — no warning raised, and `decision.ranked_models` is empty (`len(...) > 0` fails).

- [ ] **Step 3: Implement**

In `src/router/router.py`, in `route()` (current ~L128-133), extend the no-registry branch:

```python
        context = self.build_context(request)
        if self.llm_registry is None:
            # no registry at all -> the router's bare ids are the candidates. An
            # *empty* registry is a real "no candidates" and must not fall back.
            if constraints.required_capabilities:
                warnings.warn(
                    "RoutingConstraints.required_capabilities is set, but no LLMRegistry "
                    "is configured -- candidate capabilities are unknown in this mode, so "
                    "the constraint can't be checked and is not applied (ROUTER-003)",
                    stacklevel=2,
                )
            context.candidates = [_StubProfile(m) for m in self.router_model.model_ids]
        filtered = self.router_model.filter_candidates(context.candidates, request.constraints)
```

At the bottom of `src/router/router.py`, change `_StubProfile`:

```python
class _StubProfile:
    """Wraps a bare ``model_id`` string as a minimal candidate when no
    :class:`~router.llm.registry.LLMRegistry` was given -- lets
    :meth:`Router.route` run against any existing ``RouterModel`` (which
    only knows string ids) without requiring the newer ``LLMProfile`` layer
    to be populated first. ``capabilities`` is ``None`` (*unknown*), not
    ``[]`` (*known, has none*) -- see ``filter_candidates``'s capability
    check and ROUTER-003."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.capabilities: Optional[list[str]] = None
        self.output_cost_per_token = 0.0
```

In `src/router/routing/base.py`, in `filter_candidates` (~L201-225), change the capability check:

```python
            capabilities = getattr(c, "capabilities", None)
            if required and capabilities is not None and not required.issubset(set(capabilities)):
                continue
```

(replacing the current `if required and not required.issubset(set(getattr(c, "capabilities", []) or [])):`). A candidate with `capabilities=None` (unknown) now always passes the capability check; a candidate with `capabilities=[]` (known, has none) is still dropped exactly as before when `required` is non-empty — this preserves `test_filter_candidates_requires_capability`'s existing behaviour (`_Profile` there always sets a concrete list).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/test_router_pipeline.py tests/router/routing/test_router_scaffolding.py -v`
Expected: PASS, including every pre-existing test in both files (`test_filter_candidates_requires_capability` must still pass unchanged).

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/router/router.py src/router/routing/base.py tests/router/test_router_pipeline.py
git commit -m "fix(router): no-registry + required_capabilities warns and keeps candidates instead of silently emptying routing (ROUTER-003)"
```

---

## Task 4: `MLPRouterTrainer` persists `pathway`/`query_pathway`/`query_features`

**Files:**
- Modify: `src/router/models/artifacts.py` (`MLPArtifactPayload`), `src/training/trainers/mlp_router_trainer.py` (`MLPRouterTrainer.train`)
- Test: `tests/training/trainers/test_mlp_router_trainer.py`, `tests/router/models/test_artifacts.py`

**Interfaces:**
- Consumes: `fit_mlp_router(dataset, *, pathway="irt", query_pathway=None, query_features=None, ...)` (`training/trainers/mlp_router.py:35-52`, unchanged).
- Produces: `MLPArtifactPayload` gains three fields — `pathway: str = "irt"`, `query_pathway: Optional[str] = None`, `query_features: Optional[str] = None` — round-tripped through `to_state`/`from_state` (with `from_state` defaulting missing keys for old artifacts, per the Global Constraints backward-compat requirement). `MLPRouterTrainer.train` reads the resolved `pathway`/`query_pathway`/`query_features` out of the same `kw` dict it already builds and passes to `fit_mlp_router`, and records them on the payload it saves.

- [ ] **Step 1: Write the failing tests**

In `tests/router/models/test_artifacts.py`, add:

```python
def test_mlp_artifact_payload_round_trips_pathway_fields():
    payload = MLPArtifactPayload(
        state_dict={}, model_ids=["a", "b"], in_dim=8, hidden=16, dropout=0.1,
        pathway="bert", query_pathway="bert-query", query_features="len",
    )
    restored = MLPArtifactPayload.from_state(payload.to_state())
    assert restored.pathway == "bert"
    assert restored.query_pathway == "bert-query"
    assert restored.query_features == "len"


def test_mlp_artifact_payload_from_state_defaults_pathway_for_old_artifacts():
    """Backward compatibility: an artifact saved before this field existed
    has no pathway/query_pathway/query_features keys at all."""
    old_state = {
        "format": "mlp", "state_dict": {}, "model_ids": ["a"],
        "in_dim": 4, "hidden": 8, "dropout": 0.1,
    }
    restored = MLPArtifactPayload.from_state(old_state)
    assert restored.pathway == "irt"
    assert restored.query_pathway is None
    assert restored.query_features is None
```

(Check the top of `tests/router/models/test_artifacts.py` for the existing `MLPArtifactPayload` import and add these next to the file's other `MLPArtifactPayload` tests, if any.)

In `tests/training/trainers/test_mlp_router_trainer.py`, add:

```python
def test_train_persists_the_pathway_hyperparameters(tmp_path):
    """MLPROUTERTRAINER-001: pathway/query_pathway/query_features must survive
    into the saved artifact, not just fit_mlp_router's in-memory call."""
    d = FakeTrainingData(make_split_obs(seed=15), query_dim=16)
    trainer = MLPRouterTrainer(runs_dir=tmp_path)
    config = TrainingConfig(
        run_name="mlp-run-pathway",
        hyperparameters={"hidden": 16, "epochs": 2, "pathway": "irt", "query_features": None},
    )
    artifact = trainer.train(d, config)
    assert artifact.payload.pathway == "irt"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/router/models/test_artifacts.py tests/training/trainers/test_mlp_router_trainer.py::test_train_persists_the_pathway_hyperparameters -v`
Expected: FAIL — `MLPArtifactPayload.__init__() got an unexpected keyword argument 'pathway'`.

- [ ] **Step 3: Implement**

In `src/router/models/artifacts.py`, replace `MLPArtifactPayload`:

```python
@dataclass
class MLPArtifactPayload(ArtifactPayload):
    """The IRT-free ``e_q -> R^M`` MLP router (:func:`router.nirt.baselines_infer.build_mlp_router`).

    Unlike NIRT/baseline, ``training.trainers.mlp_router.fit_mlp_router``
    persisted nothing before :class:`~training.trainers.mlp_router.MLPRouterTrainer`
    -- there is no prior on-disk convention this wraps, this *is* the
    convention. Rebuild with ``build_mlp_router(in_dim, len(model_ids), hidden,
    dropout)`` + ``load_state_dict``.

    ``pathway``/``query_pathway``/``query_features`` mirror the NIRT artifact's
    ``config["data"]`` keys (MLPROUTERTRAINER-001) -- without them
    ``MLPRouter.from_artifact`` can't know which embedding space the model was
    trained on and silently serves the wrong one (MLPROUTER-001). Defaulted in
    :meth:`from_state` so an artifact saved before this field existed still
    loads, at the same ``pathway="irt"`` default the router constructor used
    before either bug was fixed."""

    state_dict: dict[str, Any]
    model_ids: list[str]
    in_dim: int
    hidden: int
    dropout: float
    pathway: str = "irt"
    query_pathway: Any = None
    query_features: Any = None

    def to_state(self) -> dict[str, Any]:
        return {
            "format": "mlp",
            "state_dict": self.state_dict,
            "model_ids": self.model_ids,
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "dropout": self.dropout,
            "pathway": self.pathway,
            "query_pathway": self.query_pathway,
            "query_features": self.query_features,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "MLPArtifactPayload":
        return cls(
            state_dict=state["state_dict"],
            model_ids=state["model_ids"],
            in_dim=state["in_dim"],
            hidden=state["hidden"],
            dropout=state["dropout"],
            pathway=state.get("pathway", "irt"),
            query_pathway=state.get("query_pathway"),
            query_features=state.get("query_features"),
        )
```

In `src/training/trainers/mlp_router_trainer.py`, in `train()`:

```python
        model, model_ids = fit_mlp_router(dataset, **kw)
        self.last_result = (model, model_ids)

        payload = MLPArtifactPayload(
            state_dict=model.state_dict(),
            model_ids=list(model_ids),
            in_dim=int(model[0].in_features),
            hidden=int(model[0].out_features),
            dropout=float(model[2].p),
            pathway=kw.get("pathway", "irt"),
            query_pathway=kw.get("query_pathway"),
            query_features=kw.get("query_features"),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/models/test_artifacts.py tests/training/trainers/test_mlp_router_trainer.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/router/models/artifacts.py src/training/trainers/mlp_router_trainer.py tests/router/models/test_artifacts.py tests/training/trainers/test_mlp_router_trainer.py
git commit -m "fix(training): MLPRouterTrainer persists pathway/query_pathway/query_features (MLPROUTERTRAINER-001)"
```

---

## Task 5: `MLPRouter.from_artifact` restores the trained pathway instead of the default

**Files:**
- Modify: `src/router/routing/routers.py` (`MLPRouter.__init__`, `.from_artifact`, `.predict_scores`)
- Test: `tests/training/trainers/test_mlp_router_trainer.py` (extend `test_train_persists_and_reloads_a_working_router`), `tests/router/routing/test_router_model_subclass_init.py`

**Interfaces:**
- Consumes: Task 4's `MLPArtifactPayload.pathway`/`.query_pathway`/`.query_features`; `mlp_router_matrix(model, model_ids, data, eval_query_ids, *, pathway="irt", query_pathway=None, query_features=None)` (`router/nirt/baselines_infer.py:135-141`, already supports both kwargs — `MLPRouter.predict_scores` just wasn't passing them through).
- Produces: `MLPRouter.__init__` gains `query_pathway: Optional[str] = None, query_features: Optional[str] = None` (mirroring `NIRTRouter`'s constructor shape exactly); `predict_scores` forwards both to `mlp_router_matrix`; `from_artifact` reads `payload.pathway`/`.query_pathway`/`.query_features` instead of leaving the router on the constructor defaults.

- [ ] **Step 1: Write the failing test**

Extend `test_train_persists_and_reloads_a_working_router` in `tests/training/trainers/test_mlp_router_trainer.py` — change the `TrainingConfig` to use a non-default pathway pairing that would mis-score if `from_artifact` ignored it. Since the test's `FakeTrainingData` only has one real pathway wired (`"irt"`), the sharpest test for *this specific bug* asserts the payload's stored pathway reaches the reloaded router's private state directly:

```python
def test_from_artifact_restores_the_trained_pathway(tmp_path):
    """MLPROUTER-001: from_artifact must not silently fall back to the
    constructor's pathway="irt" default when the artifact recorded a
    different one."""
    d = FakeTrainingData(make_split_obs(seed=16), query_dim=16)
    trainer = MLPRouterTrainer(runs_dir=tmp_path)
    config = TrainingConfig(
        run_name="mlp-run-pathway-2",
        hyperparameters={"hidden": 16, "epochs": 2, "pathway": "bert",
                         "query_pathway": "bert-query", "query_features": "len"},
    )
    artifact = trainer.train(d, config)
    assert artifact.payload.pathway == "bert"

    router = MLPRouter.from_artifact(artifact, data=d)
    assert router._pathway == "bert"
    assert router._query_pathway == "bert-query"
    assert router._query_features == "len"
```

(This only exercises persistence + restoration, not a live `fit_mlp_router` run against a `"bert"` pathway that doesn't exist in `FakeTrainingData` — `predict_scores` is not called in this test.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/training/trainers/test_mlp_router_trainer.py::test_from_artifact_restores_the_trained_pathway -v`
Expected: FAIL — `router._pathway == "irt"` (the constructor default), and `router._query_pathway`/`router._query_features` don't exist as attributes yet (`AttributeError`).

- [ ] **Step 3: Implement**

In `src/router/routing/routers.py`, `MLPRouter.__init__` (post-Task-1 signature):

```python
    def __init__(self, model, model_ids: Sequence[str], *, data=None,
                 pathway: str = "irt", query_pathway: Optional[str] = None,
                 query_features: Optional[str] = None, name: Optional[str] = None,
                 cost_model: Optional["CostModel"] = None,
                 artifact: Optional["RouterModelArtifact"] = None):
        super().__init__(model_ids, name=name or "mlp", cost_model=cost_model, artifact=artifact)
        self._model = model
        self._data = load_training_data_for_router(data)
        self._pathway = pathway
        self._query_pathway = query_pathway or pathway
        self._query_features = query_features
        self._cost_cache = _UNSET
```

`MLPRouter.from_artifact`:

```python
        router = cls(
            model, payload.model_ids, data=data, name=name or f"mlp:{artifact.artifact_id}",
            pathway=payload.pathway, query_pathway=payload.query_pathway,
            query_features=payload.query_features,
            cost_model=cost_model, artifact=artifact,
        )
        return router
```

`MLPRouter.predict_scores`:

```python
    def predict_scores(self, query_ids: Sequence[str]) -> pd.DataFrame:
        from ..nirt.baselines_infer import mlp_router_matrix

        return mlp_router_matrix(
            self._model, self._model_ids, self._data,
            [str(q) for q in query_ids], pathway=self._pathway,
            query_pathway=self._query_pathway, query_features=self._query_features,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/trainers/test_mlp_router_trainer.py tests/router/routing/test_router_model_subclass_init.py -v`
Expected: PASS, including the original `test_train_persists_and_reloads_a_working_router` (pathway defaults to `"irt"` there, so `mlp_router_matrix`'s comparison against `m_a`/`m_b` is unaffected — `query_pathway`/`query_features` default to `None`/derived-from-`pathway`, matching what that test already computes).

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass, count = baseline (853) + all new tests across Tasks 1-5.

- [ ] **Step 6: Commit**

```bash
git add src/router/routing/routers.py tests/training/trainers/test_mlp_router_trainer.py
git commit -m "fix(router): MLPRouter.from_artifact restores the trained pathway/query_pathway/query_features (MLPROUTER-001)"
```

---

## Final step: update `known_bugs/`

After all five tasks are green, update status to `fixed` (with the fixing commit's summary, following the existing convention in each file) in:
- `fixed_bugs/RouterModel/ROUTERMODEL-004-subclass-init-drops-cost-model-and-artifact/bug.md`
- `fixed_bugs/RouterModel/ROUTERMODEL-002-cost-model-never-consumed/bug.md`
- `fixed_bugs/Router/ROUTER-003-no-registry-capability-filter-drops-everything/bug.md`
- `fixed_bugs/RouterModelTrainer/MLPRouterTrainer/MLPROUTERTRAINER-001-hyperparameters-not-persisted/bug.md`
- `fixed_bugs/RouterModel/MLPRouter/MLPROUTER-001-from-artifact-loses-pathway-and-data-config/bug.md`

And update the `## Bugs` table rows and the `## Coverage` table's `RouterModel`/`RouterModelTrainer`/`MLPRouterTrainer` lines in `known_bugs/README.md` to reflect `fixed`, the same way the LLM-prefixed bugs were updated earlier in this session.
