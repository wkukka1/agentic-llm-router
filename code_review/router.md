# Code review: router (top-level modules + router.models)

**Files reviewed:** 17 (709 lines): `__init__.py`, `_normalize.py`, `config.py`, `constraints.py`, `context.py`, `decision.py`, `determinism.py`, `policy.py`, `provenance.py`, `router.py`, `models/__init__.py`, `models/artifacts.py`, `tools/__init__.py`, `tools/base.py`, `tools/mcp.py`, `tools/router_tool.py`, `tracing/__init__.py`, `tracing/traces.py` · **Context-only:** `src/evaluation/routing/oracle.py`, `src/training/data/facade.py`
**Summary:** 9 findings: 6 correctness, 2 design, 1 performance · 9 confirmed, 0 suspected

Sections for `router/tools/` and `router/tracing/` (RT-50+) come from a second reviewer and are merged below.
Behavioural claims marked "probe" were checked by running the real code (`.venv/Scripts/python.exe`).

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| RT-01 | `src/router/router.py:83-90` | correctness | medium | confirmed |
| RT-02 | `src/router/policy.py:58-69` | correctness | medium | confirmed |
| RT-03 | `src/router/router.py:93-104` | design | low | confirmed |
| RT-04 | `src/router/config.py:32-37` | correctness | low | confirmed |
| RT-05 | `src/router/config.py:100-107` | correctness | low | confirmed |
| RT-06 | `src/router/determinism.py:15-26` | design | low | confirmed |
| RT-07 | `src/router/provenance.py:43-50` | performance | low | confirmed |
| RT-08 | `src/router/provenance.py:17-40` | correctness | low | confirmed |
| RT-50 | `src/router/tools/router_tool.py:39-52` | correctness | low | confirmed |

---

## src/router/router.py

### RT-01: An empty `LLMRegistry` falls back to the router's whole pool instead of "no candidates" `:83-90`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
84:        candidates = context.candidates or [
85:            _StubProfile(m) for m in self.router_model.model_ids
86:        ]
```

**What breaks:** `build_context` returns `[]` both when no registry was given and when a registry was given but lists nothing (e.g. every profile disabled or removed). The `or` treats both the same, so a deployment whose registry is empty routes to arbitrary router-pool models that have no `LLMProfile`, and therefore no client, capabilities or costs. `_StubProfile.capabilities=[]`, so any `required_capabilities` constraint then filters everything out, while a request without capabilities goes through to unregistered models. `context.candidates` stays `[]`, so the policy receives a context that doesn't match the candidates that were scored.
**Evidence:** `test_route_without_a_registry_falls_back_to_bare_model_ids` (`tests/router/test_router_pipeline.py:65-71`) covers only `llm_registry=None`. No test covers an empty registry.
**Direction:** Fall back only when `self.llm_registry is None`, and write the resolved candidates back into `context.candidates`.

---

### RT-03: The pipeline's cost signal (per-output-token price) is inconsistent with the router's (train-mean USD/query) `:93-104`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
96:                score=float(value),
97:                expected_quality=float(value),
98:                expected_cost=getattr(by_id[model_id], "output_cost_per_token", 0.0),
```

**What breaks:** `Router.route(lam=...)` trades quality off against `default_model_costs`, the train-split mean USD per query (`routing/routers.py:63-77`). `RoutingPipeline` ignores that vector and feeds `OptimizationObjective.utility` a per-token list price, which is several orders of magnitude smaller and ignores input tokens and response length. A `cost_weight` tuned against one scale is meaningless on the other, so the pipeline and `AgenticRouter` (which uses `lam`) can't be configured to make the same decision. This also means `OptimizationObjective.utility` (`constraints.py:29-36`) is not the single definition of the utility its docstring claims (`constraints.py:19-22`); `routing_decision` has a second one.
**Evidence:** Traced `router.py:90-105` → `policy.py:59-60` → `constraints.py:29-36`, and `routing/base.py:261-271` → `nirt/routing_decision.py:38-39`. `test_cost_weight_can_change_the_winner` uses `output_cost_per_token=10.0`, an unrealistic value that hides the scale mismatch.
**Direction:** Populate `expected_cost` from the router's `default_model_costs` (or a per-request estimate), and make `routing_decision` the special case `quality_weight=1, cost_weight=lam` of one utility function.

---

## src/router/policy.py

### RT-02: NaN scores make the `DefaultRoutingPolicy` ranking undefined, and a NaN model can rank first `:58-69`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
59:        objective = context.request.constraints.objective
60:        ranked = sorted(scores, key=objective.utility, reverse=True)
61:        top = ranked[0] if ranked else None
```

**What breaks:** Every comparison with NaN is False, so `sorted` leaves NaN items where they happen to fall. Probe: utilities `[nan, 0.2, 0.9]` rank as `[nan, 0.9, 0.2]`, so the model with *no prediction* is chosen. `[0.2, nan, 0.9, 0.5]` stays completely unsorted. NaN reaches here from `RoutingPipeline.route` (`router.py:93-104`), which copies `route_text` scores without a finiteness check. Realistic sources:
- `KNNRouter(model_ids=...)` including a model absent from train observations, which `knn_router_matrix_from_embeddings` reindexes to a NaN column.
- Any custom router whose `predict_scores_text` omits a pool model, which `_route_from_scores` reindexes to NaN.

Unlike `Router.route` (`routing_decision` masks NaN to −inf), this path has no guard.
**Evidence:** Probe. `tests/router/test_router_pipeline.py` uses only finite constant scores.
**Direction:** Drop or mark `supported=False` any `ModelScore` with non-finite quality in `RoutingPipeline.route`, and/or sort with a key that maps NaN to −inf.

---

## src/router/config.py

### RT-04: `Config.__getattr__` recurses infinitely on `copy.copy`, `copy.deepcopy` and `pickle` `:32-37`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
32:    def __getattr__(self, name: str) -> Any:
33:        try:
34:            value = self._data[name]
```

**What breaks:** The copy protocols build an instance without calling `__init__` and then probe dunder attributes such as `__setstate__`. `__getattr__` accesses `self._data`, which doesn't exist yet, so `__getattr__('_data')` is called again and again. The result is `RecursionError`, which `hasattr` doesn't swallow. Probe: `copy.deepcopy(cfg)`, `copy.copy(cfg)` and `pickle.loads(pickle.dumps(cfg))` all raise `RecursionError`. Any future multiprocessing (Windows spawn pickles arguments), `DataLoader(num_workers>0)` over an object holding a `Config` (`TrainingData.cfg`), or a defensive `deepcopy` crashes.
**Evidence:** Probe. Grep for `multiprocessing|joblib|num_workers|copy.copy(|pickle.dump|deepcopy` in src/scripts: no current hits, so the bug is latent.
**Direction:** Start `__getattr__` with `if name.startswith("_"): raise AttributeError(name)`, or read `self.__dict__.get("_data")`.

---

### RT-05: `coerce_auto_bool("false")` returns `True`, and `coerce_hidden("0")` returns width 0 `:100-107`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
100:    if value in ("none", "None", 0, None):
101:        return None
107:    return bool(auto) if value in ("auto", None) else bool(value)
```

**What breaks:** Any non-empty string is truthy, so a string-typed override (`"false"`, `"False"`, `"0"`, `"no"`) for `constrain_discrimination` (`nirt/model.py:237`, `baseline/model.py:101`) enables it. `coerce_hidden` treats integer `0` as "linear head" but string `"0"` as `int("0") == 0`, which builds `nn.Linear(in, 0)`. YAML booleans and integers are parsed correctly, so this only happens when values arrive as strings (env or CLI overrides, JSON sweeps, `--set k=v` tooling). `train_nirt.py:62` parses `--query-hidden` itself, so today's CLI isn't affected. Probe: `coerce_auto_bool("false", True) -> True`, `coerce_hidden("0") -> 0`.
**Evidence:** Probe. `tests/training/data/test_families.py:30-41` covers only native bool/int/None inputs.
**Direction:** Normalise strings (`"true"/"false"/"0"/"1"/"null"`) before coercion, or raise on unrecognised strings.

---

## src/router/determinism.py

### RT-06: Setting `PYTHONHASHSEED` at runtime has no effect, and deterministic-algorithms mode is a process-wide side effect of encoding `:15-26`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
16:    os.environ.setdefault("PYTHONHASHSEED", str(seed))
24:        torch.use_deterministic_algorithms(True, warn_only=True)
```

**What breaks:**
- **Line 16:** hash randomisation is fixed at interpreter start, so this line doesn't affect the current process. Because it is `setdefault`, it also can't override an inherited value for children. Anything relying on `str` hash / set iteration order is not actually pinned.
- **Line 24:** `seed_everything` is called from the serving-side encoder (`router/embeddings/encoder.py:99,128`). This silently switches the *whole process* (including a co-located NIRT training or GPU inference) into deterministic-algorithms mode, which is slower on CUDA and emits warnings without `CUBLAS_WORKSPACE_CONFIG`. It also reseeds global RNGs on every encoder load, which resets any RNG stream a caller had advanced.
**Evidence:** Callers found by grep: `encoder.py:99,128`, `training/nirt/train.py:334`, `baseline/train.py:92`, `trainers/mlp_router.py:36`, `taxonomy/clustering.py:96`, `classical_irt.py:75`.
**Direction:** Drop the env line or document it as child-process only. Split "seed RNGs" from "enable deterministic algorithms", and don't do the latter on the serving encoder path.

---

## src/router/provenance.py

### RT-07: `file_digest` reads the whole file into memory `:43-50`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
48:    h = hashlib.new(algo)
49:    h.update(p.read_bytes())
50:    return h.hexdigest()[:n]
```

**What breaks:** Hashing large phase-0 artifacts (parquet response tables, embedding matrices) allocates their full size in RAM. `training/nirt/baseline/train.py:241-250` hashes phase-0 artifacts under `processed/` and `taxonomy/` for every baseline run, and `evaluation/pool_expansion/battery.py:470-471` hashes files through `_sha256`.
**Evidence:** Callers found by grep: `baseline/train.py:242`, `battery.py:471`.
**Direction:** Stream in chunks (`for chunk in iter(lambda: f.read(1<<20), b"")`) or use `hashlib.file_digest` (Python 3.11+).

---

### RT-08: `git_sha()` / `git_dirty()` with no `cwd` describe whatever repo the process cwd is in `:17-40`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
20:        out = subprocess.check_output(
21:            ["git", "rev-parse", "HEAD"],
22:            stderr=subprocess.DEVNULL,
```

**What breaks:** With `cwd=None`, git runs in the process working directory. A checkpoint written from a notebook or job launched outside the repo records `git_sha: null`. Launched from inside a *different* repo, it records the wrong commit. `training/nirt/baseline/checkpoint.py:52` calls `git_sha()` with no `cwd`; `battery.py:476` passes `cfg.root` correctly. `training/nirt/train.py:617` uses its own `_git_sha`, a separate implementation, despite this module's docstring saying it's the single copy.
**Evidence:** Grep for `git_sha(`: `baseline/checkpoint.py:52` (no cwd), `battery.py:476` (root), `train.py:617` (`_git_sha`, local).
**Direction:** Default `cwd` to `router.config.REPO_ROOT`, and route `train.py` through this helper.

---

## src/router/constraints.py

No separate finding. The unenforced `max_cost` / `max_latency` / `min_quality` "hard filters" are reported at the enforcement site as **RR-05** (`code_review/router.routing.md`). The two divergent utility definitions are covered in RT-03.

---

## src/router/context.py

No findings.

---

## src/router/decision.py

No findings.

---

## src/router/__init__.py

No findings.

---

## src/router/_normalize.py

No findings.

---

## src/router/models/__init__.py

No findings.

---

## src/router/models/artifacts.py

No findings. `RouterModelArtifact` has no constructor call anywhere (grep), but it is documented scaffolding referenced by `routing/base.py:49` (TYPE_CHECKING) and the tracing layer, so it isn't reported as dead code.

---

## src/router/tools/base.py

No findings.

---

## src/router/tools/mcp.py

No findings. The `NotImplementedError` in `execute` is declared scaffolding.

---

## src/router/tools/router_tool.py

### RT-50: `route_for_model_selection` mutates the caller's request before raising `:39-52`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
47:        request.mode = RoutingMode.MODEL_SELECTION
48:        raise NotImplementedError(
```

**What breaks:** A caller that catches the `NotImplementedError` (for example to fall back to `AgenticRouter.route_decision`) is left holding a `RoutingRequest` whose `mode` is permanently `MODEL_SELECTION`. If that request is later passed to `RoutingPipeline` or an orchestrator, it would be treated as a tool-originated request that must not hand off. This is latent: nothing constructs `RouterTool` today (grep `RouterTool` finds only a TYPE_CHECKING import at `router/execution/orchestrator.py:27` and docstrings).
**Evidence:** Grep `RouterTool|route_for_model_selection` over `src scripts tests configs docs`. No test covers `router/tools` (there is no `tests/router/tools/` directory).
**Direction:** Build a copy (`dataclasses.replace(request, mode=...)`) instead of mutating the argument, or set the mode only after the decision is produced.

---

## src/router/tools/__init__.py

No findings.

---

## src/router/tracing/traces.py

No findings. The re-entry guard and duplicated provenance fields are cross-package concerns, filed as A6-X02 and A6-X04 in `_cross_cutting_raw/A6.md`.

---

## src/router/tracing/__init__.py

No findings.

---

---

## Checked and ruled out
- **`Config.get` returning `None` for a key explicitly set to `null` instead of `default`:** conventional, and callers (`coerce_hidden`) rely on explicit null meaning "linear head".
- **`Config.path` has no default:** documented contract (`_contracts.md` §5); a missing key raises KeyError loudly.
- **`section()` returning the live inner dict (not a copy):** callers read only; no in-repo mutation of a returned section found.
- **`unit_rows` with integer input or zero rows:** the division promotes to float and the norm floor avoids divide-by-zero. Covered by `test_families.py:44`.
- **`DefaultRoutingPolicy.select_destination` ignoring `escalation_threshold`:** documented as intentional in the class docstring.
- **`RoutingPipeline` leaving `lam=0` in `route_text`:** documented as deliberate, to avoid double-counting cost (but see RT-03 on the cost scale).
- **`context.py` importing `decompose` at runtime:** allowed direction (`decompose` must not import `router`, not vice versa); import-linter passes.
- **`RoutingPipeline.route` scoring the full pool before filtering:** one text encoding either way; filtering first wouldn't save the forward pass for NIRT's per-model loop meaningfully at pool sizes of about 20.
- **`MCPTool` / `RouterTool` / `ExecutionTrace` as dead code:** exempt. They are documented scaffolding referenced from `router/execution/orchestrator.py:26-27`, `router/context.py:31`, and `router/models/artifacts.py:7`.
- **`router_tool.py` importing `..context` at runtime (which imports top-level `decompose`):** allowed. The `decompose` contract restricts only what `decompose` imports, and import-linter passes.
- **`ExecutionTrace` having all required fields with no defaults:** no constructor exists to misuse.
