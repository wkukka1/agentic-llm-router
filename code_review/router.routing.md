# Code review: router.routing

**Files reviewed:** 4 (1001 lines): `src/router/routing/base.py` (371), `src/router/routing/routers.py` (509), `src/router/routing/registry.py` (56), `src/router/routing/__init__.py` (65) · **Context-only:** `src/router/router.py`, `src/router/models/artifacts.py`, `src/router/models/registry.py`, `src/router/tools/router_tool.py` (found broken by this diff, not itself in scope), `docs/routing_interface.md`

**Summary:** 5 findings: 2 correctness, 3 design, 0 performance · 3 confirmed, 2 plausible

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| RR-01 | `src/router/routing/base.py:308-321` | correctness | medium | confirmed |
| RR-02 | `src/router/routing/base.py:174-188`, `routers.py` (all `__init__`s), `:252-263` | design | medium | confirmed |
| RR-03 | `src/router/tools/router_tool.py:1,27,34,42` | correctness | low | confirmed |
| RR-04 | `src/router/routing/routers.py:207-263` | design | low | plausible |
| RR-05 | `src/router/routing/base.py:187`, `routers.py:261` | design | low | plausible |

---

## src/router/routing/base.py

### RR-01: `RoutingResult.fallback` doesn't see cost/`lam`-driven fallbacks that the actual selection already made `:308-321`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
308:        mat = scores.to_numpy(np.float64)
309:        selected = routing_decision(mat, lam=lam, model_costs=costs, eligible=elig)
310:        util = np.where(np.isfinite(mat), 0.0, -np.inf)
311:        if elig is not None:
312:            util = np.where(np.asarray(elig, bool), util, -np.inf)
313:        return RoutingResult(
...
320:            fallback=no_selectable_rows(util),
321:        )
```

**What breaks:** `routing_decision` (via `cost_aware_utility` in `router/nirt/routing_decision.py`) computes the *real* selection utility as `pred - lam * cost`, then re-masks any cell that becomes non-finite after the cost subtraction to `-inf` before arg-maxing — so a row where `lam * cost` overflows/NaNs for every model is correctly forced to column 0 by `routing_decision` itself. But the `fallback` flag returned to the caller is computed from a *separate*, second utility array (`util` on lines 310-312) built only from `np.isfinite(mat)` — i.e. raw score finiteness — and never subtracts `lam * costs` at all. If `lam` is non-finite (e.g. `float("nan")` or `float("inf")`, easy to get from an upstream ratio that divides by zero) and a cost vector is resolved (`costs is not None`), `cost_aware_utility` turns every cell to `-inf` and `routing_decision` silently forces `selected[:] = 0` for every query — but the locally recomputed `util` here still sees finite `mat` scores, so `no_selectable_rows(util)` is all-`False`. `RoutingResult.fallback`/`.any_fallback` then reports **no fallback occurred** while every single row was actually a meaningless column-0 pick. The class's own docstring says callers "must check `fallback`... before dispatching" — for this input, that check passes when it shouldn't.

Note: `costs` themselves can never be non-finite here (`_resolve_costs` and every `default_model_costs` implementation guarantee finite entries), so the only trigger is a non-finite `lam`, which nothing in `route()`/`_route_from_scores` validates.

**Evidence:** Traced `route()` → `_route_from_scores()` → `routing_decision()` → `cost_aware_utility()` (`src/router/nirt/routing_decision.py:17-45`) against the locally-recomputed `util` in `_route_from_scores`; the two use different "was this row selectable" definitions (one includes the cost term, the other doesn't), and only the cost-blind one is surfaced as `RoutingResult.fallback`.

**Direction:** Compute the fallback flag from the same utility `routing_decision`/`cost_aware_utility` actually used, e.g. call `cost_aware_utility(mat, lam=lam, model_costs=costs, eligible=elig)` once, derive both `selected` (via `.argmax(axis=1)` + the all-bad-row override) and `fallback` (`no_selectable_rows`) from that single array, instead of recomputing a cost-blind approximation. This also removes the duplicate computation (same array built twice, once inside `routing_decision`, once inline). Additionally consider validating `lam` is finite in `route()`/`route_text()` the same way `_resolve_costs` validates `model_costs`.

---

### RR-02: `cost_model`/`artifact` constructor params added to `RouterModel.__init__` are unreachable through every concrete subclass `:174-188`
**Category:** design · **Severity:** medium · **Confidence:** confirmed

```python
174:    def __init__(self, model_ids: Sequence[str], *, name: Optional[str] = None,
175:                 unsupported_policy: UnsupportedCandidatePolicy = UnsupportedCandidatePolicy.DROP,
176:                 cost_model: Optional["CostModel"] = None,
177:                 artifact: Optional["RouterModelArtifact"] = None):
...
184:        #: set by RouterModelFactory.create when this router was built from a
185:        #: RouterModelArtifact; None for the many routers still built directly
186:        #: (from_run, raw kwargs) with no artifact behind them.
187:        self.cost_model = cost_model
188:        self.artifact = artifact
```

**What breaks:** This diff adds `cost_model=`/`artifact=` keyword params to the ABC `RouterModel.__init__`, but *none* of the five concrete subclasses (`MatrixRouter`, `NIRTRouter`, `KNNRouter`, `MLPRouter`, `RandomRouter`) declare or forward these kwargs in their own `__init__` — every one of them calls `super().__init__(model_ids, name=...)` (and sometimes `unsupported_policy`) only. So `NIRTRouter(model, model_index, cost_model=cm)` — or `build_router("nirt", cost_model=cm)` via `router.routing.registry.build_router`, which passes `**kwargs` straight to the class — raises `TypeError: __init__() got an unexpected keyword argument 'cost_model'`, even though the base class the docstring points to (line 184: "set by `RouterModelFactory.create`") advertises it as a real constructor parameter. The one place that actually sets these attributes, `NIRTRouter.from_artifact` (`routers.py:261-262`), bypasses the constructor entirely and assigns `router.cost_model = cost_model` / `router.artifact = artifact` directly onto the already-built instance — so the parameters just added to the ABC's `__init__` are dead from every call site in the current codebase (confirmed by grep: the only reads are `router.cost_model`/`router.artifact` identity assertions in `tests/router/models/test_factory.py`, never consumed by any routing/cost logic).

**Evidence:** Read all five subclass `__init__`s in `routers.py` (151-160, 184-205, 387-406, 459-465, 493-495) — none accept or forward `cost_model`/`artifact`. Grepped the repo for `.cost_model` / `.artifact` usages — only test identity assertions and the two direct-assignment lines in `from_artifact`.

**Direction:** Either (a) have every concrete `__init__` accept and forward `cost_model`/`artifact` to `super().__init__(...)` so the base-class parameters are actually usable, or (b) drop them from `RouterModel.__init__`'s signature and keep them as plain post-construction attributes (as `from_artifact` already does), documenting that they're artifact-factory bookkeeping, not a constructor contract every router honors.

---

## src/router/tools/router_tool.py (cross-file fallout of the `base.py` rename)

### RR-03: stale `Router` symbol still referenced after `routing/base.py`'s `Router` → `RouterModel` rename `:1,27,34,42`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
1:"""``RouterTool``: intended to expose a :class:`~router.routing.base.Router`
...
25:if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
26:    from ..decision import RoutingDecision
27:    from ..routing.base import Router
...
34:    def __init__(self, router: "Router"):
...
42:        """Route ``request`` for model selection only: a plain pick, never a
...
42:        Real ranking (:meth:`Router.route`/``route_text``) works
```

**What breaks:** `router.routing.base.Router` no longer exists — it was renamed to `RouterModel` by this branch — but `router_tool.py` still imports and type-hints `Router` from `..routing.base`. It's inside `if TYPE_CHECKING:` with `from __future__ import annotations` in effect, so it doesn't raise `ImportError` at plain runtime import, but it does break any static type checker (mypy/pyright) run against this module, and — per the review brief's own warning about the new naming collision — it is now genuinely ambiguous to a reader: `router.py` (top-level, this same branch) defines a *different*, real `Router` class (the renamed `RoutingPipeline`), so `RouterTool.__init__(self, router: "Router")` reads as if it takes the new top-level orchestrator when the file's intent (a wrapper around the per-`(query,model)` scorer) is actually the class now called `RouterModel`.

**Evidence:** Grepped the whole repo for `routing.base import Router` / `routing\.base\.Router` — this is the only remaining hit; `evaluation/routing/oracle.py`, `router/models/registry.py`, `router/agentic/router.py`, and `tests/router/routing/test_router_scaffolding.py` were all correctly updated to `RouterModel`.

**Direction:** Update the import, docstring, and both type hints in `router_tool.py` to `RouterModel` (and clarify the docstring's `Router.route`/`route_text` mention, since `route_text` lives on `RouterModel`, not the top-level `Router`).

---

## src/router/routing/routers.py

### RR-04: `NIRTRouter.from_run` / `from_artifact` duplicate the entire construction block `:207-263`
**Category:** design · **Severity:** low · **Confidence:** plausible

```python
217:        return cls(
218:            model,
219:            model_index,
220:            data=load_training_data_for_router(data, dcfg.get("phase0_config")),
221:            pathway=dcfg.get("pathway", "irt"),
222:            query_pathway=dcfg.get("query_pathway") or None,
223:            query_features=dcfg.get("query_features") or None,
224:            name=name or f"nirt:{run_name}",
225:        )
...
252:        router = cls(
253:            model,
254:            payload.model_index,
255:            data=load_training_data_for_router(data, dcfg.get("phase0_config")),
256:            pathway=dcfg.get("pathway", "irt"),
257:            query_pathway=dcfg.get("query_pathway") or None,
258:            query_features=dcfg.get("query_features") or None,
259:            name=name or f"nirt:{artifact.artifact_id}",
260:        )
```

**What breaks:** Nothing today — both blocks are currently identical apart from the `model_index`/name source — but they're two independent copies of the same "read `dcfg`, build the router" logic. A future change to how any of `pathway`/`query_pathway`/`query_features`/`phase0_config` is resolved (e.g. a new default, a renamed config key) has to be made in both places, and the review brief's own concern about the `Router`/`RouterModel` rename slipping through untouched call sites is exactly the failure mode this invites: it's easy to update one classmethod and forget the other.

**Evidence:** Line-by-line diff of the two blocks — only `model`/`model_index` acquisition (checkpoint load vs. artifact payload + `build_model`) differs; the `cls(...)` call is copy-pasted.

**Direction:** Factor a shared private classmethod, e.g. `_from_model(cls, model, model_index, cfg, *, data, name)`, that both `from_run` and `from_artifact` call after they've each obtained `model`/`cfg`/`model_index` their own way.

---

### RR-05: `cost_model` is threaded through the artifact-construction path but never consulted by any routing/cost logic `:261`
**Category:** design · **Severity:** low · **Confidence:** plausible

```python
260:        )
261:        router.cost_model = cost_model
262:        router.artifact = artifact
263:        return router
```

**What breaks:** `RouterModelFactory.create(artifact, cost_model)` accepts and plumbs a `CostModel` all the way to `router.cost_model`, which reads as though it should influence cost-aware routing (`route(lam=...)` / `default_model_costs`). In fact `default_model_costs` (for `NIRTRouter`, via `_lazy_model_costs`/`_mean_train_costs`) still only ever derives cost from the train-split mean cost in the observation table — `self.cost_model` is write-only, read back nowhere in `base.py` or `routers.py`. A caller who supplies a real `CostModel` to steer `lam`-weighted selection differently per model/provider will see no behavioral change; today the attribute is exercised only by identity assertions in `tests/router/models/test_factory.py`.

**Evidence:** Grepped for `.cost_model` across `src/` — only the assignment sites in `base.py:187` and `routers.py:261`, no reads.

**Direction:** Either wire `default_model_costs` (or a new override point) to prefer `self.cost_model` when present, or note in the docstring that `cost_model` is currently identity/bookkeeping only and not yet consulted by routing, so it isn't mistaken for a working cost-override mechanism.
