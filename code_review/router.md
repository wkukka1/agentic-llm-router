# Code review: router

**Files reviewed:** 6 (425 lines): src/router/__init__.py, src/router/constraints.py, src/router/context.py, src/router/decision.py, src/router/policy.py, src/router/router.py · **Context-only:** src/router/routing/base.py, src/router/routing/routers.py, src/router/agentic/router.py, src/router/models/artifacts.py
**Summary:** 4 findings: 2 correctness, 2 design/simplification · 3 confirmed, 1 plausible

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| RC-01 | `src/router/router.py:107-152` (docstring vs body) | correctness | medium | confirmed |
| RC-02 | `src/router/tools/router_tool.py:1,27` | correctness | low | confirmed |
| RC-03 | `src/router/router.py:104,155-165` | design | low | plausible |
| RC-04 | `src/router/router.py:65,101-104` | design | low | confirmed |

---

## src/router/router.py

### RC-01: `max_latency` / `latency_weight` are permanent no-ops -- `ModelScore.expected_latency` is never populated `:107-152`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
128:            scores.append(ModelScore(
129:                model=cand,
130:                score=value,
131:                expected_quality=value,
132:                expected_cost=cost_of.get(model_id, getattr(cand, "output_cost_per_token", 0.0)),
133:                confidence=confidence,
134:                supported=supported,
135:            ))
```

`Router.route()`'s own docstring claims: *"The hard limits in `request.constraints` (`min_quality` / `max_cost` / `max_latency`) are applied to the scored candidates before the policy ranks them."* But `ModelScore` is constructed above without an `expected_latency=` kwarg, so it silently takes the dataclass default `0.0` (`src/router/decision.py:27`). Unlike `expected_cost`, which has a two-tier fallback (`router_model.default_model_costs` -> `cand.output_cost_per_token`), there is **no** signal source for latency anywhere in this package -- `LLMProfile` (`src/router/llm/profile.py`) has no latency field either.

**What breaks:** A caller that sets `RoutingConstraints(max_latency=0.05)` expecting the hard filter in `_apply_hard_limits` (`router.py:140-152`, `s.expected_latency <= max_l`) to drop slow models gets no filtering at all -- every candidate has `expected_latency == 0.0 <= 0.05`, so all pass. Likewise `OptimizationObjective(latency_weight=...)` (`constraints.py:34`) always multiplies by `1.0 - 0.0 = ...` wait, multiplies latency term by a constant `0.0`, so the tradeoff has zero effect on ranking regardless of the weight chosen. This is a silent contract violation: the feature reads as implemented (types exist, docstring says it's applied) but never does anything.
**Evidence:** Grepped `expected_latency` repo-wide; only two writers exist: the dataclass default (`decision.py:27`) and `execution/forecast.py:81` (an unrelated, unwired path). `router.py`'s `route()` -- the only place `ModelScore` is built for the live routing path -- never sets it.
**Direction:** Either wire a real latency signal (router-level or profile-level, mirroring the cost fallback) before shipping `max_latency`/`latency_weight` as usable knobs, or make the gap explicit: drop/warn on a non-default `max_latency`/`latency_weight` so callers don't silently get no-op filtering, and correct the docstring to say latency is not yet enforced.

---

### RC-03: no-registry mode makes `required_capabilities` unsatisfiable, silently `:104,155-165`
**Category:** design · **Severity:** low · **Confidence:** plausible

```python
101:        if self.llm_registry is None:
102:            # no registry at all -> the router's bare ids are the candidates. An
103:            # *empty* registry is a real "no candidates" and must not fall back.
104:            context.candidates = [_StubProfile(m) for m in self.router_model.model_ids]
...
162:    def __init__(self, model_id: str):
163:        self.model_id = model_id
164:        self.capabilities: list[str] = []
```

`_StubProfile.capabilities` is always `[]`. `RouterModel.filter_candidates` drops any candidate that doesn't have every entry of `constraints.required_capabilities` in its `capabilities`. Combined, any caller that runs `Router` with no `llm_registry` and a non-empty `required_capabilities` gets `filtered == []` for every request, unconditionally -- not because no model in the pool actually has that capability, but purely because capability data doesn't exist in stub mode.
**What breaks:** `decide()` returns `reason="no candidates scored"` with no indication that the cause is "you didn't pass a registry," which is easy to mistake for "no model qualifies" during debugging/production triage.
**Evidence:** Traced `_StubProfile` construction (`router.py:155-165`) against `RouterModel.filter_candidates`'s capability check (`routing/base.py:220-221`, `required.issubset(set(getattr(c, "capabilities", []) or []))`).
**Direction:** Either raise/warn early when `llm_registry is None` and `constraints.required_capabilities` is non-empty (config that can never succeed), or note the limitation in `route()`'s docstring next to the existing no-registry paragraph.

---

### RC-04: candidate list is built twice, once discarded `:65,101-104`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
65:        candidates = self.llm_registry.list() if self.llm_registry is not None else []
66:        return RoutingContext(request=request, signals=signals, candidates=candidates)
...
101:        if self.llm_registry is None:
102:            # no registry at all -> the router's bare ids are the candidates. An
103:            # *empty* registry is a real "no candidates" and must not fall back.
104:            context.candidates = [_StubProfile(m) for m in self.router_model.model_ids]
```

`build_context()` always computes `candidates` (an empty list, when there's no registry) and hands it to `RoutingContext`; `route()` then immediately overwrites `context.candidates` with the stub-profile list in the no-registry branch. The first assignment is dead work every time `llm_registry is None`, and the mutate-after-construct pattern also means any other caller of `build_context()` alone (outside `route()`) gets a `RoutingContext` with `candidates=[]` even though `route()`'s comment says "the router's bare ids are the candidates" -- that fallback only happens inside `route()`, not in `build_context()` itself.
**What breaks:** Nothing crashes; it's wasted allocation plus a context-construction API (`build_context`) whose output contract (candidates reflect the pool) differs depending on whether it's called standalone or via `route()`.
**Evidence:** Read both methods in full; no other caller of `build_context()` currently exists in the repo (grepped), so this is latent rather than actively causing divergent behavior today.
**Direction:** Move the stub-profile fallback into `build_context()` itself (guarded by `if self.llm_registry is None`), so `RoutingContext.candidates` always reflects the real pool regardless of caller, and `route()` no longer needs the post-hoc overwrite.

---

## src/router/tools/router_tool.py (breakage caused by this diff, file itself out of the reviewed set)

### RC-02: stale `TYPE_CHECKING` import: `routing.base.Router` no longer exists (renamed to `RouterModel`) `:1,27`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
1:"""``RouterTool``: intended to expose a :class:`~router.routing.base.Router`
...
27:    from ..routing.base import Router
...
34:    def __init__(self, router: "Router"):
```

This diff renames `router.routing.base.Router` -> `RouterModel` (per `src/router/routing/base.py`, confirmed via `git diff`: the class is `RouterModel` on this branch). `router_tool.py` was not part of the renamed file set and still imports the old name under `if TYPE_CHECKING:`.
**What breaks:** At runtime this is harmless -- the module has `from __future__ import annotations`, so `"Router"` is never evaluated, and the `TYPE_CHECKING` block never executes. But `mypy`/`pyright` would fail with "cannot import name 'Router' from 'router.routing.base'" the moment this file is type-checked, and the Sphinx cross-reference `:class:`~router.routing.base.Router`` in the docstring now points at a symbol that doesn't exist. Grepped the whole repo: this is the only remaining live-code reference to the old `routing.base.Router` name (`src/router/router.py:5` is an intentional historical mention inside its own docstring, not a code reference).
**Evidence:** `grep -rn "routing\.base import Router\b|routing\.base\.Router\b" src/` -> only `router.py:5` (docstring, correct as historical context) and `router_tool.py:1,27` (stale).
**Direction:** Update `router_tool.py`'s import to `from ..routing.base import RouterModel` and its type hint / docstring reference to `RouterModel`, matching the rest of the package.
