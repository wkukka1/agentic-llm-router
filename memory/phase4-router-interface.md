---
name: phase4-router-interface
description: Phase 4 routing orchestrator kicked off with the modular Router interface (router.routing)
metadata:
  type: project
---

2026-09-10: started Phase 4 (the routing orchestrator) by adding the modular
`router.routing` package — an ABC `Router` with one abstract method
`predict_scores(query_ids) -> [query_id x model_id]`, plus shared
`route()` / `evaluate()` that delegate to [[routing-eval]]'s `routing_decision` /
`routing_evaluation`. Concrete: `MatrixRouter`, `NIRTRouter.from_run`,
`KNNRouter`, `MLPRouter`, `RandomRouter`; `@register` + `build_router(kind, ...)`
registry; `compare_routers()` helper. Docs in `docs/routing_interface.md`,
tests in `tests/test_routing_interface.py`.

**Why:** the codebase had NIRT/kNN/MLP predictors as loose functions
(`nirt.baselines`, `route_compare._nirt_matrix`); a single interface makes new
strategies drop-in.

**How to apply:** new routing strategies subclass `Router` + `@register`; don't
add another bespoke `*_matrix` function. `scripts/route_compare.py` is NOT yet
refactored onto this — still uses its own `preds` dict.
