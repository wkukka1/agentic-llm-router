# ROUTERMODEL-002: `RouterModel.cost_model` is stored but never read

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** medium (a wired-looking feature that does nothing)
- **Class:** `RouterModel` ([routing/base.py:187](../../../src/router/routing/base.py#L187))
- **Found:** 2026-09-19

## What

`cost_model` is assigned in three places and read nowhere:

- [base.py:187](../../../src/router/routing/base.py#L187) `self.cost_model = cost_model`
- [routers.py:261](../../../src/router/routing/routers.py#L261) `NIRTRouter.from_artifact`
- [routers.py:490](../../../src/router/routing/routers.py#L490) `MLPRouter.from_artifact`

`default_model_costs` for the data-backed routers always calls `_lazy_model_costs`, the
train-split mean from the observation table. It ignores the injected `CostModel`. So
`RouterModelFactory.create(artifact, cost_model)` accepts a cost model and has no effect
on routing.

Even after [ROUTERMODELFACTORY-001](../../RouterModelFactory/ROUTERMODELFACTORY-001-cost-model-kwarg-overwritten/bug.md)
is fixed, a caller passing `cost_model=` still sees no change.

## Failure scenario

A caller supplies a `CostModel` with current prices to the factory and expects
cost-aware routing to use it. Routing silently keeps using the train-mean USD/query.
