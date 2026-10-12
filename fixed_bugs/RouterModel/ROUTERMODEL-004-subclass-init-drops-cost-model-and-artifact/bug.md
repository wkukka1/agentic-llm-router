# ROUTERMODEL-004: Subclass constructors don't accept/forward `cost_model` or `artifact`

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** medium
- **Class:** `RouterModel` and subclasses ([routing/base.py:176-190](../../../src/router/routing/base.py#L176-L190), `router/routing/routers.py`)
- **Found:** 2026-09-22

## What

`RouterModel.__init__` accepts `cost_model` and stores it
([routing/base.py:187](../../../src/router/routing/base.py#L187), the storage
side of [ROUTERMODEL-002](../ROUTERMODEL-002-cost-model-never-consumed/bug.md)). None
of the concrete subclasses in `routing/routers.py`
(`MatrixRouter`, `RandomRouter`, `NIRTRouter`, `KNNRouter`, `MLPRouter`)
expose a `cost_model=` or `artifact=` parameter in their own `__init__`, so a
caller constructing one directly (rather than through
`RouterModelFactory`/`from_artifact`) has no way to pass either through to the
base class.

## Failure scenario

Direct construction of any router subclass with a cost model
(`NIRTRouter(..., cost_model=my_cost_model)`) raises `TypeError` — the
keyword doesn't exist on the subclass signature — rather than reaching the
base class's (currently unused, see ROUTERMODEL-002) `self.cost_model`.
