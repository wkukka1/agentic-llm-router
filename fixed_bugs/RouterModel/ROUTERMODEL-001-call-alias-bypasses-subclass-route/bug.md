# ROUTERMODEL-001: `__call__ = route` freezes the base implementation

- **Status:** fixed 2026-09-20, see [fix.md](fix.md)
- **Severity:** low
- **Class:** `RouterModel` ([routing/base.py:344](../../../src/router/routing/base.py#L344))
- **Found:** 2026-09-19

## What

`__call__ = route` binds the *function object* `RouterModel.route` at class-creation time.
A subclass that overrides `route()` does not change what `router(query_ids)` runs: the call
still goes to the base method.

## Failure scenario

Repro (`known_bugs/_repros/repro_router.py`, section `ROUTERMODEL-001`): a subclass whose `route()`
raises. `m.route([...])` raises, but `m([...])` returns a normal `RoutingResult`.

Nothing in `src/` overrides `route()` now, which is why this is latent. A future
bandit or cascade router that customizes `route()` will be bypassed by every `router(...)`
call site.
