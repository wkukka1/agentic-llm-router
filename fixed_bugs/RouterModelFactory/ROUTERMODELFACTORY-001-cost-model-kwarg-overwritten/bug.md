# ROUTERMODELFACTORY-001: `create(artifact, cost_model=...)` discards the keyword

- **Status:** fixed 2026-09-20, see [fix.md](fix.md)
- **Severity:** medium
- **Class:** `RouterModelFactory` ([registry.py:50](../../../src/router/models/registry.py#L50))
- **Found:** 2026-09-19

## What

In the artifact branch of `create`:

```python
artifact = artifact_or_id
cost_model = store_or_cost_model      # overwrites the keyword argument
```

When called as `create(artifact, cost_model=cm)`, `store_or_cost_model` is `None`, so the
line replaces the caller's `cost_model` with `None`. Only the positional form
`create(artifact, cm)` works.

## Failure scenario

Repro (`known_bugs/_repros/repro_router.py`, section `REGISTRY-001`):

```
positional: CM | keyword: None
```

`from_artifact` receives `cost_model=None`.

## Note

Fixing this changes nothing observable yet, because no router reads `cost_model`; see
[ROUTERMODEL-002](../../RouterModel/ROUTERMODEL-002-cost-model-never-consumed/bug.md).
