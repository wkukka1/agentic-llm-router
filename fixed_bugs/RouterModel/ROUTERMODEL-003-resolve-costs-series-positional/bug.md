# ROUTERMODEL-003: `_resolve_costs` treats a labelled `pd.Series` as positional

- **Status:** fixed 2026-09-20, see [fix.md](fix.md)
- **Severity:** low
- **Class:** `RouterModel` helper `_resolve_costs` ([routing/base.py:137-143](../../../src/router/routing/base.py#L137-L143))
- **Found:** 2026-09-19

## What

`_resolve_costs` aligns by label only for a `Mapping`. A `pd.Series` is not a `Mapping`,
so it goes down the array branch (`np.asarray(costs).ravel()`), which ignores the index.
A Series indexed by model id in a different order is silently mis-assigned.

## Failure scenario

Repro (`known_bugs/_repros/repro_series.py`), pool order `["big", "small"]`:

```
dict  : [0.01  0.001]   # correct
series: [0.001 0.01 ]   # swapped: big now costs 0.001
```

`route(model_costs=some_series)` and `MatrixRouter(..., model_costs=some_series)` would pick
by wrong prices, with no error. The length check passes because the Series has one entry
per model.
