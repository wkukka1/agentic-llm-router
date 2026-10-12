# MOD-002: `cost_savings_vs_ref` divides by the reference policy's cost with no zero/sign guard

- **Status:** confirmed (traced; code_review finding EN-04)
- **Severity:** low-medium
- **Where:** `routing_report` ([evaluation/nirt/routing.py:253-258](../../../src/evaluation/nirt/routing.py#L253-L258))
- **Found:** 2026-09-22

## What

```python
ref_row = df[df["policy"] == f"fixed: {reference}"]
...
if not ref_row.empty:
    rc, ra = float(ref_row["cost"].iloc[0]), float(ref_row["accuracy"].iloc[0])
    df["cost_savings_vs_ref"] = 1.0 - df["cost"] / rc
```

The only guard is `ref_row.empty` (does a reference row exist at all). There
is no check that `rc != 0` before dividing. If the reference policy's mean
cost is `0.0` (e.g. a reference model with no cost data, or a cost column
that's legitimately zero for a free/local model), `df["cost"] / rc` produces
`inf`/`-inf`/`nan` depending on sign, propagated silently into the report
dataframe's `cost_savings_vs_ref` column.

## Failure scenario

A `routing_report` run with `reference` pointing at a zero-cost model
produces `inf`/`nan` savings figures for every other policy in the same
table, with nothing flagging that the denominator was degenerate — easy to
miss in a wide comparison table.

## Suggested fix

Guard with `if not ref_row.empty and rc > 0:` (or explicitly set
`cost_savings_vs_ref` to `nan` with a comment when `rc <= 0`, so the
degenerate case is visibly handled rather than silently propagated).
