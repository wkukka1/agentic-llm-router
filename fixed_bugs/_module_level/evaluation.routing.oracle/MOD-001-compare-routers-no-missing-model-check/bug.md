# MOD-001: `compare_routers` skips the missing-model guard its sibling `evaluate_router` has

- **Status:** fixed 2026-09-25, see [fix.md](fix.md)
- **Severity:** medium
- **Where:** `compare_routers` ([evaluation/routing/oracle.py:658-678](../../../../src/evaluation/routing/oracle.py#L658-L678)), cf. `evaluate_router` ([oracle.py:85-93](../../../../src/evaluation/routing/oracle.py#L85-L93))
- **Found:** 2026-09-22

## What

`evaluate_router` explicitly raises when `true_df`/`cost_df` lack a column
for one of the router's pool models, with the comment: *"reindexing to a
model the outcome matrix lacks would add an all-NaN column and silently turn
every regret / hit rate into NaN"* ([oracle.py:86-88](../../../../src/evaluation/routing/oracle.py#L86-L88)).

`compare_routers`, in the same file, does the equivalent reindex with no such
guard:

```python
mat = r.aligned_scores(query_ids).reindex(columns=model_ids)   # oracle.py:678
```

If a router's own `model_ids` don't cover all of `true_df.columns`
(`model_ids` here), this silently introduces an all-NaN column into that
router's prediction matrix, the same failure mode `evaluate_router` was
written to prevent.

## Failure scenario

`compare_routers([router_a, router_b], true_df, ...)` where one router's
pool is missing a model that's a column in `true_df` gets that router's
comparison row silently corrupted by NaN propagation (regret/hit-rate for
that router turn NaN) instead of raising the same clear error
`evaluate_router` gives for the identical situation.
