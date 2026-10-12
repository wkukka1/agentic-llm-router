# ROUTER-001: unsupported candidates are costed on a different scale than supported ones

- **Status:** fixed 2026-09-20, see [fix.md](fix.md)
- **Severity:** medium
- **Class:** `Router` ([router.py:157](../../../src/router/router.py#L157))
- **Found:** 2026-09-19

## What

`Router.route` sets each candidate's `expected_cost` with a per-model fallback:

```python
expected_cost=cost_of.get(model_id, getattr(cand, "output_cost_per_token", 0.0))
```

`cost_of` holds the router's **USD per query** vector (`default_model_costs`). The
fallback is the profile's **USD per token**, which is about three orders of magnitude
smaller. The docstring says the profile fallback applies "only when the router carries no
cost signal", but the code applies it per model. A candidate missing from `cost_of` is
therefore priced on the wrong scale even when the router does have a cost signal.

The mix happens with `UnsupportedCandidatePolicy.SCORE_WITH_PRIOR`. A kept candidate that
is not in the router's pool has no entry in `cost_of`.

## Failure scenario

Pool `big` (0.010 USD/query) and `small` (0.001 USD/query), plus an unseen registry model
`unseen` (6e-5 USD/token). Under `SCORE_WITH_PRIOR` with `cost_weight=10`:

| model  | supported | expected_cost | expected_quality |
|--------|-----------|---------------|------------------|
| unseen | False     | 6e-05         | 0.750 (pool mean, confidence 0) |
| big    | True      | 0.01          | 0.800 |
| small  | True      | 0.001         | 0.700 |

`unseen` wins: a model the router has no evidence for is chosen because it looks about
1000x cheaper than everything else. The same wrong number is compared against
`constraints.max_cost` in `_apply_hard_limits`, so the hard cost cap passes it too.

Repro: `known_bugs/_repros/repro_router.py`, section `ROUTER-001`.
