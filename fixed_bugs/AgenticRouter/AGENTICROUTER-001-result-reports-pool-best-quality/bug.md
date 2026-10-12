# AGENTICROUTER-001: `AgenticResult.predicted_quality` is the pool's best score, shown next to a cheaper model

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** medium (revised 2026-09-23, was low) — fires whenever `lam > 0` actually
  routes to a non-best model, which is the intended, common path for a cost-aware router, not
  an edge case; the error is always in the optimistic direction (`predicted_quality` inflated
  toward the pool-best), so it can silently mask real quality loss in any downstream
  quality/cost reporting built on `AgenticResult.predicted_quality`
- **Class:** `AgenticRouter.run` ([router.py:134-140](../../../src/router/agentic/router.py#L134-L140))
- **Found:** 2026-09-19

## What

`AgenticRouter` routes with `lam` (a cost weight), so the model it selects can be a cheaper,
weaker one than the pool's best. `run()` builds the single-mode result as
`selected_model_id=root.model_id` (the model that actually answered) but
`predicted_quality=decision.predicted_quality`, which the triage docstring defines as the
**pool-best** score, "independent of the router's cost weight".

The leaf `SubCall` gets the right number (`_quality_of(decision)`, the answering model's own
score), so the two records of the same call disagree.

## Failure scenario

Repro (`known_bugs/_repros/repro_agentic.py`, section `AGENTICROUTER-001`): pool `big`
(q=0.90, cost 1.0) and `cheap` (q=0.80, cost 0.0), `lam=0.5`:

```
selected_model_id=cheap  result.predicted_quality=0.9  steps[0].predicted_quality=0.8
summary: single -> cheap (q~0.90): ...
```

The summary claims `cheap` is expected to score 0.90. With `lam=0` the numbers coincide,
which is why tests pass.
