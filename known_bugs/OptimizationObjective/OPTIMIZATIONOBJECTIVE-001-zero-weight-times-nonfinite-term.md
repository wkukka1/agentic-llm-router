# OPTIMIZATIONOBJECTIVE-001: a zero-weight term still poisons the utility when its input is non-finite

- **Status:** confirmed (arithmetic); **latent** with the shipped routers
- **Severity:** low
- **Class:** `OptimizationObjective` ([constraints.py:29-36](../../src/router/constraints.py#L29-L36))
- **Found:** 2026-09-19

## What

`utility()` computes `- self.cost_weight * score.expected_cost`. With `cost_weight=0.0`
and `expected_cost` of `nan` or `inf`, the product is `nan`, so the whole utility is `nan`.
The same holds for `latency_weight * expected_latency` and `risk_weight * (1 - confidence)`.
A term meant to be switched off (weight 0) still changes the result.

`DefaultRoutingPolicy.decide` maps a non-finite utility to `-inf`, so such a candidate is
ranked last even under a quality-only objective.

## Why it is only latent

The shipped routers can't produce a non-finite cost: `_mean_train_costs`
([routers.py:80](../../src/router/routing/routers.py#L80)) replaces non-finite entries with
the pool maximum, and `_resolve_costs` rejects non-finite explicit costs. It becomes real if
a router with a custom `default_model_costs`, or a profile with an `inf` price, feeds a
non-finite `expected_cost` into `Router.route`.

## Suggested fix

Skip a term when its weight is 0, or sanitize the inputs before multiplying.
