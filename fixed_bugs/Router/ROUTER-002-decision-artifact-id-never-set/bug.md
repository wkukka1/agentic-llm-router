# ROUTER-002: `RoutingDecision.artifact_id` is never populated

- **Status:** fixed 2026-09-20, see [fix.md](fix.md)
- **Severity:** low
- **Class:** `Router` ([router.py:162](../../../src/router/router.py#L162))
- **Found:** 2026-09-19

## What

`RoutingDecision` has an `artifact_id` field ([decision.py:45](../../../src/router/decision.py#L45)),
meant to record which fitted artifact produced the ranking. `Router.route` returns
`routing_policy.decide(context, scores)` directly. `DefaultRoutingPolicy.decide` builds
the decision without setting it, and neither `Router` nor the policy reads
`router_model.artifact`. So `artifact_id` is always `None`, even when the router was built
from a `RouterModelArtifact` (`NIRTRouter.from_artifact` / `MLPRouter.from_artifact` set
`router.artifact`).

## Failure scenario

A decision logged for audit or provenance cannot be traced back to the artifact that
produced it. The field is there for exactly that.
