# ROUTER-003: `required_capabilities` silently empties routing in no-registry mode

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** medium
- **Class:** `Router` ([router.py:197-207](../../../src/router/router.py#L197-L207))
- **Found:** 2026-09-22

## What

When `Router` is built with `llm_registry=None`, `route()` substitutes
`_StubProfile` candidates for the pool's bare model ids
([router.py:132](../../../src/router/router.py#L132)). `_StubProfile.capabilities`
is hardcoded to `[]` ([router.py:206](../../../src/router/router.py#L206)).

`RouterModel.filter_candidates` drops any candidate whose capabilities don't
cover `constraints.required_capabilities`
([routing/base.py:222-223](../../../src/router/routing/base.py#L222-L223)). Since
every stub candidate has `capabilities == []`, any non-empty
`required_capabilities` makes `required.issubset(set([]))` false for all of
them — `filtered` comes back empty.

## Failure scenario

`Router(router_model=..., llm_registry=None).route(request)` with
`request.constraints.required_capabilities = ["vision"]` (or anything) silently
filters every candidate before scoring. `scores` ends up `[]` and
`routing_policy.decide` runs on no candidates, with no warning that the
constraint was unsatisfiable rather than genuinely unmet.
