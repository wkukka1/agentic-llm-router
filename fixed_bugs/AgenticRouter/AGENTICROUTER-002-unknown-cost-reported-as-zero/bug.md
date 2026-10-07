# AGENTICROUTER-002: a call with unknown cost is reported as cost `0.0`

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** medium (misleading numbers, no error)
- **Class:** `AgenticRouter.run` ([router.py:139](../../../src/router/agentic/router.py#L139), [147](../../../src/router/agentic/router.py#L147)) and `AgenticResult.cost` ([result.py:48](../../../src/router/agentic/result.py#L48))
- **Found:** 2026-09-19

## What

`SubCall.cost` is `Optional[float]`, where `None` means unknown (the `LLMResponse` docstring
is explicit that unknown is "distinct from a genuinely free (`0`) call"). The result layer
throws that distinction away: `cost=root.cost or 0.0`, `AgenticResult.cost: float = 0.0`, and
the orchestrated total sums `(n.cost or 0.0)` over the leaves.

Any client built without prices reports `None`: `ClientRegistry.langchain(...)` clients have
no profile, and `LangChainAdapter` only prices from explicit per-token kwargs.

## Failure scenario

Repro (`known_bugs/_repros/repro_serving_cost.py`, section `AGENTICRESULT-001`):

```
SubCall.cost = None | AgenticResult.cost = 0.0
```

An orchestrated run with real LangChain models and no configured prices reports `cost=0.0`.
If only some leaves have a price, the total silently undercounts by the unpriced ones.
