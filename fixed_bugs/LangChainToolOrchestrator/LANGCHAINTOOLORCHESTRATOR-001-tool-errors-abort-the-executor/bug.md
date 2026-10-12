# LANGCHAINTOOLORCHESTRATOR-001: `handle_tool_error=True` does not turn these tool failures into observations

- **Status:** fixed 2026-09-25, see [fix.md](fix.md)
- **Severity:** medium
- **Class:** `build_router_tools` / `LangChainToolOrchestrator` ([orchestrator.py:49-92](../../../src/router/agentic/orchestrator.py#L49-L92))
- **Found:** 2026-09-19

## What

The docstring says "Tool exceptions are returned to the agent as observations
(`handle_tool_error`) instead of aborting the executor". In `langchain_core`,
`handle_tool_error=True` only catches `ToolException`. Any other exception is re-raised.

The tools' failure modes are not `ToolException`s:

- `route_query` calls `agent.route_decision`, which raises `LookupError` when the router has no
  usable score, and `TypeError` when the router can't score raw text.
- `route_and_answer` calls `agent._solve`, which reaches the same `route_decision`.

(`answer_with_model` is fine: it returns an error string for a bad id, and `_call_model`
turns model failures at `depth >= 1` into failed `SubCall`s.)

## Failure scenario

Repro (`known_bugs/_repros/repro_agentic.py`, last section), agent whose `route_decision`
raises `LookupError`:

```
route_query -> RAISED LookupError no usable score for this prompt
answer_with_model -> "ERROR: 'zzz' is not a candidate model. Choose one of: a"
```

Inside `AgentExecutor` the `LookupError` propagates out of `executor.invoke`, through
`LangChainToolOrchestrator.run`, and out of `AgenticRouter.run`, discarding the sub-answers
already paid for. This is the opposite of the failure handling the `AgenticRouter` module
docstring promises for orchestrations.
