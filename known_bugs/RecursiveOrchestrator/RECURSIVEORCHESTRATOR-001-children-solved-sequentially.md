# RECURSIVEORCHESTRATOR-001: Sub-tasks are solved sequentially, not concurrently

- **Status:** confirmed (traced; code_review finding RA-03)
- **Severity:** low (performance, not correctness)
- **Class:** `RecursiveOrchestrator` ([router/agentic/orchestrator.py:40](../../src/router/agentic/orchestrator.py#L40))
- **Found:** 2026-09-22

## What

```python
children = [agent._solve(s, depth + 1) for s in subs]
```

Each decomposed sub-task is solved one at a time in a plain list
comprehension. Sub-tasks are independent (that's the premise of
decomposition), so this leaves latency on the table proportional to
`len(subs)` — an `N`-way decomposition takes roughly `N` times a single
sub-call's latency instead of running them concurrently.

## Failure scenario

Not a wrong-answer bug. A prompt that decomposes into several independent
sub-tasks (the common case this orchestrator exists for) takes materially
longer wall-clock than necessary, and that cost scales with decomposition
width — worth knowing before treating `RecursiveOrchestrator` as
latency-competitive with a single direct call.

## Suggested fix

Run `agent._solve` for each sub-task concurrently (thread pool for
sync-blocking LLM calls, or `asyncio.gather` if the client layer grows an
async path), respecting whatever call/cost budget the parent `AgenticRouter`
run is tracking.
