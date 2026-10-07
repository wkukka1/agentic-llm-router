# AGENTICROUTER-004: per-run state on the instance makes concurrent `run()` calls interfere

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** low (matters as soon as one `AgenticRouter` serves more than one request at a time)
- **Class:** `AgenticRouter` ([router.py:82-83](../../../src/router/agentic/router.py#L82-L83), [123-128](../../../src/router/agentic/router.py#L123-L128))
- **Found:** 2026-09-19

## What

`run()` resets `self._decisions` and `self._calls` at the start and clears `_decisions` in a
`finally`. Both are attributes of the shared instance, so two overlapping `run()` calls (two
threads or two asyncio tasks) share them:

- one run's `finally` wipes the other's cached triage decisions, so `_answer_directly` falls
  back to re-routing the prompt (extra work, and a possibly different model than triage saw);
- `_calls` is reset by whichever run starts second, so `max_calls` counts a mixture.

`with_langchain_orchestrator` already builds a fresh copy via `_shallow_copy()` for this
reason, which shows the state is known to be per-run.
