# NAIVEDECOMPOSER-001: sub-tasks are dropped without a signal

- **Status:** fixed, see [fix.md](fix.md)
- **Severity:** medium (part of the user's request is never answered)
- **Class:** `NaiveDecomposer.__call__` ([decompose.py:60](../../../src/router/agentic/decompose.py#L60), [67](../../../src/router/agentic/decompose.py#L67))
- **Found:** 2026-09-19

## What

Two paths discard sub-tasks with no warning:

1. **Short parts.** `len(p.strip()) >= min_part_chars` (default 8) is a filter, not just a
   noise guard: a genuine short instruction is removed.
2. **More than `max_parts`.** `out[: self.max_parts]` (default 6) truncates. The `LLMDecomposer`
   docstring calls exactly this "a cut-off request silently loses its later sub-tasks", but
   `NaiveDecomposer`, the default, does it.

## Failure scenario

Repro (`known_bugs/_repros/repro_agentic.py`, section `NAIVEDECOMPOSER-001`):

```
"Fix bug. Add tests for the module. Update the docs."
  -> ['Add tests for the module.', 'Update the docs.']        # "Fix bug" is gone
8 numbered items -> 6 sub-tasks                                # items 7 and 8 are gone
```

The orchestrator answers the remaining parts and returns a confident combined answer that
omits part of the request. When *all but one* part is dropped the `len(out) > 1` check falls
back to the whole prompt, which is why the single-task case looks fine.
