# LLMDECOMPOSER-001: list-prefix stripping eats the start of a task that begins with a decimal number

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** low-medium (silently changes the question sent to a model)
- **Class:** `LLMDecomposer._parse` ([decompose.py:48](../../../src/router/agentic/decompose.py#L48), [105-113](../../../src/router/agentic/decompose.py#L105-L113))
- **Found:** 2026-09-19

## What

`_LINE_PREFIX_RE = ^\s*(?:[-*•]+|\d+[.)]|#+)\s*` removes list markers the model was told not
to add. The marker alternative `\d+[.)]` is followed by `\s*` (zero or more), so it also
matches the leading digits and dot of a decimal number: `3.14 times 2` loses `3.`.

## Failure scenario

Repro (`known_bugs/_repros/repro_agentic.py`, section `LLMDECOMPOSER-001`):

```
_parse("3.14 times 2 equals what?\n2. summarise the result")
  -> ['14 times 2 equals what?', 'summarise the result']
```

A sub-task about `3.14`, `2.5 GHz`, or `1.` a version string is sent to the routed model with
its first digits removed, and the answer is wrong for a reason nobody can see in the output.
