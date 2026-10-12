# NAIVEDECOMPOSER-002: the `then,` split marker only matches when no space follows

- **Status:** reproduced
- **Severity:** low
- **Class:** `NaiveDecomposer` / `_SPLIT_RE` ([decompose.py:41](../../src/router/agentic/decompose.py#L41))
- **Found:** 2026-09-19

## What

The alternation contains `\bthen,\b`. `\b` after a comma needs the next character to be a
word character, but "then, " is followed by a space, so the marker cannot match ordinary
prose. Every other marker (`\band then\b`, `\bafter that\b`, ...) ends in a letter, where
`\b` behaves as intended.

## Failure scenario

Repro (`known_bugs/_repros/repro_agentic.py`, section `NAIVEDECOMPOSER-002`):

```
"Write the parser then, write the tests"   -> ['Write the parser then, write the tests']   # not split
"Write the parser then,write the tests"    -> ['Write the parser.', 'write the tests.']    # split
```

`HeuristicTriage._MULTIPART_MARKERS` does not include `then,` at all, so a prompt joined only by
"then," is not flagged multi-part either.

## Suggested fix

Use `\bthen,` (no trailing `\b`).
