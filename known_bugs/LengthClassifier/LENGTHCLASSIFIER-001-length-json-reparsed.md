# LENGTHCLASSIFIER-001: `length.json` is parsed twice, once by each of two layers

- **Status:** confirmed (traced; code_review finding DC-05)
- **Severity:** low (duplication/inefficiency, not wrong output — both parses read the same file)
- **Class:** `LengthClassifier` ([decompose/classifiers/prompt_heads.py:142-144](../../src/decompose/classifiers/prompt_heads.py#L142-L144))
- **Found:** 2026-09-22

## What

`LengthClassifier.__init__` constructs a `LengthHead(run_dir)` (which already
parses `length.json` as part of building the head), then separately
re-reads and re-parses the same `length.json` itself.

## Failure scenario

Not a correctness bug today — both parses read the same file and (presumably)
agree. The risk is drift: if the two parsing paths are ever edited
independently, they can silently disagree about `length.json`'s schema with
no indication one of them is stale.

## Suggested fix

Have `LengthClassifier` read whatever `LengthHead` already parsed (expose it
as an attribute) instead of re-opening and re-parsing `length.json` a second
time.
