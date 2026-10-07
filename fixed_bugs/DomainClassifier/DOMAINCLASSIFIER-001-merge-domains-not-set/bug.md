# DOMAINCLASSIFIER-001: Built without `merge_domains=True`, defaults to the unmerged 10-class head

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** medium
- **Class:** `decompose.classifiers.prompt_heads.DomainClassifier` ([decompose/decomposer.py:84](../../../src/decompose/decomposer.py#L84))

  (Not to be confused with the unrelated `decompose.classifiers.prompt_decomposition.models.DomainClassifier` ABC — same
  name, different module, different concept.)
- **Found:** 2026-09-22

## What

`default_prompt_decomposer` builds its domain head as
`DomainClassifier(paths["domain"])`, with no `merge_domains=True`. Per
`docs/prompt_decomposition.md`'s intent, this head is meant to run on the
merged domain taxonomy; left at its default, it runs the unmerged, finer
10-class split instead.

## Failure scenario

Every prompt routed through `default_prompt_decomposer` gets domain signals
in the unmerged taxonomy, not the merged one downstream consumers (and the
documented intent) expect — a systematic mismatch, not an edge case, for
every call that reaches this path.
