# MOD-002: `default_prompt_decomposer` cannot set a defer threshold, so `should_defer` is always False

- **Status:** confirmed (traced; not executed)
- **Severity:** low
- **Where:** `default_prompt_decomposer` ([decompose/decomposer.py:53-91](../../../src/decompose/decomposer.py#L53-L91)) → `DomainClassifier` / `TaskClassifier` ([prompt_heads.py:87-121](../../../src/decompose/classifiers/prompt_heads.py#L87-L121)) → `CalibratedHead` ([heads/base.py:31](../../../src/decompose/classifiers/prompt_decomposition/heads/base.py#L31))
- **Found:** 2026-09-24

## What

`CalibratedHead` defaults `defer_below=0.0`, and `should_defer` is
`confidence < defer_below`. With the default, `confidence < 0.0` is never true.
The factory takes only `artifact_paths` and `embedder`; the domain and task
classifiers are constructed as `DomainClassifier(path, merge_domains=True)` and
`TaskClassifier(path)`, with no route for a threshold to reach the head.

## Failure scenario

The `domain` and `task` signals carry `should_defer` in their metadata, and the
head docstrings tell the caller to "widen the candidate set, fall back, or ask"
when it is set. Through the factory (the only documented way to get a working
decomposer) it is `False` on every prompt, including a 0.30-confidence
prediction. A consumer that trusts the flag is silently never told the head is
unsure. Building the classifiers by hand with `defer_below=` works, so this is a
missing option on the factory rather than a broken head.

## Suggested fix

Give the factory a `defer_below: Mapping[str, float] | None` (keys `"domain"`,
`"task"`), validated the same way as `artifact_paths`, and pass it through. Note
in the signal docs that the flag is off unless a threshold is configured.
