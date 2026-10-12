# PROMPTDECOMPOSER-002: A duplicate classifier is accepted, then fails every request

- **Status:** reproduced (`known_bugs/_repros/repro_decompose.py`, section 4)
- **Severity:** low
- **Class:** `PromptDecomposer` ([decompose/decomposer.py:37-50](../../src/decompose/decomposer.py#L37-L50)), with `PromptSignals.add` ([signals.py:26-29](../../src/decompose/signals.py#L26-L29))
- **Found:** 2026-09-24

## What

`add_classifier` appends without checking whether a classifier with that name,
or one that emits the same signal names, is already registered. The collision is
only detected when `extract_signals` calls `PromptSignals.add`, which raises
`ValueError("Signal <name> already exists")`.

`remove_classifier(name)` filters by `name` and so removes *every* classifier
sharing it, with no return value or error to say how many went.

## Failure scenario

```python
pd = PromptDecomposer([SurfaceClassifier()])
pd.add_classifier(SurfaceClassifier())      # accepted
pd.extract_signals(ClassificationInput(prompt="hi"))
# ValueError: Signal surface already exists
```

A caller who layers `add_classifier(SurfaceClassifier())` onto
`default_prompt_decomposer()` (which already includes it) gets a decomposer that
constructs fine and then raises on the first `Router.route()` call, not where the
mistake was made. Two heads emitting the same signal name (for example a second
`DomainClassifier` for A/B comparison) fail the same way.

## Suggested fix

Reject a repeated classifier `name` in `add_classifier` (and in `__init__`) with a
clear error at registration, and make `remove_classifier` raise or return a count
when nothing matches. Signal-name overlap between differently-named classifiers
can only be known at classify time, so keep the `PromptSignals.add` check as the
backstop but name the offending classifier in its message.
