# MOD-001: `default_prompt_decomposer` is built but `Router` never calls it

- **Status:** confirmed (traced; code_review finding DC-04)
- **Severity:** low (integration gap, not a wrong-output bug)
- **Where:** `default_prompt_decomposer` (`src/decompose/decomposer.py`), `Router.__init__` ([router/router.py:57](../../../src/router/router.py#L57))
- **Found:** 2026-09-22

## What

`Router.__init__` constructs its decomposer as
`self.prompt_decomposer = prompt_decomposer or PromptDecomposer()` — a bare,
classifier-less `PromptDecomposer`, not `default_prompt_decomposer()` (the
factory that wires up `SurfaceClassifier` + the four trained head
classifiers from `artifact_paths`). A repo-wide grep shows
`default_prompt_decomposer` is referenced only from its own module, the
`prompt_heads.py` docstring, and its own test file — never from `Router` or
any other production call site.

## Failure scenario

`Router()` constructed without an explicit `prompt_decomposer=` gets a
decomposer that runs zero classifiers — `RoutingContext.signals` is always
empty on the default path, silently. Anyone relying on prompt-decomposition
signals reaching `Router.route()` out of the box gets nothing, with no error.

## Suggested fix

Either default `Router.prompt_decomposer` to `default_prompt_decomposer()`
(needs `artifact_paths`/config plumbed to `Router.__init__`), or, if that
integration isn't ready yet, say so explicitly in `Router`'s docstring so the
gap is a documented state rather than a silent one.
