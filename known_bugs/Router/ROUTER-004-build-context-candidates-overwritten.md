# ROUTER-004: `build_context`'s `candidates` is thrown away in no-registry mode

- **Status:** confirmed (traced; code_review finding RC-04)
- **Severity:** low (cosmetic/dead work, not wrong output)
- **Class:** `Router` ([router.py:62-67](../../src/router/router.py#L62-L67), [router.py:128-132](../../src/router/router.py#L128-L132))
- **Found:** 2026-09-22

## What

`build_context()` always computes `candidates` from the registry
(`self.llm_registry.list() if self.llm_registry is not None else []`,
[router.py:66](../../src/router/router.py#L66)). `route()` then
unconditionally overwrites `context.candidates` with a fresh
`_StubProfile` list whenever `self.llm_registry is None`
([router.py:132](../../src/router/router.py#L132)) — in that branch
`build_context` had already produced `[]`, so the work is a no-op, but the
split makes `build_context`'s contract ("returns the candidates that will be
scored") misleading: callers who use `build_context()` directly get a
different (empty) candidate set than what `route()` actually scores against.

## Failure scenario

No wrong routing result — this is a correctness-adjacent readability/API
wart. A caller inspecting `router.build_context(request).candidates` in
no-registry mode sees `[]` and could reasonably conclude routing has no
candidates, when `route()` on the same request scores against the full
model-id pool.

## Suggested fix

Move the `_StubProfile` fallback into `build_context()` itself so the method
returns the actual candidate set that will be scored, and drop the
in-`route()` override.
