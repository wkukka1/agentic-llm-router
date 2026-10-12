# MOD-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-25, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`guess_provider` now requires a right-hand boundary, allowing an optional trailing version number (`llama31_8b_instruct`, `qwen2.5-7b-instruct`, `gemini15_flash` still resolve; `gpt4all-13b-snoozy` no longer matches). Tests: `test_guess_provider_requires_a_right_hand_boundary_too`, `test_guess_provider_allows_a_trailing_version_number_after_the_hint`.

## Fix suggested when the bug was filed

Add the same right-hand boundary the docstring already claims to have:
`rf"(?:^|[^a-z0-9]){re.escape(needle)}(?:$|[^a-z0-9])"`.

## Regression tests

`tests/router/llm/adapters/test_langchain_adapter.py:59` `test_guess_provider_requires_a_right_hand_boundary_too`

```python
def test_guess_provider_requires_a_right_hand_boundary_too():
    """MOD-001 (router.llm.adapters): a hint matching mid-token (inside a
    longer alnum run) must not count as a match -- only RA-19's existing
    left-boundary guard existed before this fix."""
    assert guess_provider("gpt4all-13b-snoozy") is None
    # existing passing cases must still match -- a boundary on the right,
    # not "no match at all"
    assert guess_provider("o1-mini") == "openai"
    assert guess_provider("gpt-4o") == "openai"
    assert guess_provider("yolo1") is None
    assert guess_provider("claude-3-5-sonnet") == "anthropic"
    assert guess_provider("some-random-model") is None
```

`tests/router/llm/adapters/test_langchain_adapter.py:73` `test_guess_provider_allows_a_trailing_version_number_after_the_hint`

```python
def test_guess_provider_allows_a_trailing_version_number_after_the_hint():
    """A hint may be followed by a version number (llama3, qwen2.5, gemini15)
    -- real configured ids -- but not by more letters (gpt4all)."""
    assert guess_provider("gpt4all-13b-snoozy") is None
    assert guess_provider("llama31_8b_instruct") == "groq"
    assert guess_provider("llama2-70b-steerlm-chat") == "groq"
    assert guess_provider("qwen1.5-7b-chat") == "together"
    assert guess_provider("qwen2.5-0.5b-instruct") == "together"
    assert guess_provider("qwen2.5-7b-instruct") == "together"
    assert guess_provider("qwen25_7b_instruct") == "together"
    assert guess_provider("qwen3-235b-a22b-2507") == "together"
    assert guess_provider("gemini15_flash") == "google_genai"
    # previously pinned cases still hold
    assert guess_provider("o1-mini") == "openai"
    assert guess_provider("gpt-4o") == "openai"
    assert guess_provider("yolo1") is None
    assert guess_provider("claude-3-5-sonnet") == "anthropic"
    assert guess_provider("some-random-model") is None
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/llm/adapters/langchain_adapter.py`

```diff
--- a/src/router/llm/adapters/langchain_adapter.py
+++ b/src/router/llm/adapters/langchain_adapter.py
@@ -34,12 +34,15 @@ _PROVIDER_HINTS = [
 
 def guess_provider(model_id: str) -> Optional[str]:
     """Best-effort LangChain ``model_provider`` from a bare model id -- a last
-    resort. Each hint must start a token (``o1-mini`` matches ``o1``,
-    ``yolo1`` does not). Prefer an explicit ``provider=``/``model=`` per pool
-    id: a substring can't tell which host actually serves a given model."""
+    resort. Each hint must start its own token and end on a boundary; a hint
+    may be followed by a version number such as ``llama3`` / ``qwen2.5`` /
+    ``gemini15``, but not by more letters (``o1-mini`` matches ``o1``,
+    ``yolo1`` does not, ``gpt4all-13b-snoozy`` does not).
+    Prefer an explicit ``provider=``/``model=`` per pool id: a substring
+    can't tell which host actually serves a given model."""
     low = str(model_id).lower()
     for needle, provider in _PROVIDER_HINTS:
-        if re.search(rf"(?:^|[^a-z0-9]){re.escape(needle)}", low):
+        if re.search(rf"(?:^|[^a-z0-9]){re.escape(needle)}(?:\d+(?:\.\d+)*)?(?:$|[^a-z0-9])", low):
             return provider
     return None
 
```
