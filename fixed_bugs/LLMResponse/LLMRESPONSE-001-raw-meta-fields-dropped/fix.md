# LLMRESPONSE-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

Added `raw: object | None = None` and `meta: dict | None = None` back to `router.llm.client.LLMResponse`. No adapter populates them yet (all existing `LLMResponse(...)` call sites use keyword args, so the new trailing fields are additive and don't change any call site). Covered by `tests/router/llm/test_client.py::test_llm_response_carries_raw_and_meta`.

## Fix suggested when the bug was filed

Either add `raw: object | None = None` / `meta: dict | None = None` back to
`LLMResponse` (cheap, forward-compatible), or note the intentional drop in
the class docstring so it doesn't get "rediscovered" as a regression later.

## Regression tests

`tests/router/llm/test_client.py:46` `test_llm_response_carries_raw_and_meta`

```python
def test_llm_response_carries_raw_and_meta():
    """LLMRESPONSE-001: ``raw``/``meta`` fields dropped during the migration
    from ``agentic.llm_clients.LLMResponse`` must exist again."""
    resp = LLMResponse(content="ok", raw={"finish_reason": "stop"}, meta={"provider": "echo"})
    assert resp.raw == {"finish_reason": "stop"}
    assert resp.meta == {"provider": "echo"}
    assert LLMResponse(content="ok").raw is None
    assert LLMResponse(content="ok").meta is None
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/llm/client.py`

```diff
--- a/src/router/llm/client.py
+++ b/src/router/llm/client.py
@@ -31,6 +31,11 @@ class LLMResponse:
     output_tokens: Optional[int] = None
     cost: Optional[float] = None
     latency: Optional[float] = None
+    #: provider-specific raw response object / metadata (finish reason,
+    #: tool-call payloads, moderation flags, ...). Nothing reads these today;
+    #: kept so an adapter can attach them without another migration.
+    raw: Optional[object] = None
+    meta: Optional[dict] = None
 
 
 class LLMClient:
```
