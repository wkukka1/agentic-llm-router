# CLIENTREGISTRY-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

Regression test `tests/router/test_review_fixes.py::test_register_then_unknown_id_still_warns`

`register()` now also sets `self._has_real_clients = True` -- a client added
directly is exactly the "real client" case the warning exists to protect,
whether it arrived through the constructor or `register()`.

## Fix suggested when the bug was filed

Compute `_has_real_clients` on demand (`bool(self._clients) and any(non-synthesized)`), or set
it in `register()`.

## Regression tests

`tests/router/test_review_fixes.py:417` `test_register_then_unknown_id_still_warns`

```python
def test_register_then_unknown_id_still_warns():
    """CLIENTREGISTRY-001: the echo-fallback warning must fire whether real
    clients arrived via the constructor or via register()."""
    reg = ClientRegistry()
    reg.register("real-model", LLMClient(CallableAdapter("real-model", lambda p: p)))
    with pytest.warns(UserWarning, match="echo"):
        reg.get("unregistered-model")
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/agentic/llm_clients.py`

```diff
--- a/src/router/agentic/llm_clients.py
+++ b/src/router/agentic/llm_clients.py
@@ -91,6 +91,9 @@ class ClientRegistry:
 
     def register(self, model_id: str, client: LLMClient) -> None:
         self._clients[str(model_id)] = client
+        # a client added directly is exactly the "real client" case the
+        # echo-fallback warning below exists to protect (CLIENTREGISTRY-001)
+        self._has_real_clients = True
 
     def get(self, model_id: str) -> LLMClient:
         mid = str(model_id)
```
