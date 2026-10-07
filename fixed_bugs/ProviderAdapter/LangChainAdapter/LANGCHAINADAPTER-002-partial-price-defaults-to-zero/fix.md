# LANGCHAINADAPTER-002 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-25, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`LangChainAdapter._cost` returns `None` when a side with a nonzero token count has no configured price (same rule as `LLMClient.price`). Tests: `test_complete_prices_nothing_when_only_one_side_is_configured` / `test_complete_prices_normally_when_both_sides_are_configured`.

## Fix suggested when the bug was filed

Treat a profile with exactly one of the two per-token prices set as a
configuration error (raise, or warn loudly) instead of silently defaulting
the missing side to `0.0`.

## Regression tests

`tests/router/llm/adapters/test_langchain_adapter.py:144` `test_complete_prices_nothing_when_only_one_side_is_configured`

```python
def test_complete_prices_nothing_when_only_one_side_is_configured():
    """LANGCHAINADAPTER-002: a half-priced adapter (only input_cost_per_token
    given) must not silently price the unset output side at $0."""
    chat = _PricedChatModel()
    adapter = LangChainAdapter("gpt-4", chat_model=chat, input_cost_per_token=1e-6)

    resp = adapter.complete("hi")

    assert resp.cost is None
```

`tests/router/llm/adapters/test_langchain_adapter.py:155` `test_complete_prices_normally_when_both_sides_are_configured`

```python
def test_complete_prices_normally_when_both_sides_are_configured():
    chat = _PricedChatModel()
    adapter = LangChainAdapter(
        "gpt-4", chat_model=chat, input_cost_per_token=1e-6, output_cost_per_token=2e-6,
    )

    resp = adapter.complete("hi")

    assert resp.cost == pytest.approx(100 * 1e-6 + 50 * 2e-6)
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/llm/adapters/langchain_adapter.py`

```diff
--- a/src/router/llm/adapters/langchain_adapter.py
+++ b/src/router/llm/adapters/langchain_adapter.py
@@ -120,7 +125,12 @@ class LangChainAdapter(ProviderAdapter):
         return get("input_tokens"), get("output_tokens")
 
     def _cost(self, in_tok: Optional[int], out_tok: Optional[int]) -> Optional[float]:
+        """``None`` (unknown) when there are no token counts at all, or when a
+        side with a nonzero token count has no configured price -- an unpriced
+        side must not be silently treated as free (LANGCHAINADAPTER-002)."""
         in_price, out_price = self._prices
-        if (in_tok is None and out_tok is None) or (in_price is None and out_price is None):
+        if in_tok is None and out_tok is None:
+            return None
+        if (in_tok and in_price is None) or (out_tok and out_price is None):
             return None
         return float((in_tok or 0) * (in_price or 0.0) + (out_tok or 0) * (out_price or 0.0))
```
