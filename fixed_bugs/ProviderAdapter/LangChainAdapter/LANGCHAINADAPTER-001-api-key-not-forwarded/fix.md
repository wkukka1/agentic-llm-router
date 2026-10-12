# LANGCHAINADAPTER-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-25, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`LangChainAdapter._model()` now merges `self.api_key` into the kwargs passed to `init_chat_model` when it is not `None`. Tests: `tests/router/llm/adapters/test_langchain_adapter.py::test_model_forwards_api_key_to_init_chat_model` / `test_model_omits_api_key_when_not_given`.

## Fix suggested when the bug was filed

Include `api_key` in the kwargs passed to `init_chat_model` when it is not `None`.

## Regression tests

`tests/router/llm/adapters/test_langchain_adapter.py:93` `test_model_forwards_api_key_to_init_chat_model`

```python
def test_model_forwards_api_key_to_init_chat_model(monkeypatch):
    """LANGCHAINADAPTER-001: an explicit api_key must reach init_chat_model,
    not be silently dropped in favour of ambient credentials."""
    pytest.importorskip("langchain")
    import langchain.chat_models

    captured = {}

    def fake_init_chat_model(name, *, model_provider=None, **kw):
        captured["name"] = name
        captured["model_provider"] = model_provider
        captured.update(kw)
        return _StubChatModel()

    monkeypatch.setattr(langchain.chat_models, "init_chat_model", fake_init_chat_model)

    adapter = LangChainAdapter("gpt-4", provider="openai", api_key="sk-test")
    adapter._model()

    assert captured["api_key"] == "sk-test"
```

`tests/router/llm/adapters/test_langchain_adapter.py:115` `test_model_omits_api_key_when_not_given`

```python
def test_model_omits_api_key_when_not_given(monkeypatch):
    """No api_key= given -- must not pass a spurious api_key=None that could
    shadow init_chat_model's own default resolution."""
    pytest.importorskip("langchain")
    import langchain.chat_models

    captured = {}

    def fake_init_chat_model(name, *, model_provider=None, **kw):
        captured.update(kw)
        return _StubChatModel()

    monkeypatch.setattr(langchain.chat_models, "init_chat_model", fake_init_chat_model)

    adapter = LangChainAdapter("gpt-4", provider="openai")
    adapter._model()

    assert "api_key" not in captured
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/llm/adapters/langchain_adapter.py`

```diff
--- a/src/router/llm/adapters/langchain_adapter.py
+++ b/src/router/llm/adapters/langchain_adapter.py
@@ -95,6 +98,8 @@ class LangChainAdapter(ProviderAdapter):
             from langchain.chat_models import init_chat_model  # optional dep
 
             name, provider, kw = self._spec
+            if self.api_key is not None:
+                kw = {**kw, "api_key": self.api_key}
             self._chat = init_chat_model(name, model_provider=provider, **kw)
         return self._chat
 
```
