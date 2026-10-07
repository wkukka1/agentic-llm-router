# LANGCHAINADAPTER-001: an explicit `api_key` never reaches `init_chat_model`

- **Status:** fixed 2026-09-25, see [fix.md](fix.md)
- **Severity:** medium
- **Class:** `LangChainAdapter` ([langchain_adapter.py:85-98](../../../../src/router/llm/adapters/langchain_adapter.py#L85-L98))
- **Found:** 2026-09-19 (first reported by the earlier diff review; re-checked top-down)

## What

`__init__` takes `api_key` as a named parameter, so it is *not* part of `client_kwargs`. It
is passed to `super().__init__` (which stores it) and then dropped: `self._spec` holds only
`(model, provider, client_kwargs)`, and `_model()` calls
`init_chat_model(name, model_provider=provider, **kw)` with no key.

## Failure scenario

`LLMClientFactory.client_for(profile, api_key="sk-...")` with this adapter uses whatever
credentials the environment has, or fails with an auth error. The key the caller supplied
is silently ignored.
