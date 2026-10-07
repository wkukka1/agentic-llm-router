# MOD-001: `guess_provider` matches hints mid-token, mis-guessing real model ids

- **Status:** fixed 2026-09-25, see [fix.md](fix.md)
- **Severity:** medium
- **Where:** `guess_provider` ([router/llm/adapters/langchain_adapter.py:35-44](../../../../src/router/llm/adapters/langchain_adapter.py#L35-L44))
- **Found:** 2026-09-22

## What

```python
if re.search(rf"(?:^|[^a-z0-9]){re.escape(needle)}", low):
    return provider
```

The lookaround only guards the **start** of the match (must follow a
non-alnum char or string start); nothing guards the end. A hint like `"gpt"`
matches inside `"gpt4all-13b-snoozy"` even though that model is not an
OpenAI GPT model — `guess_provider("gpt4all-13b-snoozy")` returns
`"openai"`.

## Failure scenario

Any `configs/model_registry.yaml` entry whose id happens to contain a
provider hint as a substring (not just a prefix/token) gets routed to the
wrong LangChain provider when no explicit `provider=` is given — a real,
not hypothetical, entry (`gpt4all-13b-snoozy`) already triggers it.
