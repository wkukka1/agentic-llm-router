# LLMRESPONSE-001: `raw`/`meta` fields lost in the migration from `agentic.llm_clients.LLMResponse`

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** low (nothing reads them today, but the loss is undocumented)
- **Class:** `LLMResponse` ([router/llm/client.py:25-33](../../../src/router/llm/client.py#L25-L33))
- **Found:** 2026-09-22

## What

The pre-migration `agentic.llm_clients.LLMResponse` carried `raw`/`meta`
fields alongside content/tokens/cost/latency. The current, unified
`router.llm.client.LLMResponse` — now used by the agentic flow too — only has
`content`, `input_tokens`, `output_tokens`, `cost`, `latency`. Nothing in
`src/` currently reads `raw`/`meta`, so this hasn't broken a live path, but
the field loss isn't called out anywhere (docstring, changelog, or migration
note).

## Failure scenario

Latent. A future caller expecting provider-specific raw response data or
metadata (e.g. finish reason, tool-call payloads, moderation flags) attached
to a completion has nowhere to get it from `LLMResponse` — it would need to
be re-added, and nothing today signals that it used to exist.
