# LLMCLIENTFACTORY-001: the client cache key omits the profile, so a re-priced profile keeps its old prices

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** medium
- **Class:** `LLMClientFactory.client_for` ([client.py:112-124](../../../src/router/llm/client.py#L112-L124))
- **Found:** 2026-09-19

## What

The cache key is `(provider, profile.model_id, api_key, client_kwargs)`. The `LLMClient` it
stores holds the `LLMProfile` it was created with, and `LLMClient.price()` reads
`profile.input_cost_per_token` / `output_cost_per_token`. A later call with a different
profile for the same provider and model id hits the cache and returns the client bound to
the **first** profile.

`LLMRegistry.register(profile, replace=True)` exists specifically so a model can be re-priced
("silently keeping the last one would route with the wrong price"). This cache reintroduces
that failure one layer down.

## Failure scenario

Repro (`known_bugs/_repros/repro_agentic.py`, section `LLMCLIENTFACTORY-001`): register a
profile at 1e-6 per output token, get its client, register a replacement at 5e-6, get the client
again:

```
client_for(new).price(0, 1000) = 0.001   (expected 0.005)
```

Every response priced by `LLMClient` after a price change is costed at the old rate until the
factory is rebuilt.
