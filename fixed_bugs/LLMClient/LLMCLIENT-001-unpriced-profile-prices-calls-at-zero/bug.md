# LLMCLIENT-001: a profile with no configured price makes every counted call cost `0.0`

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** low
- **Class:** `LLMClient.price` ([client.py:58-64](../../../src/router/llm/client.py#L58-L64)) with `LLMProfile` defaults ([profile.py:22-23](../../../src/router/llm/profile.py#L22-L23))
- **Found:** 2026-09-19

## What

`LLMResponse` documents `None` as "unknown", distinct from "genuinely free (`0`)". `price()`
returns `None` when there is no profile or no token counts, which is right. But
`LLMProfile.input_cost_per_token` / `output_cost_per_token` default to `0.0`, so a profile whose
prices were never set is indistinguishable from a free model: `price()` returns `0.0`.

## Failure scenario

Repro (`known_bugs/_repros/repro_serving_cost.py`, section `LLMCLIENT-001`):

```
price(1000, 1000) = 0.0     # profile built with name/provider/model_id only
```

A registry populated from a config that omits prices reports zero cost for every call, and
`CostModel.predict_cost` predicts zero, so a cost-weighted router sees the model as free.
