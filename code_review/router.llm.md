# Code review: router.llm

**Files reviewed:** 9 (237 lines) · **Context-only:** `src/router/agentic/llm_clients.py`, `src/router/router.py`, `configs/model_registry.yaml`, `src/training/data/model_registry.py`
**Summary:** 9 findings: 4 correctness, 4 design, 1 performance · 8 confirmed, 1 suspected

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| RL-01 | `src/router/llm/client.py:56-69` | correctness | medium | confirmed |
| RL-02 | `src/router/llm/client.py:54-68` | correctness | medium | suspected |
| RL-03 | `src/router/llm/__init__.py:1-14` | design | low | confirmed |
| RL-04 | `src/router/llm/adapters/base.py:13-20` | design | low | confirmed |
| RL-05 | `src/router/llm/adapters/anthropic.py:19-27` | correctness | low | confirmed |
| RL-06 | `src/router/llm/registry.py:21-22` | correctness | low | confirmed |
| RL-07 | `src/router/llm/client.py:22-28` | design | low | confirmed |
| RL-08 | `src/router/llm/adapters/openai.py:18-21` | design | low | confirmed |
| RL-09 | `src/router/llm/client.py:56-69` | performance | low | confirmed |

---

## src/router/llm/__init__.py

### RL-03: LLM profile/client/adapter layer is scaffolding, not wired `:1-14`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
1:  """LLM infrastructure: profiles, a registry, a client interface, and provider
2:  adapters. Scaffolding for the production router design -- see
```

**What breaks:** Nothing today. `LLMClientFactory`, `LLMClient`, and all adapters are unreachable from live code, and every adapter raises `NotImplementedError`. `LLMProfile` and `LLMRegistry` are used by `RoutingPipeline` (`src/router/router.py:35,63`) for candidate lists only, never for dispatch.
**Evidence:** Ran `grep -rn "router\.llm\|LLMClientFactory\|LLMRegistry\|ProviderAdapter" src scripts tests configs docs` (excluding the package). Hits: `router/router.py:14,35,48,63,110`, `routing/base.py:178` (docstring), `tests/router/test_router_pipeline.py:15-16,45-96` (profile and registry only), `docs/architecture.md:231-237`. By design.
**Direction:** Keep as-is. The findings below are the things to fix when you implement the adapters.

---

## src/router/llm/client.py

### RL-01: `client_for` drops profile pricing and client kwargs, so `LLMResponse.cost` can only be 0.0 `:56-69`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
56:    def client_for(self, profile: "LLMProfile", *, api_key: Optional[str] = None) -> LLMClient:
69:        return LLMClient(Adapter(profile.model_id, api_key=api_key))
```

**What breaks:** The adapter receives only `model_id` and `api_key`. `LLMProfile.input_cost_per_token` / `output_cost_per_token` (profile.py:16-17) never reach the adapter or `LLMClient`, and `LLMClient.complete` (lines 39-40) passes the adapter's response through without computing cost from tokens. A correctly implemented adapter therefore has no way to fill `cost`, which stays at its `0.0` default (line 27). Once `ExecutionResult.actual_cost` is filled from `LLMResponse.cost`, every `BudgetLedger.commit` records zero spend and the budget never binds. Adapter `**client_kwargs` (timeout, base_url, max_tokens) are also unreachable through the factory.
**Evidence:** Traced `LLMProfile` -> `client_for` (line 69) -> `Adapter.__init__(model, *, api_key, **client_kwargs)` (e.g. `adapters/openai.py:18-21`) -> `LLMClient.complete` (line 40). No other path carries the profile. No tests touch `LLMClientFactory` (`grep -rn "LLMClientFactory" tests` returns nothing).
**Direction:** Have `LLMClient` hold the profile and compute `cost = in_tok*input_cost_per_token + out_tok*output_cost_per_token` after the adapter returns. Let `client_for` accept and forward `**client_kwargs`.

### RL-02: Provider allow-list covers only 3 providers; most of the configured model pool would raise `:54-68`
**Category:** correctness · **Severity:** medium · **Confidence:** suspected

```python
54:    _PROVIDERS = {"openai", "anthropic", "google"}
64:        else:
65:            raise ValueError(
```

**What breaks:** `configs/model_registry.yaml` tags models with providers such as `meta`, `mistralai`, `zhipu`, `deepseek`, `databricks`, `tii`, `lmsys`, `stanford`, `microsoft`, `01-ai` (e.g. lines 105, 118-131), and the loader defaults to `"unknown"` (`src/training/data/model_registry.py:59`). If `LLMProfile.provider` is populated from that registry, `client_for` raises `ValueError` for most of the RouterBench/Arena candidate pool. The live `ClientRegistry` falls back to a default factory instead (`llm_clients.py:183-187`). The two-place provider list (the `_PROVIDERS` set plus the if-chain) can also drift apart.
**Evidence:** Ran `grep -rhn "provider:" configs` to collect provider values. Suspected because no code currently builds `LLMProfile`s from `model_registry.yaml`. Maintainer question: will `LLMProfile.provider` come from `model_registry.yaml`'s `provider` field (the model author) or from the serving provider (openai/anthropic/google/together/...)? If the latter, document that the two vocabularies differ.
**Direction:** Separate "model author" from "serving provider" on `LLMProfile`. Drive dispatch from a single dict `{provider: adapter import path}`, with a configurable fallback adapter (e.g. an OpenAI-compatible endpoint).

### RL-07: `LLMResponse` uses 0 defaults that can't distinguish "unknown" from "free" `:22-28`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
25:    input_tokens: int = 0
26:    output_tokens: int = 0
27:    cost: float = 0.0
```

**What breaks:** An adapter whose provider omits usage metadata (common in streaming) returns tokens=0 and cost=0.0, which is indistinguishable from a genuinely free call. The same goes for `latency=0.0`: `LLMClient` never times the call, so latency is 0 unless every adapter measures it. Downstream budget and forecast calibration would silently learn "free and instant".
**Evidence:** Read lines 22-43. `LLMClient.complete` / `stream` are pure delegation with no timing.
**Direction:** Use `Optional[...] = None` for usage, cost, and latency, and measure latency once in `LLMClient` around the adapter call.

### RL-09: A new adapter (SDK client) is constructed on every `client_for` call `:56-69`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
69:        return LLMClient(Adapter(profile.model_id, api_key=api_key))
```

**What breaks:** Once adapters create real SDK clients (HTTP connection pools, auth setup), resolving a client per request rebuilds that per call. The working `ClientRegistry.get` caches per `model_id` (`llm_clients.py:183-187`).
**Evidence:** `LLMClientFactory` holds no cache (lines 46-69).
**Direction:** Memoize clients by `(provider, model_id, api_key)` in the factory, or resolve them once per registry entry.

---

## src/router/llm/profile.py

No findings. (Note for the `router.router` reviewer, outside this slice: `router.py:98` feeds `output_cost_per_token`, a per-token price, into `ModelScore.expected_cost` as if it were a per-request cost.)

---

## src/router/llm/registry.py

### RL-06: `register` silently overwrites an existing profile with the same `model_id` `:21-22`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
21:    def register(self, profile: LLMProfile) -> None:
22:        self._profiles[profile.model_id] = profile
```

**What breaks:** Profiles are keyed only by `model_id`. The same model served by two providers (e.g. a Llama model via `together` and via `groq`, each with different pricing) collapses to whichever was registered last, with no error. `RoutingPipeline.build_context` (`router.py:63`) then lists one candidate with the wrong price.
**Evidence:** Read lines 17-28. `tests/router/test_router_pipeline.py:45-96` registers unique ids only.
**Direction:** Raise on duplicate `model_id` (or require an explicit `replace=True`), or key by `(provider, model_id)`.

---

## src/router/llm/adapters/__init__.py

No findings.

---

## src/router/llm/adapters/base.py

### RL-04: The adapter interface carries only a bare prompt string, and `stream` returns a finished response `:13-20`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
15:    def complete(self, prompt: str) -> "LLMResponse":
19:    def stream(self, prompt: str) -> "LLMResponse":
```

**What breaks:** The interface has no way to pass a system prompt, a message history, `max_tokens`, `temperature`, or stop sequences. `RoutingRequest.conversation` exists (`router.py:61`) but can't reach a provider through `LLMClient.complete(prompt)`. `stream` has the same return type as `complete` (a single `LLMResponse`), so an implementation can't actually yield incremental chunks. Implementers will either buffer the whole stream or break the signature.
**Evidence:** Read `base.py:13-20` and `client.py:39-43`. The live client already accepts `**kw` (`llm_clients.py:50`).
**Direction:** Accept `messages` plus generation kwargs in `complete`, and make `stream` return an `Iterator[str]` (or chunk objects) with the final usage exposed separately.

---

## src/router/llm/adapters/anthropic.py

### RL-05: Anthropic's required `max_tokens` has no path through the factory `:19-27`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
19:    def __init__(self, model: str, *, api_key: Optional[str] = None, **client_kwargs):
24:    def complete(self, prompt: str) -> LLMResponse:
```

**What breaks:** The Anthropic Messages API rejects requests without `max_tokens`. The only channel for it is `client_kwargs`, which `LLMClientFactory.client_for` never forwards (`client.py:69`), and `complete(prompt)` takes no per-call kwargs. An implementation wired through the factory must hard-code a token cap or fail on the first call.
**Evidence:** Traced `client.py:56-69` -> `AnthropicAdapter.__init__`. Specific instance of RL-01/RL-04.
**Direction:** Resolved by forwarding `client_kwargs` (RL-01) or per-call generation kwargs (RL-04). Otherwise, default `max_tokens` from `LLMProfile.context_window`, or from a config value.

---

## src/router/llm/adapters/google.py

No findings beyond RL-08 (the duplicated constructor, which this file shares).

---

## src/router/llm/adapters/openai.py

### RL-08: Identical constructors copy-pasted across all three adapters `:18-21`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
18:    def __init__(self, model: str, *, api_key: Optional[str] = None, **client_kwargs):
19:        self.model = model
20:        self.api_key = api_key
```

**What breaks:** The same 4-line `__init__` appears in `openai.py:18-21`, `anthropic.py:19-22`, and `google.py:12-15`, while `ProviderAdapter` (base.py:13) has none. Any change to the construction contract, such as adding `profile` (RL-01), must be made three times, and `LLMClientFactory` assumes all three accept the same signature.
**Evidence:** Ran `grep -rn "def __init__(self, model: str" src`: exactly those three hits.
**Direction:** Move the constructor into `ProviderAdapter`.

---

## Checked and ruled out
- `registry.py:27` defining a method named `list` whose return annotation is `list[LLMProfile]`: with `from __future__ import annotations`, the annotation is never evaluated in class scope, and `get_type_hints` resolves `list` from module globals. No runtime issue.
- `client.py:57` calling `profile.provider.lower()` on a `None` provider: `LLMProfile.provider` is a required str field, and every in-repo construction passes one.
- `client.py` lazy adapter imports: the adapters import only `..client` / `.base`, so there is no circular import (`client.py` imports the adapter base only under TYPE_CHECKING).
- `registry.py:24-25` `get` raising a bare `KeyError`: acceptable for a lookup, and `RoutingPipeline` uses `list()` only.
- Import-linter: `router.llm` imports nothing outside `router`; no violation.
