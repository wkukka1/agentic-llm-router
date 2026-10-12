# Code review: router.llm

**Files reviewed:** 5 (290 lines): `adapters/fakes.py` (48, new) · `adapters/langchain_adapter.py` (126, new) · `adapters/openai.py` (25) · `cost.py` (53, new) · `registry.py` (38)
**Context-only:** `adapters/base.py`, `adapters/anthropic.py`, `adapters/google.py`, `adapters/__init__.py`, `client.py`, `profile.py`, `llm/__init__.py`, `router/context.py`, `router/routing/base.py`, `router/models/registry.py`, `router/agentic/llm_clients.py`, `router/agentic/router.py`
**Summary:** 4 findings: 3 correctness, 1 design · 2 confirmed, 2 plausible

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| RL-01 | `src/router/llm/adapters/langchain_adapter.py:83-95` | correctness | high | confirmed |
| RL-02 | `src/router/llm/adapters/langchain_adapter.py:20-33` | correctness | medium | confirmed |
| RL-03 | `src/router/llm/adapters/langchain_adapter.py:98-104` | correctness | medium | plausible |
| RL-04 | `src/router/llm/adapters/fakes.py`, `langchain_adapter.py` | design | low | plausible |

---

## src/router/llm/adapters/langchain_adapter.py

### RL-01: `complete`/`stream` silently drop `generation_kwargs` on the one adapter that makes real network calls `:83-95`
**Category:** correctness · **Severity:** high · **Confidence:** confirmed

```python
83:    def complete(self, prompt: Prompt, **generation_kwargs) -> LLMResponse:
84:        msg = self._model().invoke(prompt)
85:        in_tok, out_tok = self._tokens(getattr(msg, "usage_metadata", None))
86:        return LLMResponse(
...
92:    def stream(self, prompt: Prompt, **generation_kwargs) -> Iterator[str]:
93:        for chunk in self._model().stream(prompt):
94:            yield message_text(chunk)
```

**What breaks:** `ProviderAdapter` (`adapters/base.py:18-20`) documents `generation_kwargs` as the per-call channel for `max_tokens`, `temperature`, `stop`, etc. — and `adapters/anthropic.py`'s docstring explicitly relies on that contract ("take `max_tokens` from `generation_kwargs`"). `LangChainAdapter` — the module docstring's own words, "the one `ProviderAdapter` that actually reaches a network today" — accepts `**generation_kwargs` in both `complete` and `stream` but never forwards it to `self._model().invoke(...)`/`.stream(...)`. Any caller that does `LLMClient(LangChainAdapter(...)).complete(prompt, temperature=0, max_tokens=200)` — exactly the call shape the interface invites — gets a real, billed API call made with whatever defaults the underlying chat model happens to have, silently ignoring the caller's temperature/max_tokens/stop. No exception, no warning.
**Evidence:** Compared against `adapters/base.py`'s `ProviderAdapter.complete(self, prompt, **generation_kwargs)` contract and `adapters/anthropic.py`'s docstring. Confirmed no forwarding exists in either method body. This behavior is carried over unchanged from the pre-refactor `LangChainClient.invoke` (`git diff main...HEAD -- src/router/agentic/llm_clients.py`, old `invoke(self, prompt, **kw): msg = self._model().invoke(prompt)`), so it is not a new regression from this branch, but the migration was an opportunity to fix it and didn't. Today's only call site (`router/agentic/router.py:208`, `self.clients.complete(model_id, prompt)`) never passes generation kwargs, so it isn't exercised on the current live path — but nothing stops a direct caller (or a future agentic feature that does pass `temperature=`) from silently losing cost/determinism control on a real paid endpoint.
**Direction:** Forward the kwargs: `self._model().invoke(prompt, **generation_kwargs)` and `self._model().stream(prompt, **generation_kwargs)`.

---

### RL-02: `guess_provider`'s token-start-only matching produces confirmed wrong guesses against this repo's own model ids `:20-33`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
20:    _PROVIDER_HINTS = [
21:        ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"), ("davinci", "openai"),
22:        ("claude", "anthropic"),
23:        ("gemini", "google_genai"), ("palm", "google_genai"), ("bison", "google_genai"),
24:        ("mistral", "mistralai"), ("mixtral", "mistralai"),
25:        ("llama", "groq"), ("command", "cohere"), ("qwen", "together"),
...
29:    def guess_provider(model_id: str) -> Optional[str]:
...
35:        low = str(model_id).lower()
36:        for needle, provider in _PROVIDER_HINTS:
37:            if re.search(rf"(?:^|[^a-z0-9]){re.escape(needle)}", low):
38:                return provider
```

**What breaks:** The regex anchors only the *start* of a hint token (`(?:^|[^a-z0-9])gpt`), never its end, so any id that merely *starts with* a hint's characters after a boundary matches — not just ids equal to or hyphen-joined with the hint. Using this codebase's own `configs/model_registry.yaml`:
- `gpt4all-13b-snoozy` (provider `nomic`) → matches the `"gpt"` hint → `guess_provider` returns `"openai"`.
- `llama2-70b-steerlm-chat` (provider `nvidia`) and `code-llama-34b-instruct` (provider `meta`) → match `"llama"` → both return `"groq"`.

`ClientRegistry.langchain(model_ids)` (`router/agentic/llm_clients.py:79-90`) calls this as its default provider resolution when a pool id has no explicit `provider=`/`model=` override, and then hands the guessed provider straight to `init_chat_model(model_provider=...)`. For `gpt4all-13b-snoozy` this means a real OpenAI API call is attempted for a model OpenAI doesn't serve, using whatever OpenAI credentials are configured — a wrong-provider dispatch, not just a cosmetic label error.
**Evidence:** Traced hint list against `configs/model_registry.yaml` provider fields (`nomic`, `nvidia`, `meta` for the ids above) and the `guess_provider` regex logic by hand.
**Direction:** Require a token boundary on both ends (`rf"(?:^|[^a-z0-9]){re.escape(needle)}(?:$|[^a-z0-9])"`), or restrict hints to whole family prefixes with an explicit separator (`gpt-`, `gpt3`, `gpt4`) rather than a bare `"gpt"`/`"llama"` substring.

---

### RL-03: partial per-call pricing silently prices the unset side at 0 instead of falling back to profile-based pricing `:98-104`
**Category:** correctness · **Severity:** medium · **Confidence:** plausible

```python
98:    @staticmethod
99:    def _tokens(usage) -> tuple[Optional[int], Optional[int]]:
...
102:    def _cost(self, in_tok: Optional[int], out_tok: Optional[int]) -> Optional[float]:
103:        in_price, out_price = self._prices
104:        if (in_tok is None and out_tok is None) or (in_price is None and out_price is None):
105:            return None
106:        return float((in_tok or 0) * (in_price or 0.0) + (out_tok or 0) * (out_price or 0.0))
```

**What breaks:** `_cost` only returns `None` (triggering `LLMClient.complete`'s profile-based fallback pricing, `client.py:51-52`) when *both* `input_cost_per_token` and `output_cost_per_token` are unset. If a caller supplies only one price (e.g. copies `input_cost_per_token` from a price table that only had prompt pricing, or forgets `output_cost_per_token`), the adapter still returns a concrete (non-`None`) cost — with the missing side silently treated as free (`out_price or 0.0`) — rather than returning `None` and letting the `LLMProfile`-based price (if a profile is attached) fill the gap. This is the same shape as the "pricing dropped" issue the task flagged to check for recurrence: here it's not dropped to zero total, but a real (nonzero) yet systematically *under*-priced cost is produced with no signal that anything is missing.
**Evidence:** Read `client.py:46-53` (`LLMClient.complete`: only calls `self.price(...)` when `resp.cost is None`) alongside this method. This exact logic (single `usage` arg version) also existed pre-refactor in `router/agentic/llm_clients.py`'s old `LangChainClient._cost`, so it's a persisting gap rather than a new one; flagged because the migration was the natural point to fix it and the task explicitly asked to check for recurrence of this class of bug.
**Direction:** Either require both prices to be set together (raise/warn if exactly one is given), or return `None` when either price is missing so `LLMClient` can fall back to the profile's price for that model instead of assuming the unset side is free.

---

## src/router/llm/adapters/fakes.py, src/router/llm/adapters/langchain_adapter.py

### RL-04: no dedicated unit tests for the two new adapter modules `:-`
**Category:** design · **Severity:** low · **Confidence:** plausible

**What breaks:** `tests/router/llm/` contains only `test_cost.py`; there is no `tests/router/llm/adapters/` suite. `EchoAdapter`/`CallableAdapter`/`LangChainAdapter` are exercised only indirectly through `tests/router/agentic/test_agentic_router.py` and `tests/router/test_review_fixes.py`, which cover the agentic-facing behavior (via `ClientRegistry`) but never call `LangChainAdapter.complete`/`.stream` with `generation_kwargs`, never test `_cost`'s partial-price branch, and never test `guess_provider` against the repo's own registry ids — which is exactly how RL-01/RL-02/RL-03 above went unnoticed.
**Evidence:** `find tests/router/llm -type f` → only `test_cost.py`; grepped `EchoAdapter|CallableAdapter|LangChainAdapter|guess_provider` and found all references live under `tests/router/agentic/`.
**Direction:** Add a `tests/router/llm/adapters/test_langchain_adapter.py` (with a stub `chat_model` object, no real `langchain` dependency needed) covering: `generation_kwargs` forwarding, the partial-price `_cost` branch, and a couple of `guess_provider` cases pulled from `configs/model_registry.yaml`.
