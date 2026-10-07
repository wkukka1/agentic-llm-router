# Code review: router.agentic

**Files reviewed:** 3 (314 lines): `src/router/agentic/__init__.py` (85), `src/router/agentic/llm_clients.py` (114), `src/router/agentic/router.py` (243) · **Context-only:** `src/router/agentic/{orchestrator,decompose,triage,result}.py` (unchanged), `src/router/llm/{client.py,cost.py,registry.py,adapters/{base,fakes,langchain_adapter}.py}`, `docs/agentic_router.md`, `README.md`

**Summary:** 3 findings: 0 correctness, 2 design, 0 performance, 1 convention · 2 confirmed, 1 plausible

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| RA-01 | `src/router/agentic/llm_clients.py:34-36` (via `router/llm/client.py:24-33`) | design | medium | confirmed |
| RA-02 | `src/router/llm/client.py:1-10`, `src/router/llm/__init__.py:1-14`, `README.md:76` | convention | low | confirmed |
| RA-03 | `src/router/agentic/orchestrator.py:41` (reached through `router.py`'s now-adapter-backed `_call_model`) | performance | low | plausible |

None of the four previously-known findings (missed real-model cost / double-counted nested-orchestration cost, no error handling, silent echo for unknown ids, broken sentence splitter) recur in this rewrite. `_call_model` still wraps the client call in `try/except` and only swallows the error at `depth > 0` (`src/router/agentic/router.py:204-220`); `_solve` still sums `cost` over leaves only (`router.py:176`); `ClientRegistry.get` still warns when synthesising an echo client onto a registry that has real clients (`llm_clients.py:99-104`); `NaiveDecomposer` is untouched by this diff. All 47 tests in `tests/router/agentic/` and `tests/router/test_review_fixes.py` pass against the migrated code.

---

## src/router/agentic/llm_clients.py

### RA-01: `LLMResponse` silently loses `raw`/`meta` (and the `model_id`/echo-marker) on the adapter migration `:34-36`
**Category:** design · **Severity:** medium · **Confidence:** confirmed

```python
34: from ..llm.adapters.fakes import EchoAdapter
35: from ..llm.adapters.langchain_adapter import LangChainAdapter, guess_provider, message_text
36: from ..llm.client import LLMClient, LLMResponse
```

The pre-migration `LLMResponse` (deleted by this diff) carried:
```python
text: str
model_id: str
cost: Optional[float] = None
raw: object = None
meta: dict = field(default_factory=dict)
```
`LangChainClient.invoke` populated `raw=msg` (the full LangChain `AIMessage`) and `meta={"response_metadata": ..., "usage_metadata": ...}`; `EchoClient.invoke` set `meta={"echo": True}`.

The new `router.llm.client.LLMResponse` this file now re-exports is:
```python
content: str
input_tokens: Optional[int] = None
output_tokens: Optional[int] = None
cost: Optional[float] = None
latency: Optional[float] = None
```
`raw`, `meta` (and `model_id`, though that one is redundant with `SubCall.model_id`) have no equivalent field. `LangChainAdapter.complete` (`src/router/llm/adapters/langchain_adapter.py:101-109`) only extracts `input_tokens`/`output_tokens` from `usage_metadata` and discards the message object and `response_metadata` entirely; there is nowhere left to put them since the dataclass itself dropped the fields.

**What breaks:** nothing today — I grepped the repo and no code reads `.raw`, `.meta`, or `.model_id` off an `LLMResponse`/`SubCall`. But this is a real, silent capability regression in the shared response type for the one adapter that actually reaches a network: `response_metadata` is the normal place to find `finish_reason` (truncation/length-based retry logic), request ids, and safety-filter flags, and `raw` is the only way to inspect tool calls or multi-block content on the original message. Any future consumer (retry-on-truncation, audit logging, `LangChainToolOrchestrator`-style inspection through the routed-model client rather than the orchestrating LLM) has no field to read it from anymore, and the loss isn't called out anywhere in the diff or docstrings.
**Evidence:** diffed the deleted `LLMResponse` dataclass in `llm_clients.py` (pre-migration) against `router/llm/client.py:24-33`; confirmed via `grep -rn "\.raw\b|response_metadata|resp\.meta"` under `src/` that no current caller depends on the dropped fields (only unrelated `evaluation.judge` and `training.nirt.baseline` hits, both different types).
**Direction:** either add `raw`/`response_metadata` fields to `router.llm.client.LLMResponse` (all adapters can leave them `None`/`{}`) or explicitly document in the `LLMResponse` docstring that this is a deliberate narrowing and where finish-reason/raw-message data should now come from.

---

## src/router/llm/client.py / src/router/llm/__init__.py / README.md

### RA-02: Cross-package docstrings still describe the pre-migration agentic client as "what's live today" `:1-10`
**Category:** convention · **Severity:** low · **Confidence:** confirmed

`src/router/llm/client.py:6-9`:
```python
6: A different, non-interacting provider vocabulary ... `router.agentic.llm_clients` layer
   (``invoke(prompt)``, no ``complete``/``stream`` split, no adapter indirection)
   -- that one is what the live agentic flow actually calls today.
```
`src/router/llm/__init__.py:5-9` makes the same claim ("the simpler, already-working client layer the live agentic flow actually dispatches through today (``invoke(prompt)``, no ``complete``/``stream`` split, no ``ProviderAdapter`` indirection)"), and `README.md:76` still names the deleted `LangChainClient` class ("for `LangChainClient` and `LangChainToolOrchestrator`").

**What breaks:** nothing at runtime, but this diff is precisely the commit that made both statements false: `router.agentic.llm_clients.LLMClient`/`LLMResponse` are now literally `router.llm.client.LLMClient`/`LLMResponse` (imported and re-exported unchanged, `complete`/`stream` split and adapter indirection included) — the two "distinct, not-yet-unified" layers the docstrings describe no longer exist as separate things. A reader following these docstrings (which `src/router/agentic/llm_clients.py`'s own module docstring points to implicitly via the shared naming) would conclude the agentic path bypasses the adapter layer, which is now wrong.
**Evidence:** `git diff main...HEAD -- src/router/agentic/llm_clients.py` shows the direct `from ..llm.client import LLMClient, LLMResponse` import; `router/llm/client.py` and `router/llm/__init__.py` were not touched by this branch's agentic-migration commits (`d72444b`, `bd2c6c3`) even though their docstrings are specifically about the relationship this migration changed.
**Direction:** update both module docstrings (and the README line) to say the two layers were unified by the adapter migration, keeping only the still-true note that `LLMClientFactory`/`LLMRegistry` (profile-based provider selection) have no production caller yet.

---

## src/router/agentic/orchestrator.py (reached via the migrated `router.py`)

### RA-03: Sibling sub-prompts are still dispatched to the LLM sequentially, one blocking `.complete()` at a time `:41`
**Category:** performance · **Severity:** low · **Confidence:** plausible

```python
41: children = [agent._solve(s, depth + 1) for s in subs]
```
`RecursiveOrchestrator.run` decomposes a compound prompt into independent sub-prompts and then solves each one in a plain list comprehension. Each `agent._solve` → `_answer_directly` → `_call_model` → `self.clients.complete(...)` (`router.py:208`) is a synchronous, blocking network call now routed through `LLMClient.complete`/`ProviderAdapter.complete` (`router/llm/client.py:46-53`, `router/llm/adapters/langchain_adapter.py:101-109`). Sub-tasks produced by `NaiveDecomposer`/`LLMDecomposer` have no data dependency on each other, so they could be dispatched concurrently (thread pool, since the LangChain SDKs are sync) instead of one after another.
**What breaks:** not a correctness bug, but wall-clock latency for an orchestrated request scales linearly with fan-out (`max_depth=2` plus multi-part decomposition can already reach several sequential network round-trips per `run()`), which is exactly the kind of avoidable sequential-await cost this migration to a real network-calling adapter (vs. the old echo-only default) makes newly consequential in practice.
**Evidence:** `orchestrator.py:41` is unchanged by this diff, but the diff is what makes the sequential dispatch actually reach a network (`LangChainAdapter` is "the one `ProviderAdapter` that actually reaches a network today", per its own module docstring) rather than only ever hitting the deterministic in-process `EchoClient` in tests/default use.
**Direction:** run independent children concurrently, e.g. `concurrent.futures.ThreadPoolExecutor` over `subs` in `RecursiveOrchestrator.run`, preserving order when zipping results back into `children`.
