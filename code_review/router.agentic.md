# Code review: router.agentic

**Files reviewed:** 7 (879 lines) · **Context-only:** `src/router/routing/base.py`
**Summary:** 21 findings: 16 correctness, 3 design, 2 performance · 15 confirmed, 6 suspected

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| RA-01 | `src/router/agentic/router.py:137-142` | correctness | medium | confirmed |
| RA-02 | `src/router/agentic/triage.py:60-77` | correctness | medium | confirmed |
| RA-03 | `src/router/agentic/llm_clients.py:114-124` | correctness | medium | confirmed |
| RA-04 | `src/router/agentic/llm_clients.py:183-187` | correctness | medium | confirmed |
| RA-05 | `src/router/agentic/router.py:144-156` | correctness | medium | confirmed |
| RA-06 | `src/router/agentic/triage.py:124-130` | correctness | medium | confirmed |
| RA-07 | `src/router/agentic/decompose.py:28-37` | correctness | medium | confirmed |
| RA-08 | `src/router/agentic/decompose.py:74-80` | correctness | medium | confirmed |
| RA-09 | `src/router/agentic/orchestrator.py:109-119` | correctness | medium | suspected |
| RA-10 | `src/router/agentic/orchestrator.py:98-100` | correctness | medium | suspected |
| RA-11 | `src/router/agentic/llm_clients.py:114-120` | correctness | medium | suspected |
| RA-12 | `src/router/agentic/triage.py:26-29` | correctness | low | confirmed |
| RA-13 | `src/router/agentic/orchestrator.py:32-36` | performance | low | confirmed |
| RA-14 | `src/router/agentic/router.py:121-131` | performance | low | confirmed |
| RA-15 | `src/router/agentic/router.py:133-136` | correctness | low | confirmed |
| RA-16 | `src/router/agentic/result.py:42-50` | correctness | low | confirmed |
| RA-17 | `src/router/agentic/orchestrator.py:32-39` | design | low | confirmed |
| RA-18 | `src/router/agentic/triage.py:100-112` | design | low | confirmed |
| RA-19 | `src/router/agentic/llm_clients.py:130-145` | design | low | suspected |
| RA-20 | `src/router/agentic/result.py:52-55` | correctness | low | suspected |
| RA-21 | `src/router/agentic/decompose.py:77-80` | correctness | low | suspected |

---

## src/router/agentic/llm_clients.py

### RA-03: Real-model cost is never recorded, so `AgenticResult.cost` is always 0 and there is no budget `:114-124`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
117:        meta = getattr(msg, "response_metadata", {}) or {}
118:        usage = getattr(msg, "usage_metadata", None)
119:        return LLMResponse(
```

**What breaks:** `LangChainClient` fills `meta["usage_metadata"]` (token counts) but never sets `LLMResponse.cost`, which defaults to `None`. `router.py:110,141` treats `None` as `0.0`, so any run with real LangChain models reports `cost == 0.0`. The LLM calls made by `LLMTriage`, `LLMDecomposer`, `LLMSynthesizer` and the `LangChainToolOrchestrator` agent never go through a client at all, so they are not traced or costed either. Nothing caps spend: no budget parameter exists, and the only limit is `max_depth` (see RA-17).
**Evidence:** Grep for `cost=` in `src/router/agentic` finds it only in `EchoClient` (`:70`, 0.0), `answer_with_model`/`_answer_directly` (which pass `resp.cost` through), and the sums. `test_cost_is_summed_from_client_responses` (`tests/router/agentic/test_agentic_router.py:181-185`) asserts `cost == 0.0` with cost-less clients, so it treats the missing value as the expected result.
**Direction:** Compute cost from `usage_metadata` using a per-model price table (or record tokens and price them later), and send triage/decompose/synthesize/orchestrator LLM calls through a costed client so their spend shows up in the result.

### RA-04: An unregistered model id silently gets an EchoClient, and the tool-calling agent can name any id `:183-187`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
185:        if mid not in self._clients:
186:            self._clients[mid] = self._default_factory(mid)
187:        return self._clients[mid]
```

**What breaks:** `AgenticRouter(router, clients={"gpt-4o": LangChainClient(...)})` wraps the dict in `ClientRegistry` with the echo default factory (`router.py:62-63`, `:166`). If the router picks a pool model that has no client, the user gets `"[answer::<id>] <prompt>"` back as a real answer, with no error or warning. `orchestrator.py:60-66` (`answer_with_model`) passes any LLM-chosen `model_id` to `clients.invoke` without checking `agent.router.model_ids`. So a hallucinated id either echoes (default registry) or, under `ClientRegistry.langchain`, calls `init_chat_model` for a model outside the routed pool, bypassing the router.
**Evidence:** `test_registry_synthesises_unknown_ids` (`test_agentic_router.py:66-71`) locks in the silent synthesis. No test mixes real and echo clients or passes an out-of-pool id to `answer_with_model`.
**Direction:** Make the echo fallback opt-in, or at least tag synthesized responses so `AgenticResult` exposes them. Validate `model_id` against the router pool in `answer_with_model` and return a tool error string for unknown ids.

### RA-11: List-of-blocks message content is stringified as a Python repr `:114-120`
**Category:** correctness · **Severity:** medium · **Confidence:** suspected

```python
116:        text = getattr(msg, "content", msg)
120:            text=text if isinstance(text, str) else str(text),
```

**What breaks:** LangChain `AIMessage.content` can be a `list[dict]` of content blocks (Anthropic with tool use or thinking, some Gemini responses, multimodal output). `str(list)` then returns `"[{'type': 'text', 'text': ...}]"` as the answer. The same pattern appears in `triage.py:107` (the verdict never starts with `SINGLE`/`DECOMP`, so it always falls back to the heuristic), `decompose.py:77` (the repr is split into bogus sub-tasks) and `decompose.py:115` (the synthesized answer is a repr).
**Evidence:** No test uses a real chat model. It is suspected because it depends on provider and model settings. Maintainer question: are any target models configured to return block content?
**Direction:** Use `msg.text` (langchain-core >= 0.3 / 1.x) or join the `text` blocks, in one shared helper used by all four call sites.

### RA-19: `guess_provider` substring hints mis-assign providers, and pool ids are not provider model names `:130-145`
**Category:** design · **Severity:** low · **Confidence:** suspected

```python
134:    ("mistral", "mistralai"), ("mixtral", "mistralai"),
135:    ("llama", "groq"), ("command", "cohere"), ("qwen", "together"),
```

**What breaks:** Every id containing `llama` maps to `groq`, including RouterBench-style ids like `meta/code-llama-instruct-34b-chat` that Groq does not serve. First-match substring order also means short needles (`o1`, `o3`, `palm`) can match unrelated ids. Separately, `LangChainClient` passes the pool id itself as the model name (`:104`, `model or model_id`). RouterBench and IRT-Router ids such as `mistralai/mistral-7b-chat` are not valid provider API names, so `ClientRegistry.langchain(pool)` fails at first invoke for most of the real pool.
**Evidence:** Grep of `configs/` for `o1|o3` found no current collision, which is why this is suspected. `test_guess_provider` covers only `gpt-4o`, `claude-3-5-sonnet` and an unknown id.
**Direction:** Use an explicit pool-id to (provider, api model name) mapping, for example from `configs/model_profiles.yaml`, and keep guessing only as a last resort.

---

## src/router/agentic/router.py

### RA-01: Cost is double-counted for nested orchestration (default `max_depth=2`) `:137-142`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
141:            cost=sum((n.cost or 0.0) for c in children for n in c.walk()),
```

**What breaks:** A branch `SubCall`'s `cost` is already the sum of its descendants. Then both this line and `run` (`:110`, `sum(... for s in root.children for n in s.walk())`) walk every node, branch nodes included, so each leaf is counted once for itself plus once for every orchestrated ancestor below the root. Reproduced with unit-cost `CallableClient`s, a forced-orchestrate triage, and a decomposer that splits the root into 2 subs and the first sub into 2 leaves (`max_depth=2`): there are 3 leaf calls, but `res.cost == 5.0`.
**Evidence:** Ran the scenario above with `.venv` Python and got `leaf calls 3 reported cost 5.0`. `test_cost_is_summed_from_client_responses` uses cost-less clients and one level of orchestration, so every cost is 0 and nesting never happens. `test_recursion_is_depth_capped` uses `max_depth=1`.
**Direction:** Sum only leaves (`not n.children`), or have branch nodes carry only their own overhead cost and sum the whole walk.

### RA-05: No error handling or retries around model calls; one failure discards the whole run `:144-156`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
152:        resp = self.clients.invoke(model_id, prompt)
```

**What breaks:** Every external call is unguarded: `_answer_directly` (`:152`), `RecursiveOrchestrator` children (`orchestrator.py:37`), `LangChainToolOrchestrator`'s `executor.invoke` (`orchestrator.py:114`), and the tool bodies (`:62`, `:71`). A single provider error (rate limit, timeout, content filter) on sub-task k of n raises out of `run()`. The k-1 answers already paid for are lost, and no partial `AgenticResult` is returned. In the LangChain path, `StructuredTool.handle_tool_error` defaults to `False` (verified: langchain-core 1.6.2 in `.venv`), so tool exceptions are not returned to the agent as observations and abort the executor. There are no retries anywhere in the package (grep `retry|backoff|except` in `src/router/agentic` finds only the three LLM fallbacks in triage/decompose).
**Evidence:** Traced call chain `run -> _solve -> orchestrator.run -> _solve/_answer_directly -> ClientRegistry.invoke -> LLMClient.invoke`, with no `try` on the path. No test injects a failing client.
**Direction:** Catch errors per sub-call, recording a failed `SubCall` (error field, answer placeholder), with bounded retry and backoff in the client layer. Set `handle_tool_error=True` on the StructuredTools.

### RA-14: Triage, possibly a paid `LLMTriage` call, still runs at `depth >= max_depth`, where its verdict is ignored `:121-131`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
124:        if decision is None:
125:            decision = self.triage_prompt(prompt)
127:        if decision.mode == "single" or depth >= self.max_depth:
```

**What breaks:** At the depth cap only `selected_model_id` / `predicted_quality` from routing are needed, but `triage_prompt` also calls `self.triage(...)`. With `LLMTriage`, that is one chat-model call per leaf whose result is discarded. For a 6-way split at depth 2, that means up to 36 wasted triage calls.
**Evidence:** `triage_prompt` (`:89-92`) always invokes `self.triage`. The depth check comes after it.
**Direction:** When `depth >= max_depth`, route and answer directly without calling triage.

### RA-15: A collapsed orchestration is reported as `single`, with the "decompose" reason, the agent's answer dropped, and the wrong depth `:133-136`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
134:        if len(children) == 1 and children[0].prompt == prompt and not children[0].children:
135:            # nothing to decompose -> it collapses back to a single routed call
136:            return children[0]
```

**What breaks:** (a) `run` then builds `mode="single"` but keeps `triage_reason=decision.reason`, for example "...; decompose to raise it" (`:102-106`). (b) In `LangChainToolOrchestrator`, if the agent makes exactly one `route_and_answer(prompt)` or `answer_with_model(prompt, m)` call, the agent's own final `answer` is discarded in favour of the raw tool output. The collapsed node also carries `depth=depth+1` (`orchestrator.py:64,71`), so a root answer reports depth 1.
**Evidence:** `test_weak_but_indivisible_prompt_collapses_to_single` asserts only `mode` and `selected_model_id`.
**Direction:** Record the collapse in `triage_reason`, and only collapse when the orchestrator did not produce its own synthesized answer. Normalize depth when collapsing.

---

## src/router/agentic/triage.py

### RA-02: NaN predicted quality is treated as a strong model and answered directly `:60-77`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
60:        weak = predicted_quality < self.quality_threshold
```

**What breaks:** `nan < 0.55` is `False`, so an unscorable prompt goes to `single` with reason `"predicted quality nan >= 0.55; answer directly"`. NaN arises whenever the router has no score. In `base.py:247-250`, `aligned_scores` reindexes a missing `query_id` to NaN, and `routing_decision` then falls back to column 0 (see `_contracts.md` §4). Repro: `MatrixRouter` plus `query_resolver=lambda t: "unknown"` gives `mode=single, selected=a, predicted_quality=nan`, so the first pool column answers silently. `LLMTriage` (`:87`) likewise tells the LLM "predicted quality nan".
**Evidence:** Ran the repro above in `.venv`. `test_matrix_router_needs_a_resolver` only uses a resolvable id.
**Direction:** Treat non-finite `predicted_quality` as unknown, either raising or at least labelling it in the reason. Don't take the "strong" branch.

### RA-06: With `lam > 0`, triage judges difficulty by the cost-selected model, not the best model in the pool `:124-130`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
129:    mid = routing_result.selected_model_ids[0]
130:    return mid, float(routing_result.selected_scores[0]), scores
```

**What breaks:** `AgenticRouter.route_decision` passes `lam=self.lam` (`router.py:81,87`), so `selected` maximizes `pred - lam*C`. The module docstring (`:5-10`) defines the triage signal as the best predicted quality `q*` in the pool. Instead the cheap model's lower quality is used, so with `lam>0` easy prompts that a strong model handles well are flagged `weak` and sent to orchestration. That adds more calls and cost, which defeats the point of cost-aware routing. `LLMTriage` also presents the cost-chosen model as "Best available model".
**Evidence:** Traced `route_text(..., lam)` → `_route_from_scores` → `routing_decision` (cost-subtracted argmax) → `selected_scores` (raw quality of that column). There are no tests with `lam != 0`.
**Direction:** Return `max(scores)` as the difficulty signal alongside the cost-aware selected model, and threshold on the former.

### RA-12: Multi-part markers `"1."` / `"2."` match decimals and version numbers `:26-29`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
27:    " and then ", " after that ", " followed by ", "step by step", "first,", "1.", "2.",
```

**What breaks:** `"What is 1.5 plus 2.25?"` at `predicted_quality=0.9` → `mode="orchestrate"` (verified). With the default `NaiveDecomposer` it usually collapses back (RA-15), at the cost of an extra route (RA-13). With `LLMDecomposer` or `LangChainToolOrchestrator` it launches a paid orchestration for a trivial prompt. `"step by step"` similarly flags any "explain step by step" request.
**Evidence:** Ran `HeuristicTriage()("What is 1.5 plus 2.25?", model_id="m", predicted_quality=0.9).mode` and got `orchestrate`.
**Direction:** Match enumerations only at line start (`^\s*\d+[.)]\s`, multiline) instead of bare substrings.

### RA-18: Broad `except Exception` hides programming errors, and `LLMDecomposer` leaves no record of the fallback `:100-112`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
108:        except Exception as exc:  # pragma: no cover - network / provider errors
109:            d = self._fallback(prompt, model_id=model_id,
```

**What breaks:** The `try` wraps prompt formatting and sorting as well as the network call, so a bug (such as a non-numeric score) silently becomes heuristic triage. `LLMTriage` at least records the exception in `reason`. `LLMDecomposer` (`decompose.py:81-82`) and `LLMSynthesizer` (`:116-117`) swallow errors with no trace at all, so a misconfigured chat model looks like "worked, used regex split / concatenation".
**Evidence:** All three are marked `pragma: no cover`, and no test exercises them.
**Direction:** Narrow the `try` to the `invoke` call and log or record the fallback reason on the result.

---

## src/router/agentic/orchestrator.py

### RA-09: Hitting `max_iterations` returns LangChain's stop message as the final answer `:109-119`
**Category:** correctness · **Severity:** medium · **Confidence:** suspected

```python
114:        out = executor.invoke({"input": prompt})
115:        answer = out.get("output", "") if isinstance(out, dict) else str(out)
116:        if not steps:  # agent answered without a tool call -> fall back to direct routing
```

**What breaks:** In langchain 0.3 `AgentExecutor`, `early_stopping_method="force"` (the default) returns `{"output": "Agent stopped due to iteration limit or time limit."}` when `max_iterations=8` is exhausted. Because `steps` is non-empty by then, that string becomes `SubCall.answer` and, at the root, `AgenticResult.answer`, even though the sub-answers exist in `steps`. The limit is easy to hit because each `route_and_answer` can recurse (RA-17).
**Evidence:** `langchain` is not installed in `.venv` (only langchain-core), so this could not be run. It relies on known AgentExecutor behaviour, and no test runs the executor. Maintainer question: confirm the stop output in the pinned langchain version.
**Direction:** Detect a stopped run (`intermediate_steps` length or output sentinel) and fall back to `agent.synthesizer(prompt, [(s.prompt, s.answer) for s in steps])`.

### RA-10: `langchain>=0.3` has no upper bound; `AgentExecutor` / `create_tool_calling_agent` are gone from `langchain.agents` in 1.x `:98-100`
**Category:** correctness · **Severity:** medium · **Confidence:** suspected

```python
99:        from langchain.agents import AgentExecutor, create_tool_calling_agent
```

**What breaks:** `pyproject.toml:37` declares `agentic = ["langchain>=0.3", "langchain-core>=0.3"]`. The dev venv already has langchain-core 1.6.2, so a fresh `pip install .[agentic]` resolves langchain 1.x. There, the legacy executor moved to `langchain_classic`, and this import raises `ImportError` on the first orchestrated request. Tool building (`langchain_core.tools`) still works, so `test_langchain_orchestrator_tools_build_and_recurse` passes while `run` is broken.
**Evidence:** `pyproject.toml:33,37`. Installed langchain-core is 1.6.2 (checked in `.venv`). The locally reported langchain 0.3.17 is outside `.venv`. Suspected because it depends on which langchain version gets resolved.
**Direction:** Pin `langchain>=0.3,<1`, or port to `langchain.agents.create_agent` (1.x) with a version guard, and add a smoke test that constructs the executor.

### RA-13: Collapse paths re-route a prompt whose route was already computed `:32-36`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
34:        if len(subs) <= 1:
35:            sc = agent._answer_directly(prompt, depth)
36:            return sc.answer, [sc]
```

**What breaks:** `_answer_directly` without `model_id` calls `route_decision` again (`router.py:149-151`), which for `NIRTRouter` is another encoder forward pass plus a per-model scoring loop. It is the identical computation `triage_prompt` just did. This hits every "weak but indivisible" prompt, and every single-part prompt flagged by RA-12. The same happens in the LangChain no-tool fallback (`:116-118`).
**Evidence:** `_solve` (`router.py:133`) calls `orchestrator.run(prompt, self, depth)` without passing the `decision`, so its model id is unavailable to the orchestrator.
**Direction:** Pass `decision` (or its `selected_model_id` / `predicted_quality`) into `orchestrator.run` and reuse it on collapse.

### RA-17: Fan-out is bounded only by depth; no global cap on calls or spend `:32-39`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
37:        children = [agent._solve(s, depth + 1) for s in subs]
```

**What breaks:** Recursion does terminate (`depth >= max_depth` at `router.py:127`), but the work grows geometrically: `NaiveDecomposer.max_parts=6` gives up to 6^max_depth leaf model calls, each with a route and a triage. `LLMDecomposer.max_parts=8`. In the LangChain path, each nested `route_and_answer` starts a fresh `AgentExecutor` with its own `max_iterations=8`, so up to 8^max_depth agent LLM calls. There is no per-run call counter, deadline, or budget. Combined with RA-03 (cost always 0), a runaway request is invisible.
**Evidence:** Traced `_solve -> RecursiveOrchestrator.run -> _solve(depth+1)` and `route_and_answer -> _solve(depth+1) -> LangChainToolOrchestrator.run`. No shared counter crosses these calls.
**Direction:** Thread a per-run ledger (max calls / max cost / deadline) through `_solve` and have the tools refuse when it is exhausted.

---

## src/router/agentic/decompose.py

### RA-07: `re.IGNORECASE` makes the "capital letter" lookahead match any letter, splitting mid-sentence `:28-37`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
32:    r"|[?.]\s+(?=[A-Z])"                 # sentence boundary before a capital
36:    re.IGNORECASE,
```

**What breaks:** With `IGNORECASE`, `(?=[A-Z])` also matches lowercase, so every `". x"` becomes a split point, including abbreviations. Verified: `"Use numpy, e.g. arrays, to compute the mean of a list of numbers."` → `['Use numpy, e.g.', 'arrays, to compute the mean of a list of numbers.']`. Both fragments are then routed and answered independently and concatenated. Any prompt containing "e.g. ", "i.e. ", "vs. ", "Dr. " or "U.S. " is torn apart once triage flags it (for example when it is longer than 600 chars).
**Evidence:** Ran `NaiveDecomposer()` on the string above in `.venv`. The tests only cover `and then` and a single `?` question.
**Direction:** Drop `IGNORECASE` for that alternative (scoped `(?-i:...)`, or compile the linking phrases separately), and skip common abbreviations.

### RA-08: Silent truncation drops parts of the request and sub-answers `:74-80`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
76:            msg = self._chat.invoke(self._PROMPT.format(prompt=prompt[:4000]))
```

**What breaks:** `LLMDecomposer` sends only the first 4000 chars, so requirements past that point never become sub-tasks and are never answered. Nothing signals this, and `HeuristicTriage` sends every prompt longer than 600 chars to decomposition. `LLMSynthesizer` (`:113-114`) truncates the joined sub-answers to `body[:8000]`, so with a few long sub-answers the later ones are cut off before synthesis and the final answer omits them. `LLMTriage` truncates to 2000 (`triage.py:105`), which is less harmful there.
**Evidence:** Direct string slicing, with no length check or warning. No test covers the LLM classes.
**Direction:** Budget by tokens, and when content doesn't fit, chunk it (decompose in windows / map-reduce synthesis) or fall back to `concat_synthesizer` rather than slicing.

### RA-21: `LLMDecomposer` parsing keeps numbering and preamble lines as sub-tasks `:77-80`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
78:            parts = [ln.strip(" \t-*.").strip() for ln in text.splitlines() if ln.strip()]
79:            parts = [p for p in parts if len(p) >= 4][: self.max_parts]
```

**What breaks:** `strip(" \t-*.")` does not remove `1.`/`2)` prefixes or Markdown headers. More importantly, a chat preamble like "Here are the sub-tasks:" becomes a sub-task that is routed, answered, and paid for, and it uses one of the `max_parts` slots, which can push the last real sub-task out. It also strips the trailing `.` from a single-line echo, so that line no longer equals `prompt`. This is harmless in `RecursiveOrchestrator`, which checks `len(subs) <= 1` rather than equality.
**Evidence:** The prompt asks for "no numbering or commentary", but models often ignore that. Suspected because it depends on the model.
**Direction:** Ask for JSON list output (or structured output) and parse it, dropping lines that end with `:`.

---

## src/router/agentic/result.py

### RA-16: `models_used()` lists the triage-selected model in orchestrated mode even though it was never called `:42-50`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
48:        if self.selected_model_id and self.selected_model_id not in seen:
49:            seen.insert(0, self.selected_model_id)
```

**What breaks:** `run` sets `selected_model_id=decision.selected_model_id` for orchestrated results (`router.py:113`), but that model only picked the triage signal and may not be invoked for any sub-task. `models_used()` and `summary()` still report it first, so usage and attribution analytics over-count it.
**Evidence:** In `test_compound_prompt_is_orchestrated_and_synthesised` the assertion `set(models_used()) <= set(POOL)` passes regardless.
**Direction:** Only walk `steps`, or insert `selected_model_id` only when `mode == "single"`.

### RA-20: `summary()` raises `TypeError` when `predicted_quality` is `None` `:52-55`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
55:                    f"(q~{self.predicted_quality:.2f}): {self.triage_reason}")
```

**What breaks:** `predicted_quality: Optional[float] = None` is the dataclass default, and `format(None, ".2f")` raises. `AgenticRouter.run` always passes a float, so this only affects results built elsewhere (custom orchestrators or tests).
**Evidence:** Grep for `AgenticResult(` in src/tests finds only `router.py:103,111`, which is why this is suspected (latent).
**Direction:** Guard the format for `None`.

---

## src/router/agentic/__init__.py

No findings.

---

## Checked and ruled out
- **Infinite recursion:** `_solve` always hits `depth >= self.max_depth` (`router.py:127`), and both orchestrators recurse only with `depth + 1`, including the LangChain `route_and_answer` tool. A decomposer echoing the prompt as a sub-task just burns one level. Termination holds, though the fan-out is unbounded (RA-17).
- **Echo default synthesizing a client per id (unbounded registry growth):** bounded by the set of distinct ids requested, which is fine.
- **`_shallow_copy` sharing `clients`/`router` between copies:** intentional, since the cache is keyed by id and read-mostly.
- **`best_from_result` on a multi-row result:** only called with 1-prompt routes (`router.py:81,87`).
- **`NaiveDecomposer` numbered lists with a leading `\s*` consuming the `\n`:** backtracking handles it. `"1. A\n2. B\n3. C"` splits correctly (verified).
- **`concat_synthesizer` with one pair:** returns the raw answer, and callers only use it with 2 or more children.
- **`AgenticRouter.__init__` `orchestrator or RecursiveOrchestrator()` truthiness:** orchestrator classes define no `__len__`/`__bool__`.
- **Dead code:** grep over `src scripts tests configs docs` for `LLMTriage|LLMSynthesizer|LLMDecomposer|CallableClient|guess_provider|build_router_tools` shows each is exported in `router/agentic/__init__.py` and documented in `docs/agentic_router.md`, so none is unused.
