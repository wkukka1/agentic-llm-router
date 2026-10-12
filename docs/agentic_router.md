# Agentic router (`router.agentic`)

A trained [`RouterModel`](routing_interface.md) decides *which model* should answer a
query. `AgenticRouter` puts an orchestrator around it: for a live prompt it
routes, **triages** on the router's own predicted-quality signal, and then either
answers directly with the routed model or decomposes the request and routes each
sub-task (recursively, with the router exposed as a tool).

```
prompt ──▶ route_decision (encoder ▸ RouterModel)     best model + q*
        │
        ├─▶ triage(q*, prompt)  ──▶ "single"     ──▶ client(best_model).complete(prompt)
        │                        └▶ "orchestrate" ──▶ orchestrator
        │                                              ├ decompose → sub-prompts
        │                                              ├ _solve(sub) ── recurse (re-triage)
        │                                              └ synthesize → final answer
        ▼
   AgenticResult(mode, answer, steps[SubCall], models_used, cost, …)
```

## Triage — the single-vs-orchestrate decision

The decision is driven by the router, not a separate classifier
(`router/agentic/triage.py`):

- **`q*` (best predicted quality) high** → the pool has a model that handles this
  → **single shot**.
- **`q*` low** → even the best candidate is predicted to struggle → **decompose**
  so each part can be routed to a strong model.
- a lexical multi-part check (`"and then"`, lists, `≥3` "?", length) also forces
  decomposition.

`HeuristicTriage(quality_threshold=0.55)` is the default. `LLMTriage(chat_model)`
asks a model for `SINGLE`/`DECOMPOSE` (router scores passed as context), falling
back to the heuristic on any failure. If triage says decompose but the request
can't be split, it collapses back to a single routed call.

`LLMTriage(chat_model, margin=0.1)` gates that extra LLM round-trip: the
heuristic decides alone (no call, nothing spent against `max_calls`) when the
prompt is lexically multi-part or `|q* - quality_threshold| > margin`; the LLM
is asked only in the ambiguous band or when there is no usable score. The
default `margin=inf` always asks.

## LLM clients — calling the routed model

`router/agentic/llm_clients.py`. A `RouterModel` yields a `model_id`; a client
(`router.llm.client.LLMClient`, wrapping a `ProviderAdapter`) turns it into a
callable LLM.

| adapter (`router.llm.adapters`) | use |
|---|---|
| `EchoAdapter` | deterministic, no network — **the default**, and what tests use |
| `CallableAdapter(model, fn)` | wrap any `str -> str` |
| `LangChainAdapter(model, provider=…)` | a LangChain chat model (instance or `init_chat_model` spec); provider guessed from the id |
| `AnthropicAdapter(model, max_tokens=…)` | Claude via the official `anthropic` SDK: `complete` + `stream`, prompt caching on by default, cache-aware pricing when given per-token prices, a refusal raises |

`ClientRegistry` maps `model_id -> LLMClient` and synthesises a default
(`LLMClient(EchoAdapter(...))`) for unseen ids, so an unfamiliar pool never
crashes the run. `ClientRegistry.langchain(pool)` serves every id through
`LangChainAdapter`. `ClientRegistry.anthropic(serve_map, pool=...)` serves
every pool id through Claude per [`configs/serve_anthropic.yaml`](../configs/serve_anthropic.yaml):
pool id → capability tier → Claude model (with prices). The router's pool is
mostly non-Claude models, so that map is a rough tiering for exercising real
latency / cost / caching, not a basis for quality claims. (`OpenAIAdapter` and
`GoogleAdapter` are still stubs; reach those providers through `LangChainAdapter`.)

## Orchestrators

`router/agentic/orchestrator.py`, both `run(prompt, agent, depth) -> (answer, [SubCall])`:

- **`RecursiveOrchestrator`** (default) — plain control flow: `decompose` →
  `agent._solve` each sub-task (which re-triages, so nesting is possible up to
  `max_depth`) → `synthesize`. Deterministic, no agent framework, fully tested.
- **`LangChainToolOrchestrator(llm)`** — exposes the router + clients as
  LangChain `StructuredTool`s and runs a tool-calling agent that decides the
  decomposition itself:
  | tool | effect |
  |---|---|
  | `route_query(q)` | inspect the router's choice (model + predicted quality) |
  | `answer_with_model(q, model_id)` | dispatch `q` to a specific client |
  | `route_and_answer(q)` | route `q` to the best model, **recursively decomposing** if still complex |

- **plan orchestrator** — `router.execution.routed.plan_orchestrator(budget=…)`
  (a `PlanOrchestratorBridge` over `RoutedOrchestrator`): decompose, route
  **all** sub-prompts in one batched router call, forecast the plan's cost /
  quality / latency (`RouterPlanForecaster`), reserve that cost on a
  per-request `BudgetLedger` (refused up front if it doesn't fit), run
  independent steps **in parallel**, and replan a failed step onto its
  next-best model. Steps are leaves (not re-triaged). Calls still go through
  the agent, so `max_calls` and conversation history apply.

  **Adaptive plans.** The planner may mark a step `[after N]`; such a step
  receives the successful outputs of the steps it depends on. With
  `route_late`, each step is routed again on its final prompt (plus dependency
  results): `SubCall.meta["planned_model"]` is the upfront choice and
  `meta["routed_model"]` the one used. A failed step escalates to the next-best
  model (`meta["tried_models"]`, `meta["escalation"]` = `escalated` /
  `exhausted`) and never retries a model already tried. With
  `plan.quality_check.enabled`, a judge scores the synthesized answer; on a
  failure a repair round escalates the weak steps, re-runs their dependents and
  adds the checker's follow-ups as steps (capped by `limits.max_fanout`;
  unscorable follow-ups are dropped). `max_replans` is the number of repair
  rounds. `AgenticResult.quality_check` holds the best round's verdict plus
  `best_round` and `rounds`; the best answer is returned, flagged if no round
  passed. Superseded paid attempts stay in `steps` with
  `meta["superseded"]=True`, so `cost` / `models_used()` count all spend.

Decomposition / synthesis helpers (`router/agentic/decompose.py`):
`NaiveDecomposer` + `concat_synthesizer` (dependency-free defaults),
`LLMDecomposer` / `LLMSynthesizer` (chat-model backed).

## Usage

```python
from router.routing import NIRTRouter
from router.agentic import AgenticRouter, ClientRegistry

# offline / research: echo clients, heuristic triage, recursive orchestrator
ar = AgenticRouter(NIRTRouter.from_run("nirt-2d-projected"))
ar.run("What is the capital of France?").mode           # "single"
r = ar.run("Explain quicksort, implement it in Rust, and analyse its complexity.")
r.mode                                                  # "orchestrated"
r.summary(); r.models_used(); r.answer

# production: real models + LLM-driven orchestration
from langchain.chat_models import init_chat_model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
ar = AgenticRouter(
    NIRTRouter.from_run("nirt-2d-projected"),
    clients=ClientRegistry.langchain(["gpt-4o", "claude-3-5-sonnet-latest", "gpt-4o-mini"]),
    triage=LLMTriage(llm),
).with_langchain_orchestrator(llm)
ar.run(user_prompt)
```

### Multi-turn, streaming, warm-up

```python
ar.warmup()                       # at startup: encoder, stores, cost vector -> no cold first request

msgs = [{"role": "user", "content": "I am planning a trip to Japan."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "Which cities should I visit?"}]
r = ar.run(msgs, conversation_id="c1")   # also: a decompose.conversation.Conversation
r.turn_index, r.timings                  # 2, {"route": ..., "triage": ..., "dispatch": ..., "total": ...}

gen = ar.stream(msgs, on_result=log)     # yields text chunks; StopIteration.value is the AgenticResult
```

- **Conversations:** the router scores *all user turns* joined with blank lines —
  exactly how the training data rendered Arena conversations — while triage and
  decomposition judge only the *latest* turn, and every routed call is sent the
  earlier messages as history (`router/agentic/conversation.py`). A bare string
  is a one-turn conversation with no history, so single-prompt behaviour is
  unchanged.
- **Recording:** `AgenticRouter(..., result_sink=router.tracing.run_store.sqlite_sink())`
  writes every request's prompt, history, answer, model calls and expected vs.
  actual cost to SQLite (`data/runs/agentic_runs.sqlite`), off the request
  path; see the runbook's "Recording runs" section.
- **Streaming:** routing and triage happen before the first chunk; pricing,
  `SubCall`/`AgenticResult` assembly and `on_result` (on a background thread)
  happen after the stream ends. An orchestrated request yields its synthesized
  answer as one chunk.

`AgenticRouter(router, ...)` knobs: `clients`, `triage`, `orchestrator`,
`decomposer`, `synthesizer`, `encoder`, `lam` (cost weight for routing),
`max_depth` (recursion cap, default 2), `query_resolver` (map text → an existing
`query_id` when the router can't score raw text, e.g. `MatrixRouter`).

## Live-text routing

`AgenticRouter` needs the router to score raw prompt text. `NIRTRouter` and
`KNNRouter` implement `predict_scores_text` (`can_route_text = True`) — they embed
the prompt with the project's encoder (`router.embeddings`) for their pathway and
score directly (NIRT: one batched forward pass over the query × pool grid; kNN:
nearest train queries). The live router loads a `cfg`-only view of the training
data (`load_training_data(cfg, serving=True)`), never the response tables. For a
router without a text path (`MatrixRouter`, `RandomRouter`), pass
`query_resolver=`.

## Dependencies

The core path (echo clients, `RecursiveOrchestrator`, `HeuristicTriage`,
`NaiveDecomposer`) needs nothing beyond the base install. LangChain is the
optional `agentic` extra: `pip install -e ".[agentic]"`. `AnthropicAdapter`
needs the `anthropic` extra (`pip install -e ".[anthropic]"`). Other provider
SDKs are pulled by LangChain / the `labeling` extra as needed.

Latency measurements and how to reproduce them: [`agentic_latency.md`](agentic_latency.md). Setup, running and troubleshooting: [`agentic_router_runbook.md`](agentic_router_runbook.md).
