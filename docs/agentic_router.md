# Agentic router (`router.agentic`)

A trained [`Router`](routing_interface.md) decides *which model* should answer a
query. `AgenticRouter` puts an orchestrator around it: for a live prompt it
routes, **triages** on the router's own predicted-quality signal, and then either
answers directly with the routed model or decomposes the request and routes each
sub-task (recursively, with the router exposed as a tool).

```
prompt ──▶ route_decision (encoder ▸ Router)          best model + q*
        │
        ├─▶ triage(q*, prompt)  ──▶ "single"     ──▶ client(best_model).invoke(prompt)
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

## LLM clients — calling the routed model

`router/agentic/llm_clients.py`. A `Router` yields a `model_id`; a client turns it
into a callable LLM.

| client | use |
|---|---|
| `EchoClient` | deterministic, no network — **the default**, and what tests use |
| `CallableClient(fn)` | wrap any `str -> str` |
| `LangChainClient(model_id, model=…, provider=…)` | a LangChain chat model (instance or `init_chat_model` spec); provider guessed from the id |

`ClientRegistry` maps `model_id -> client` and synthesises a default
(`EchoClient`) for unseen ids, so an unfamiliar pool never crashes the run.
`ClientRegistry.langchain(pool)` serves every id through `LangChainClient`.

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

`AgenticRouter(router, ...)` knobs: `clients`, `triage`, `orchestrator`,
`decomposer`, `synthesizer`, `encoder`, `lam` (cost weight for routing),
`max_depth` (recursion cap, default 2), `query_resolver` (map text → an existing
`query_id` when the router can't score raw text, e.g. `MatrixRouter`).

## Live-text routing

`AgenticRouter` needs the router to score raw prompt text. `NIRTRouter` and
`KNNRouter` implement `predict_scores_text` (`can_route_text = True`) — they embed
the prompt with the project's encoder (`router.embeddings`) for their pathway and
score directly (NIRT: forward pass per model; kNN: nearest train queries). For a
router without a text path (`MatrixRouter`, `RandomRouter`), pass
`query_resolver=`.

## Dependencies

The core path (echo clients, `RecursiveOrchestrator`, `HeuristicTriage`,
`NaiveDecomposer`) needs nothing beyond the base install. LangChain is the
optional `agentic` extra: `pip install -e ".[agentic]"`. Provider SDKs
(`openai`, `anthropic`, …) are pulled by LangChain / the `labeling` extra as
needed.
