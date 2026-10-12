# Agentic router latency (2026-10-02)

How long `AgenticRouter.run` takes from request to routing decision to answer,
what changed to bring it down, and how to re-measure it. All numbers here use
**echo clients** (no network), so they measure the router's own overhead. The
time a real model takes to start answering is not measured yet (see
[Not measured yet](#not-measured-yet)).

## Setup

- Router: `NIRTRouter.from_run("nirt-2d-projected")` (BERT `irt` query
  pathway, 29-model pool), on an RTX 3050 Ti laptop GPU.
- Data: [`scripts/agentic/build_multiturn_set.py`](../scripts/agentic/build_multiturn_set.py)
  takes 200 conversations of 2 or more turns from Chatbot Arena 55k, all from
  the **held-out test split** (the router was trained on Arena). They are
  stratified by length: 68 with 2 turns, 66 with 3, 66 with 4 or more; 720
  turns in total. Turn k is sent with the real turns 1..k-1 in front of it.
- Bench: [`scripts/agentic/latency_bench.py`](../scripts/agentic/latency_bench.py).
  Raw per-turn rows are in `investigations/agentic_latency/*.json`.

## Where the time went (before)

| Stage | Cost | Cause |
|---|---|---|
| First request | **37.3 s** | The router lazily read ~650 MB of training tables (`responses`, `queries`) only to get a config object; then it loaded BERT. With a warm disk cache this is about 5.8 s + 8.4 s. |
| Every request: scoring | 3.4 ms | One NIRT forward pass per pool model (29 passes) |
| Every request: RNG reseed | 0.5 ms | `TextEncoder.encode` reseeded the global RNGs, including CUDA's, on every call (TEXTENCODER-001) |
| Orchestrated request | up to ~110 ms extra at p95 | Sub-prompts encoded and routed one at a time |
| `LLMTriage` (when used) | one extra LLM round-trip | It was asked on every request, even clear-cut ones |

## What changed

| Change | Where |
|---|---|
| The live router loads a config-only view of the training data (`load_training_data(cfg, serving=True)`) | `training/data/facade.py`, `router/routing/routers.py` |
| `warmup()` on `RouterModel` / `NIRTRouter` / `AgenticRouter` moves the encoder load, stores and cost vector to startup | same + `router/agentic/router.py` |
| Batched NIRT scoring (one pass over the query × pool grid) with a cached pool reference; scores match the old loop to 2e-7 | `routers.py` |
| No per-call reseed; the one-time seed at model load is scoped and restores the caller's RNG state | `router/embeddings/encoder.py` |
| `LLMTriage(margin=…)` asks the LLM only in the ambiguous band | `router/agentic/triage.py` |
| `AgenticRouter.stream()`: the answer streams; pricing, result assembly and `on_result` run after the last chunk | `router/agentic/router.py`, `router/llm/client.py` |
| Plan orchestrator: all sub-prompts routed in one batch, independent steps run in parallel | `router/execution/routed.py` |

## Results

Warm-path numbers are milliseconds per turn. The laptop GPU's clocks vary
between runs (the same configuration measured 13 ms and 28 ms routing p50 in
consecutive runs), so the comparisons that matter were run back to back.

**Cold start** (30 conversations, joined-text input):

| | Before | After |
|---|---|---|
| First routing decision | 37.3 s | **0.012 s** |
| Startup `warmup()` | — | 8.5 s (almost all BERT load) |
| Warm routing p50 / p95 | 16.3 / 28.4 | **11.8 / 19.4** |
| Full `run` p50 / p95 | 18.5 / 136.3 | **12.6 / 88.0** |

**Orchestrators**, back to back, 60 conversations (204 turns, 13 orchestrated), message input:

| | Routing p50 | `run` p50 / p95 | Orchestrated dispatch p95 |
|---|---|---|---|
| `RecursiveOrchestrator` | 13.1 | 13.4 / 58.8 | 40.3 |
| plan orchestrator | 13.9 | 14.1 / **30.9** | **17.8** |
| plan orchestrator, `stream()` | 12.3 | 12.5 / 32.1 | 17.2 |

The p95 gain comes from orchestrated turns: the plan orchestrator scores all
sub-prompts in one encoder pass instead of one pass each. With echo clients, a
streamed answer's time to first token is just routing plus triage (12.5 ms
p50). With a real model it adds that model's own time to start answering.

**Behaviour change on conversations.** Triage now judges only the latest user
turn. On the 720 Arena turns, orchestration dropped from about a third of turns
(when the whole joined conversation was triaged) to **7%** (52 of 720). A
multi-part *earlier* turn no longer forces the current, simple turn to be
decomposed. Routing still scores every user turn, matching training.

## Reproduce

```bash
python scripts/agentic/build_multiturn_set.py            # -> data/eval/arena_multiturn_test.jsonl (git-ignored)
python scripts/agentic/latency_bench.py --limit 60 --orchestrator recursive
python scripts/agentic/latency_bench.py --limit 60 --orchestrator plan
python scripts/agentic/latency_bench.py --limit 60 --orchestrator plan --stream
```

## Not measured yet

- **A real model's time to first token, and cache hits by turn.**
  `latency_bench.py --anthropic --stream` serves the pool through Claude via
  [`configs/serve_anthropic.yaml`](../configs/serve_anthropic.yaml) and reports
  `cache_read_share_by_turn`. It has not been run; it costs real money. The
  Anthropic adapter is so far tested only against a mocked SDK.
- **Answer quality.** The serve map's tiers are a rough grouping, so a live run
  measures latency, cost and caching, not whether the router picks well.
- **`ModelScore.expected_latency`** is still always 0 in `router.router.Router`
  (RC-01). Nothing feeds observed latency back into routing yet.
