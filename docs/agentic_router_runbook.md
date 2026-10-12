# Agentic router runbook

How to set up, start, run and troubleshoot `router.agentic.AgenticRouter`. For
how it works, see [`agentic_router.md`](agentic_router.md). For latency numbers,
see [`agentic_latency.md`](agentic_latency.md).

`AgenticRouter` takes a prompt or a conversation and works in three steps:

1. **Route.** A trained NIRT router scores every model in the pool.
2. **Triage.** It decides whether the best model can answer in one call, or
   whether the request should be split into sub-tasks.
3. **Answer.** It calls the chosen model, or runs an orchestrator that answers
   each sub-task and combines the results.

Model calls go to **echo clients** unless you configure real ones. Echo clients
need no network or API key and return a placeholder, which is how you run it
offline.

All commands are run from the repo root.

---

## 1. One-time setup

### 1.1 Install

```bash
python -m venv .venv
.venv/Scripts/activate                      # Windows; source .venv/bin/activate on POSIX
pip install -e ".[dev]"                     # router, torch, transformers, pytest
pip install -e ".[anthropic]"               # only to call Claude
pip install -e ".[agentic]"                 # only for LangChain clients / LLMTriage / LangChainToolOrchestrator
```

Python 3.11 or newer is required.

### 1.2 Use the GPU build of torch

The router embeds every prompt with BERT. On CPU that is about 11× slower.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If this prints `False` on a machine with an NVIDIA GPU, pip installed the CPU
wheel. Replace it with the CUDA build (`cu130` matches the dev machine's driver):

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps torch==2.13.0
```

`configs/phase0.yaml` has `embedding.device: auto`, which uses `cuda`
automatically once the CUDA build is present.

### 1.3 Files the router needs

The default run is `nirt-2d-projected` (BERT `irt` pathway, 29-model pool). It
needs these under `data/processed/`:

| File | What it is | How to build it if missing |
|---|---|---|
| `nirt_runs/nirt-2d-projected/model.pt` (+ `run.json`) | the trained router | `python scripts/nirt/train_nirt.py --dim 2 --model-params projected --name nirt-2d-projected` |
| `embeddings/model_profile__irt/` | model profile embeddings (projected runs score against these) | `python scripts/embeddings/build_profile_embeddings.py --pathway irt` |
| `nirt_observations.parquet` | per-model cost vector used in routing | `python scripts/data/build_nirt_dataset.py` |

`data/processed/` is git-ignored, so a fresh clone has none of these. Building
them from scratch takes the data pipeline first (`python -m training.phase0`;
see the README). `bert-base-uncased` downloads from Hugging Face on first use.
The "unauthenticated requests" warning is harmless; set `HF_TOKEN` to silence
it.

### 1.4 Credentials (only for real models)

- **Claude:** the SDK reads `ANTHROPIC_API_KEY`, or a profile from `ant auth login`.
- **LangChain-served models:** set each provider's own key (for example `OPENAI_API_KEY`).

---

## 2. Preflight check

Run this before relying on a new environment. It loads the router, warms it up
and answers one prompt with echo clients. There are no network calls.

```bash
python - <<'EOF'
import time, torch
from router.routing import NIRTRouter
from router.agentic import AgenticRouter

print("cuda:", torch.cuda.is_available())
t = time.perf_counter(); router = NIRTRouter.from_run("nirt-2d-projected")
print(f"router loaded in {time.perf_counter() - t:.1f}s, pool of {len(router.model_ids)}")
agent = AgenticRouter(router)
print("warmup:", {k: round(v, 2) for k, v in agent.warmup().items()})
res = agent.run("What is the capital of France?")
print(res.summary()); print("timings (s):", {k: round(v, 4) for k, v in res.timings.items()})
EOF
```

Expected output:

- `cuda: True`
- `warmup` total around 8–20 s, almost all of it `encoder`
- `single -> <model> ...`
- `route` around 0.01–0.03 s

If any of these differ, see [Troubleshooting](#8-troubleshooting).

---

## 3. Starting it in a service

Build **one** `AgenticRouter` at startup, call `warmup()` once, and reuse that
instance for every request.

```python
from router.routing import NIRTRouter
from router.agentic import AgenticRouter

agent = AgenticRouter(NIRTRouter.from_run("nirt-2d-projected"))   # ~2 s: loads the checkpoint
agent.warmup()                                                      # ~8 s: BERT, stores, cost vector
# ... then serve requests with agent.run(...) / agent.stream(...)
```

Or build everything from the settings file, [`configs/agentic.yaml`](../configs/agentic.yaml):

```python
agent = AgenticRouter.from_settings()                        # configs/agentic.yaml; warms up if warmup: true
agent = AgenticRouter.from_settings("configs/my_agentic.yaml")
```

The file is validated by the Pydantic models in
[`src/router/agentic/schemas.py`](../src/router/agentic/schemas.py)
(`AgenticRouterSettings`) before anything loads. It sets:

- the router run and the clients (`echo` / `anthropic` / `langchain`)
- `max_calls`, `max_depth` and `lam`
- triage, decomposer and synthesizer (each `heuristic`/`naive`/`concat`, or `llm` with a `model`)
- the orchestrator (`recursive` / `plan` / `tool_agent`) and its settings
- the result sink

A misspelled key, an out-of-range value, an `llm` helper without a `model`,
or a reference to a prompt that doesn't exist is an error at load time.

**Prompts** live in the same file, under `prompts:`, as `name: template`
(`str.format` templates). It is the only copy of every LLM prompt text
(triage, decompose, synthesize, the tool agent's system prompt). The file's
header lists the `{placeholders}` each prompt is rendered with. To use a
different prompt, add an entry and point the helper's `prompt:` at it; a name
that isn't defined is an error at load time.

- **Always warm up.** Without `warmup()`, the first request pays the BERT load
  (about 8 s; up to 37 s on a cold disk).
- **Don't build a router per request.** Every load is cached on the router
  instance, so a new `NIRTRouter` repeats the whole cold start.
- **One instance is safe to share.** `run()` and `stream()` can be called
  concurrently from threads or asyncio tasks; each call keeps its own state
  (call budget, history, timings).

---

## 4. Running requests

### 4.1 A single prompt

```python
res = agent.run("Explain ownership, and then write a borrow example.")
res.mode               # "single" or "orchestrated"
res.answer             # final answer (synthesized when orchestrated)
res.summary()          # one-line explanation of the decision
res.models_used()      # models actually called
res.cost               # USD; None when any call's cost is unknown
res.timings            # seconds: route, triage, dispatch, total
res.errors()           # failed sub-calls (their answers are placeholders)
```

### 4.2 A conversation (multi-turn)

Pass the whole history. The last message must be the user turn to answer.

```python
msgs = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "I am planning a trip to Japan in April."},
    {"role": "assistant", "content": "April is cherry blossom season, so book early."},
    {"role": "user", "content": "Which cities should I visit?"},
]
res = agent.run(msgs, conversation_id="c-123")
res.turn_index         # 2 (user turns so far)
```

- **Routing scores all user turns** joined with blank lines. This is how the
  router saw multi-turn conversations in training.
- **Triage and decomposition look only at the latest turn**, so an earlier
  multi-part request doesn't force the current one to be split.
- **The routed model receives the full message list.** Claude's prompt caching
  then reuses earlier turns.
- A `decompose.conversation.Conversation` object works too.

### 4.3 Streaming

```python
gen = agent.stream(msgs, on_result=lambda r: log.info(r.summary()))
try:
    while True:
        print(next(gen), end="", flush=True)    # text chunks as they arrive
except StopIteration as stop:
    res = stop.value                            # the AgenticResult
res.timings["ttft"]                             # request -> first chunk, seconds
```

- Routing and triage run before the first chunk.
- Pricing, building the result and `on_result` run after the last chunk. The
  `on_result` callback runs on a background thread, so it doesn't delay your
  caller.
- An orchestrated request is solved first, then its combined answer comes back
  as **one** chunk.

### 4.4 Calling Claude

```python
import yaml
from router.agentic import AgenticRouter, ClientRegistry

serve_map = yaml.safe_load(open("configs/serve_anthropic.yaml"))
clients = ClientRegistry.anthropic(serve_map, pool=router.model_ids)
agent = AgenticRouter(router, clients)
```

[`configs/serve_anthropic.yaml`](../configs/serve_anthropic.yaml) maps each
pool model to a tier (strong / mid / small), and each tier to a Claude model
with its prices. It also sets the default `max_tokens` (1024); a call can pass
its own.

The pool is mostly non-Claude models, so this mapping is rough. Use live runs
to measure latency, cost and caching, not to judge whether the router picks the
right model.

- **Prompt caching** is on by default. Short conversations fall under the
  model's minimum cacheable length and won't cache. That is expected, not an
  error.
- **Per-call details** are in `res.steps[i].meta`: `stop_reason`,
  `cache_read_input_tokens`, `request_id` and `latency`.
- **Live runs cost money.** Start with a few prompts.

To call one Claude model directly, without routing:

```python
from router.llm.client import LLMClient
from router.llm.adapters.anthropic import AnthropicAdapter
LLMClient(AnthropicAdapter("claude-haiku-4-5", max_tokens=64)).complete("Reply with: pong").content
```

To check that your credentials work with one cheap call, run the opt-in live
test: `ANTHROPIC_LIVE_TEST=1 pytest tests/router/llm/adapters/test_anthropic_adapter.py -k live`.

`OpenAIAdapter` and `GoogleAdapter` are not implemented. Reach those providers
through `ClientRegistry.langchain([...])` (needs the `agentic` extra).

### 4.5 Plan orchestrator, with a budget

Use this for requests that get split into sub-tasks, when you want a cost cap
and parallel sub-calls:

```python
from router.execution.routed import plan_orchestrator

agent = AgenticRouter(router, clients, orchestrator=plan_orchestrator(budget=0.05))  # USD per request
```

For each orchestrated request it:

1. Splits the request.
2. Routes every sub-prompt in one batch.
3. Forecasts the plan's cost and refuses it up front if it is over `budget`.
4. Runs independent sub-tasks in parallel.
5. Retries a failed sub-task on its next-best model.

**A refused plan** comes back as `res.errors()[0].error == "over_budget"`, with
nothing spent.

| Option | Default | Effect |
|---|---|---|
| `budget` | `inf` | USD cap per request |
| `max_workers` | 4 | sub-tasks run in parallel |
| `sequential` | `False` | `True` makes each step see the previous step's answer, and they run one by one |
| `limits` | from `budget` | an `ExecutionLimits`; `max_replans` (repair rounds after a failed step or quality check) defaults to 1 |
| `route_late` | `True` in `configs/agentic.yaml` | route each step on its prompt + dependency results |
| `planner_prompt` | `plan` | numbered steps with `[after N]` dependencies (LLM decomposers) |
| `quality_check.enabled` | `True` in `configs/agentic.yaml` | judge the synthesized answer; repair weak steps / add follow-ups |
| `quality_check.min_quality` | 0.8 | judge = cheapest pool model predicted >= this |
| `quality_check.pass_score` | 7 | PASS also needs SCORE >= this (0-10) |
| `quality_check.prompt` | `verify` | prompt name for the judge |
| `registry` | none | an `LLMRegistry`, for per-token prices in the forecast |

Without a `registry`, the forecast prices each step at the router's average
cost per query from its training data.

### 4.6 Caps and other options

| Option | Default | Effect |
|---|---|---|
| `max_calls=` | `None` | Caps **all** LLM calls in one request: routed calls, `LLMTriage`, `LLMDecomposer` and `LLMSynthesizer`. A request over the cap raises `RuntimeError("max_calls=... exhausted")`; inside an orchestration, the extra calls become failed sub-calls. |
| `max_depth=` | 2 | how deeply `RecursiveOrchestrator` may split sub-tasks again |
| `lam=` | 0 | how much cost counts against quality when picking a model (0 means quality only) |
| `triage=` | `HeuristicTriage()` | the single-vs-split decision; no LLM |
| `triage=LLMTriage(chat, margin=0.1)` | — | asks a chat model only when the heuristic is unsure. Without `margin`, every request pays an extra LLM round-trip. |
| `decomposer=` / `synthesizer=` | `NaiveDecomposer` / `concat_synthesizer` | `LLMDecomposer(chat)` / `LLMSynthesizer(chat)` for better splits and merges |

---

## 5. Recording runs (SQLite)

Nothing is saved unless you add a result sink. To record every request's
prompt, history, answer, model calls, and **expected vs. actual cost**:

```python
from router.tracing.run_store import sqlite_sink

agent = AgenticRouter(router, clients, result_sink=sqlite_sink(router="nirt-2d-projected"))
```

This writes to `data/runs/agentic_runs.sqlite` (git-ignored); pass a path to
use a different file. Both `run()` and `stream()` are recorded, on the
background bookkeeping thread, so recording never delays a request. A failing
sink is logged (logger `router.agentic.router`), never raised. From the
benchmark, add `--db data/runs/bench.sqlite`.

| Table / view | One row per | Holds |
|---|---|---|
| `runs` | request | prompt, `history_json`, answer, mode, triage reason, chosen model, stage timings, summed `expected_cost` / `actual_cost` |
| `calls` | node of the call tree (`is_leaf = 1` = a real model call) | prompt, routed `model_id`, serving `provider_model`, answer / error, `expected_cost` + `expected_cost_source`, `actual_cost`, tokens, cache reads, latency, stop reason, request id, plan `step_id` |
| `plans` | plan revision (plan orchestrator) | forecast cost / quality / latency vs. what that attempt spent; `outcome` = `completed` / `step_failed` / `over_budget` |
| `v_call_costs`, `v_run_costs`, `v_plan_costs` | — | expected, actual, `cost_error` (actual − expected), `cost_ratio` (actual ÷ expected) |
| `v_model_cost_accuracy` | routed model × serving model × estimate source | call count, total expected vs. actual, mean ratio, errors, mean latency |

How to read the expected cost (`expected_cost_source`):

- **`profile`:** the serving client's per-token prices × estimated tokens
  (characters ÷ 4 for input, a 256-token prior for output). This is the same
  basis as the actual cost, so `cost_ratio` tells you how good the estimate is.
- **`router_prior`:** the client has no prices, so this is the router's
  training-data average cost for the *pool* model. It is a coarse prior, and
  the actual cost is not expected to match it.
What `actual_cost` is (`calls.cost_source`):

- **`actual`:** the provider's token usage × the model's prices.
- **`estimate`:** the call ran, but its cost couldn't be priced (no prices
  configured, or the provider reported no usage). It is charged its expected
  cost instead, so totals and the budget ledger never count a real call as
  free. `runs.n_estimated` counts these per request.
- **`unknown`:** no actual cost and no estimate either. The run's total is
  then `NULL`, and the plan orchestrator commits 0 to its ledger with a
  warning.
- **`failed`:** the call errored, so nothing was billed.

The `cost_error` / `cost_ratio` columns use only `actual` rows, because an
estimate compared with itself would look perfectly accurate.

- **Echo clients** report an actual cost of 0, so the comparison only means
  something with real, priced clients (for example `ClientRegistry.anthropic`).

To look at the data:

```bash
python scripts/agentic/runs_report.py                                  # totals, by-model accuracy, latest runs and plans
python scripts/agentic/runs_report.py --sql "SELECT * FROM v_call_costs WHERE cost_ratio > 2"
```

A plan step that failed and was retried on another model is stored once, as
its final attempt; the failed attempt's spend is in that plan revision's
`actual_cost`.

## 6. Measuring latency

```bash
python scripts/agentic/build_multiturn_set.py                     # once: 200 held-out Arena conversations -> data/eval/
python scripts/agentic/latency_bench.py --limit 60                # echo clients, plan orchestrator
python scripts/agentic/latency_bench.py --limit 60 --stream
python scripts/agentic/latency_bench.py --limit 5 --anthropic --stream    # live Claude: costs money
```

The benchmark prints p50 and p95 per stage, the cold vs. warm stages, and the
`single`/`orchestrated` split. Live runs also show the cache-read share by
turn. Add `--json out.json` to keep per-turn rows.

Laptop GPUs vary from run to run, so only compare runs made **back to back**.
Reference numbers are in [`agentic_latency.md`](agentic_latency.md).

---

## 7. Verifying a change

```bash
pytest tests/router -q                  # ~45 s; agentic, llm, execution, routing
pytest tests -q                         # ~4 min; full suite
lint-imports                            # the router must not import training/evaluation (one allowlisted crossing)
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| First request takes 10–40 s | No warm-up, or a new router per request | Call `agent.warmup()` at startup and reuse one instance (§3) |
| Warm routing far above ~15 ms; `cuda: False` | CPU torch wheel | Reinstall the CUDA build (§1.2) |
| `FileNotFoundError: NIRTRouter: embeddings not built for pathway 'irt'` / `profile embeddings ... not built` | Profile store missing | `python scripts/embeddings/build_profile_embeddings.py --pathway irt` |
| `FileNotFoundError: ...nirt_observations.parquet not found, and this is a serving-mode TrainingData` | Cost table missing (the router does not rebuild it from raw data) | `python scripts/data/build_nirt_dataset.py` |
| `FileNotFoundError` on `nirt_runs/<run>/model.pt` | Run not trained, or wrong name | `ls data/processed/nirt_runs`, then use an existing name or train it (§1.3) |
| `TypeError: ... cannot score raw text; pass query_resolver=` | Router can't score text (`MatrixRouter`, `RandomRouter`) | Use `NIRTRouter` / `KNNRouter`, or pass `query_resolver=` |
| `LookupError: ... no usable score for this prompt` | Router produced nothing usable for this prompt | Check the run and its stores; the agent won't fall back to a default model on purpose |
| `ValueError: the last message must be a non-empty user turn` | Conversation ends with an assistant/system message, or an empty user turn | End the message list with the user turn to answer |
| `RuntimeError: max_calls=N model calls exhausted` | The request hit its call cap | Raise `max_calls`, gate `LLMTriage` with `margin`, or use cheaper decompose/synthesize |
| `res.errors()` shows `over_budget` | The plan orchestrator's forecast was over `budget` | Raise `budget`, or check the forecast with a `registry` that has real prices |
| `ValueError: AnthropicAdapter needs max_tokens` | No `max_tokens` from the call or the serve map | Set `max_tokens` in `serve_anthropic.yaml` or per call |
| `KeyError: pool model 'x' has no tier in the serve map` | Pool id not mapped and `default_tier` is null | Add it under `pool:` or set `default_tier` |
| `RuntimeError: <model> refused the request` | Claude declined (`stop_reason: refusal`) | Expected for some content; inside an orchestration it becomes a failed sub-call |
| `RuntimeError: <model> returned no answer (empty response)` | The adapter call succeeded but came back with blank content | Treated as a failure (raises at the root, a failed sub-call inside an orchestration) -- a blank response is never read as success |
| Answers look like `[answer::model] ...` | Echo clients (the default) | Pass real `clients=` (§4.4) |
| `LookupError: no client registered for model ...; refusing to answer with an echo placeholder` | Real clients given, but not for every pool id | Register every id in `router.model_ids`, or use `ClientRegistry.anthropic` / `.langchain` |
| No rows in `data/runs/agentic_runs.sqlite` | No `result_sink=`, or the process exited before the background write ran | Pass `result_sink=sqlite_sink()`; on shutdown, writes already queued finish first if the process exits normally |
| Cache reads stay 0 on Claude | Prompt below the model's minimum cacheable length (512–4096 tokens) | Expected for short chats; longer conversations cache from turn 2 |

---

## 9. Known limitations

- The real time to first token with Claude has not been measured yet; all
  reference numbers use echo clients.
- `router.router.Router` (the decision-only pipeline) doesn't use latency yet:
  `max_latency` and `latency_weight` have no effect and only produce a warning
  (RC-01).
- The plan orchestrator treats sub-tasks as final: they are not re-triaged or
  split again. Use `RecursiveOrchestrator` for nested decomposition.
- `RecursiveOrchestrator` (the default) still runs sub-tasks one at a time.
  Use `plan_orchestrator()` for parallel sub-tasks.
- The default `NaiveDecomposer` splits only on clear markers (`" and then "`,
  `"; then"`, numbered lists, and so on). `", then"` does **not** split
  (NAIVEDECOMPOSER-002, open). A prompt triage flags for splitting that can't
  be split is answered by a single model; `res.summary()` then ends with
  "nothing to decompose, answered directly". Use `LLMDecomposer` for real
  splitting.
- `HeuristicTriage`'s `quality_threshold` (0.55) isn't tuned per router run. For
  `nirt-2d-projected`, the best predicted quality on held-out Arena
  conversations has a median of 0.80, and 16.5% fall below 0.55.
