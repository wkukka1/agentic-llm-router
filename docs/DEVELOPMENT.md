# Developing agentic-llm-router

The research code behind the gateway: data pipeline, NIRT training, the
agentic router, evaluation. For using the gateway with Claude Code, see the
[README](../README.md).

Research code for a **cost-aware LLM router**. Each candidate LLM is a
test-taker with a latent ability vector, each query is an item with
difficulty / discrimination / relevance, and observed benchmark performance
is the response data (NIRT / IRT-Router style). A trained router scores
`P(model answers correctly)` per `(query, model)` pair and picks a model
under a cost budget; an agentic layer on top can triage a query into a
single routed call or decompose it into sub-tasks routed independently.

```
query → encoder → embedding → NIRT response model → P(correct) per model
      → cost-aware routing decision → (single call) or (triage → decompose → route each sub-task → synthesize)
```

The project is organized as four layered packages under `src/`, each with a
single responsibility and an enforced import direction
(`router` → nothing research-y; `training` → `router`; `evaluation` → both):

| package | role | may import |
|---|---|---|
| [`router`](../src/router/) | **serving**: model definitions, checkpoint inference, the `RouterModel` interface, the agentic orchestrator | — |
| [`training`](../src/training/) | the data pipeline + fitting/training code | `router` |
| [`evaluation`](../src/evaluation/) | anything that needs ground truth: oracle labels, routing/regret metrics, OOD eval, the LLM-judge pipeline | `router`, `training` |
| [`decompose`](../src/decompose/) | task decomposition signals/classifiers, kept dependency-free | (embedders wrap `router.embeddings.encoder` only) |

The concrete classifiers under [`decompose/classifiers/`](../src/decompose/classifiers/)
are the prompt-decomposition heads: domain, task type, expected output length,
and eleven free-label attributes. They turn a prompt into `PromptSignals` for
`Router`. Their corpora and sweeps live in
[`training/prompt_decomposition/`](../src/training/prompt_decomposition/) and their
audits in [`evaluation/prompt_decomposition/`](../src/evaluation/prompt_decomposition/).

This layering is enforced by `import-linter` (`[tool.importlinter]` in
[`pyproject.toml`](../pyproject.toml)) — see
[docs/architecture.md](architecture.md) for the full class hierarchy and
rationale.

## Documentation map

The README is a front door; the depth lives in `docs/`:

| doc | covers |
|---|---|
| [RUNBOOK.md](RUNBOOK.md) | Copy-paste commands to actually run every pipeline, end to end, in order |
| [architecture.md](architecture.md) | Class hierarchies across `router`/`training`/`evaluation`, the package-layering contract, intentional name collisions |
| [workflows.md](workflows.md) | Every runnable pipeline as a diagram + call-chain table (data → embeddings → training → eval → agentic serving) |
| [data_structures.md](data_structures.md) | `Config`, canonical parquet schemas, splits, the `TrainingData` facade, embedding stores |
| [data_sources.md](data_sources.md) | RouterBench / Arena / GPT-4 Judge / lm-eval-harness ingestion, chance correction, model registry |
| [query_and_model_representation.md](query_and_model_representation.md) | Encoder pathways, LLM profiles, taxonomy/relevance/FAISS query bank |
| [nirt_model.md](nirt_model.md) | The NIRT response model, both orientations, the capacity workstream (P1–P6) and its findings |
| [baseline_control_arm.md](baseline_control_arm.md) | The frozen Bernoulli/BCE and continuous (Normal/Beta/ZOIB) control arms |
| [irt_router.md](irt_router.md) | The IRT-Router paper's 20-LLM × 12-dataset benchmark, replicated as an isolated pipeline |
| [prompt_signals.md](prompt_signals.md) | Which prompt signals are measurable, which were measured impossible, and what each is worth |
| [prompt_decomposition.md](prompt_decomposition.md) | The five heads: taxonomies, calibration, the handoff vector, known limits |
| [prompt_decomposition_experiments.md](prompt_decomposition_experiments.md) | Every head experiment including the failures, and why each was abandoned |
| [pool_expansion.md](pool_expansion.md) / [pool_expansion_results.md](pool_expansion_results.md) | Candidate-pool expansion workstream (E0–E2 done, E3 superseded by irt_router.md) and its results ledger |
| [routing_interface.md](routing_interface.md) | The `RouterModel` ABC, `RoutingResult`, registered strategies (`matrix`/`nirt`/`knn`/`mlp`/`random`) |
| [routing_evaluation.md](routing_evaluation.md) | Oracle labels, regret/hit metrics, the prediction-vs-routing distinction |
| [agentic_router.md](agentic_router.md) | `AgenticRouter`: triage → single call or recursive decompose/route/synthesize |
| [gateway.md](../aar-gateway/docs/gateway.md) | HTTP front door (Anthropic/OpenAI wire protocols) over the `ClientRegistry`, config format, Claude-subscription vs. metered-key lanes |
| [anchor_judge.md](anchor_judge.md) | The query-matched gold/preference overlap pipeline resolving whether correctness and human/judge preference are one utility or two |

## Installation

Requires Python ≥ 3.11.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on POSIX
pip install -e ".[dev,research]"  # router + torch / transformers / sentence-transformers / pytest / import-linter
```

Base install is deliberately light (`numpy`/`pandas`/`pyarrow`/`pyyaml`/
`python-dotenv`/`tqdm`) — everything else is an extra, so installing just
what you're touching doesn't pull in what you're not:

* `pip install -e ".[research]"` — the data pipeline / training / evaluation
  stack: `torch`, `transformers`, `sentence-transformers`, `faiss-cpu`,
  `umap-learn`, `hdbscan`, `scikit-learn`, `matplotlib`, `scipy`,
  `huggingface_hub`, `tabulate`, `datasets`. Nothing in
  `router.agentic`/`router.llm` imports any of this.
* `pip install -e ".[agentic]"` — `langchain` / `langchain-core`, for
  `LangChainClient` and `LangChainToolOrchestrator`. The core agentic path
  (echo/callable backends, `RecursiveOrchestrator`, `HeuristicTriage`) needs
  none of this.
* `pip install -e ".[labeling]"` — `openai` / `anthropic` SDKs, for the
  anchor-judge pipeline and the future LLM cluster-labelling stage.
* `pip install -e ./aar-gateway` — just `fastapi` / `uvicorn`, for the HTTP
  gateway (`aar-gateway`) that exposes the `ClientRegistry` over the
  Anthropic/OpenAI wire protocols. A seconds-long install with no `research`
  needed — this is the path for someone who only wants to run the gateway
  locally. See [docs/gateway.md](../aar-gateway/docs/gateway.md).
* `pip install lm-eval` — to run lm-evaluation-harness yourself.

## Quickstart: routing

```python
from router.routing import NIRTRouter
from router.agentic import AgenticRouter

router = NIRTRouter.from_run("nirt-2d-projected")
result = router.route_text(["Prove that sqrt(2) is irrational."])
result.selected_model_ids, result.model_mix()

# triage → single routed call, or recursive decompose/route/synthesize
agent = AgenticRouter(router)
outcome = agent.run("Plan a 3-day trip to Kyoto and translate the itinerary to Japanese.")
outcome.mode   # "single" or "decompose"
```

`RouterModel` subclasses implement only `predict_scores(query_ids)`; the ABC
supplies `.route()` / `.route_text()` / `.model_ids` /
`.default_model_costs`. Registered strategies: `matrix`, `nirt`, `knn`,
`mlp`, `random` (`router.routing.registry.build_router`). Full contract:
[docs/routing_interface.md](routing_interface.md). Oracle-relative
scoring (regret, hit-rate — needs ground truth) is a free function,
`evaluation.routing.oracle.evaluate_router`, deliberately kept out of
`RouterModel` itself so serving code never imports `evaluation`.

## Data pipeline (Foundation)

Heterogeneous benchmark sources are turned into one clean, reproducible
dataset consumed through a single facade, `training.data.facade`.

```bash
python -m training.phase0                          # deterministic parts only
python -m training.phase0 --embeddings              # + the slow encode
python -m training.phase0 --taxonomy --query-bank   # + clustering / FAISS bank
```

Or stage by stage — see [docs/workflows.md](workflows.md) for the full
diagram; short version:

```bash
python scripts/data/download_routerbench.py
python scripts/data/download_arena.py                     # also fetches judge battles
python scripts/data/build_response_matrix.py               # → data/processed/*.parquet
python scripts/data/build_splits.py                        # → data/splits/*.json
python scripts/models/build_model_profiles.py               # → model_profiles.parquet
python scripts/data/build_nirt_dataset.py                   # → nirt_observations.parquet
python scripts/embeddings/build_query_embeddings.py   --pathway retrieval
python scripts/embeddings/build_query_embeddings.py   --pathway irt --from-nirt
python scripts/embeddings/build_profile_embeddings.py --all-pathways
```

```python
from router.config import load_config
from training.data.facade import load_training_data

d = load_training_data(load_config())
R      = d.correctness_matrix(split="train", combine_metrics=["accuracy", "mc_accuracy"])
q_emb  = d.query_embeddings(pathway="irt")
ds     = d.nirt_dataset(split="train", pathway="irt")       # -> {query_embedding, model_embedding, target}
```

Current outputs from the raw data (rebuilt 2026-10-12 with `training.phase0
--sources routerbench,chatbot_arena,gpt4_judge,irt_router`; anchor_judge's
8,000 rows need its gated pipeline, §7 of the runbook). Since `force_cold:
[mixtral-8x7b-instruct]` in `configs/phase0.yaml`, RouterBench's Mixtral is a
cold-start model: 28 warm models and 72,966 fewer NIRT observations than the
29 / 1,449,159 recorded before that change.

```
1,904,408 observations   271,305 queries   90 models   100 datasets
by metric : mc_accuracy 869,456 | accuracy 693,796 | judge_preference 226,202 | arena_preference 114,954
by source : routerbench 802,626 | irt_router 760,626 | gpt4_judge 218,202 | chatbot_arena 114,954 | anchor_judge 8,000
train / val / test queries : 216,939 / 27,072 / 27,235
cold-start models : claude-2.0, gpt_4o_mini, mistral_7b_instruct_v02, mixtral-8x7b-instruct, mixtral_8x7b_instruct, vicuna-13b
NIRT observations : 1,376,193  (train 1,101,536 / val 136,686 / test 137,971; 28 warm models)
```

Dataset sources, schema, chance correction and split mechanics:
[docs/data_sources.md](data_sources.md) and
[docs/data_structures.md](data_structures.md).

## NIRT model

[`src/router/nirt/`](../src/router/nirt/) implements a bilinear IRT core with
two orientations (`model.orientation` in [`configs/nirt.yaml`](../configs/nirt.yaml)):
**`query_latent`** (ours — `theta_q = f(e_q)`, `(a_m, b_m)` on the model) and
**`model_latent`** (IRT-Router, Song et al. ACL'25 — `theta_m` ability on the
LLM, `(a_q, b_q)` on the query).

```bash
python scripts/nirt/train_nirt.py --config configs/nirt.yaml --dim 2 --model-params projected --name nirt-2d-projected
python scripts/nirt/compare.py        --run nirt-2d-projected --split test
python scripts/nirt/ab_orientation.py --dim 2 --model-params projected
python scripts/nirt/coldstart.py      --run nirt-2d-projected --split test
```

The capacity workstream (P1–P6) found the model was never underfitting — it
**overfits** past ~epoch 15, and early stopping is what had been acting as
the regularizer; `model_params: free` beats `projected`; the RouterBench
0-/5-shot confound (`query_features`) is the single biggest lever; and
routing-aware losses (pairwise/listwise) are the first change that moves the
actual routing decision rather than only prediction metrics. Cold-start from
a profile alone still doesn't beat the `warm_mean` baseline. Full findings:
[docs/nirt_model.md](nirt_model.md).

A separate, frozen Bernoulli/BCE control arm plus continuous
(Normal/Beta/ZOIB) response heads live in
[`src/training/nirt/baseline/`](../src/training/nirt/baseline/) — see
[docs/baseline_control_arm.md](baseline_control_arm.md). The IRT-Router
paper's own 20-LLM × 12-dataset benchmark is replicated as an isolated
parallel pipeline (`configs/irt_router.yaml`) — see
[docs/irt_router.md](irt_router.md).

## Agentic router

[`src/router/agentic/`](../src/router/agentic/) wraps a `RouterModel` with an LLM
client registry and a triage step: a query is either answered with a single
routed model call, or decomposed into sub-tasks that are each routed and
answered independently, then synthesized. Triage is driven by the router's
own predicted quality (`HeuristicTriage`, default threshold 0.55) or an
`LLMTriage`; orchestration is `RecursiveOrchestrator` by default, or
`LangChainToolOrchestrator` (exposes `route_query` / `answer_with_model` /
`route_and_answer` as LangChain tools) with the `agentic` extra installed.
Full write-up: [docs/agentic_router.md](agentic_router.md).

## Gateway

[`aar-gateway/`](../aar-gateway/) (its own package, mirrored to the standalone `aar-gateway` repo) runs Claude Code through the
router, the way claude-code-router does: a local process Claude Code points
at, keeping the user's own login and tools. At every new message the trained
router picks the model for that turn. It only ever picks a model as cheap as
or cheaper than the one the user chose, it accounts for the prompt-cache
cost of switching, and the user sets the cost preference. Claude models go
to Anthropic on the user's login; other models (OpenAI-compatible APIs,
Ollama) get the request with its tools translated. A dashboard shows actual
cost against "no router" and "always the strongest model".
`aar-gateway code` / `aar-gateway dashboard`; walkthrough in
[docs/gateway.md](../aar-gateway/docs/gateway.md).

## Configuration

Everything experimental is in [`configs/phase0.yaml`](../configs/phase0.yaml)
(paths, per-source locations, chance correction, embedding pathways,
taxonomy, anchor-judge, splits). Model-specific configs:
[`configs/nirt.yaml`](../configs/nirt.yaml) (NIRT model + training),
[`configs/phase1.yaml`](../configs/phase1.yaml) /
[`configs/phase2.yaml`](../configs/phase2.yaml) (Bernoulli / continuous control
arms), [`configs/irt_router.yaml`](../configs/irt_router.yaml) (isolated
paper-replication track), plus
[`configs/model_registry.yaml`](../configs/model_registry.yaml),
[`configs/model_profiles.yaml`](../configs/model_profiles.yaml) and
[`configs/model_prices.yaml`](../configs/model_prices.yaml). Source code does
not hard-code these; pass `--config path/to.yaml` to any script.

## Tests

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 pytest -q
```

**Both environment variables matter, and without them the suite hangs rather
than fails** — no output, no error, just a process that never returns:

* `OMP_NUM_THREADS=1` — faiss deadlocks inside `IndexFlat::search` when its
  OpenMP pool contends with the other native libraries here. The run looks
  merely slow. To tell the difference, check the process twice a few seconds
  apart with `ps -o pid,time,etime -p <pid>`: if `TIME` is frozen while
  `ELAPSED` grows, it is blocked, not working, and `sample <pid> 3` names the
  call it is stuck in. With the pin the full suite takes ~3.5 minutes.
* `HF_HUB_OFFLINE=1` — a half-finished model download leaves a lock in
  `~/.cache/huggingface` that is never released, and the next run parks on it
  forever with its HTTPS socket in `CLOSE_WAIT`. Offline turns that into an
  immediate, readable error. Everything the suite needs is already cached.

891 tests across `tests/router/{agentic,embeddings,nirt,routing}`,
`tests/training/{data,models,nirt,retrieval,taxonomy}`,
`tests/evaluation/{data,judge,nirt,pool_expansion,routing}`,
`tests/decompose/`. Shared builders (config fixtures, FAISS bank, synthetic
NIRT worlds) live in `tests/helpers/` — import as `helpers`. Package
layering is checked separately: `lint-imports` (via `import-linter`,
installed with the `dev` extra).

## Known limitations / decisions needing human review

* **Pairwise sources are relative, not absolute.** Arena + Judge preference
  rows carry the opponent in metadata and are returned separately by the
  facade; how a preference head should consume them is an open modeling
  question — see [docs/data_sources.md](data_sources.md) and
  [docs/anchor_judge.md](anchor_judge.md), which investigates whether
  correctness and preference are one utility or two.
* **GPT-4 Judge is one matchup.** All 109k battles are `gpt-4-1106-preview`
  vs `mixtral-8x7b-instruct`, so `judge_preference` only constrains those two
  models' relative placement per query.
* **RouterBench has no token counts** (only cost); lm-harness runs would add
  real ones.
* **Cold-start from a profile alone doesn't work yet** — 9 training LLMs is
  too few for the projection to place a held-out model competitively against
  the `warm_mean` baseline.
* **The RouterBench pool is GPT-4-dominated**, so quality-only routing on it
  collapses to "always GPT-4"; the IRT-Router benchmark
  ([docs/irt_router.md](irt_router.md)) is the pool where learned
  routing actually beats a fixed-model baseline on accuracy *and* cost.
* **Model registry cross-source merges are conservative** — several
  probably-safe merges are kept separate pending review. See
  [docs/data_sources.md](data_sources.md).
* **Two intentionally similarly-named classes**: `router.llm.client.LLMClient`
  (model registry / provider adapters) and
  `router.agentic.llm_clients.LLMClient` (the agentic orchestrator's chat
  interface) are unrelated — see [docs/architecture.md](architecture.md).
