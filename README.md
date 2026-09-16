# agentic-llm-router

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
| [`router`](src/router/) | **serving**: model definitions, checkpoint inference, the `Router` interface, the agentic orchestrator | — |
| [`training`](src/training/) | the data pipeline + fitting/training code | `router` |
| [`evaluation`](src/evaluation/) | anything that needs ground truth: oracle labels, routing/regret metrics, OOD eval, the LLM-judge pipeline | `router`, `training` |
| [`decompose`](src/decompose/) | task decomposition signals/classifiers, kept dependency-free | (embedders wrap `router.embeddings.encoder` only) |

This layering is enforced by `import-linter` (`[tool.importlinter]` in
[`pyproject.toml`](pyproject.toml)) — see
[docs/architecture.md](docs/architecture.md) for the full class hierarchy and
rationale.

## Documentation map

The README is a front door; the depth lives in `docs/`:

| doc | covers |
|---|---|
| [architecture.md](docs/architecture.md) | Class hierarchies across `router`/`training`/`evaluation`, the package-layering contract, intentional name collisions |
| [workflows.md](docs/workflows.md) | Every runnable pipeline as a diagram + call-chain table (data → embeddings → training → eval → agentic serving) |
| [data_structures.md](docs/data_structures.md) | `Config`, canonical parquet schemas, splits, the `TrainingData` facade, embedding stores |
| [data_sources.md](docs/data_sources.md) | RouterBench / Arena / GPT-4 Judge / lm-eval-harness ingestion, chance correction, model registry |
| [query_and_model_representation.md](docs/query_and_model_representation.md) | Encoder pathways, LLM profiles, taxonomy/relevance/FAISS query bank |
| [nirt_model.md](docs/nirt_model.md) | The NIRT response model, both orientations, the capacity workstream (P1–P6) and its findings |
| [baseline_control_arm.md](docs/baseline_control_arm.md) | The frozen Bernoulli/BCE and continuous (Normal/Beta/ZOIB) control arms |
| [irt_router.md](docs/irt_router.md) | The IRT-Router paper's 20-LLM × 12-dataset benchmark, replicated as an isolated pipeline |
| [pool_expansion.md](docs/pool_expansion.md) / [pool_expansion_results.md](docs/pool_expansion_results.md) | Candidate-pool expansion workstream (E0–E2 done, E3 superseded by irt_router.md) and its results ledger |
| [routing_interface.md](docs/routing_interface.md) | The `Router` ABC, `RoutingResult`, registered strategies (`matrix`/`nirt`/`knn`/`mlp`/`random`) |
| [routing_evaluation.md](docs/routing_evaluation.md) | Oracle labels, regret/hit metrics, the prediction-vs-routing distinction |
| [agentic_router.md](docs/agentic_router.md) | `AgenticRouter`: triage → single call or recursive decompose/route/synthesize |
| [anchor_judge.md](docs/anchor_judge.md) | The query-matched gold/preference overlap pipeline resolving whether correctness and human/judge preference are one utility or two |

## Installation

Requires Python ≥ 3.11.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on POSIX
pip install -e ".[dev]"           # router + torch / transformers / sentence-transformers / pytest / import-linter
```

Optional extras:

* `pip install -e ".[agentic]"` — `langchain` / `langchain-core`, for
  `LangChainClient` and `LangChainToolOrchestrator`. The core agentic path
  (echo/callable backends, `RecursiveOrchestrator`, `HeuristicTriage`) needs
  none of this.
* `pip install -e ".[labeling]"` — `openai` / `anthropic` SDKs, for the
  anchor-judge pipeline and the future LLM cluster-labelling stage.
* `pip install lm-eval` — to run lm-evaluation-harness yourself.

Core deps also include `matplotlib`, `faiss-cpu`, `umap-learn`, `hdbscan`
(query taxonomy + FAISS bank + diagnostic plots).

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

`Router` subclasses implement only `predict_scores(query_ids)`; the ABC
supplies `.route()` / `.route_text()` / `.model_ids` /
`.default_model_costs`. Registered strategies: `matrix`, `nirt`, `knn`,
`mlp`, `random` (`router.routing.registry.build_router`). Full contract:
[docs/routing_interface.md](docs/routing_interface.md). Oracle-relative
scoring (regret, hit-rate — needs ground truth) is a free function,
`evaluation.routing.oracle.evaluate_router`, deliberately kept out of
`Router` itself so serving code never imports `evaluation`.

## Data pipeline (Foundation)

Heterogeneous benchmark sources are turned into one clean, reproducible
dataset consumed through a single facade, `training.data.facade`.

```bash
python -m training.phase0                          # deterministic parts only
python -m training.phase0 --embeddings              # + the slow encode
python -m training.phase0 --taxonomy --query-bank   # + clustering / FAISS bank
```

Or stage by stage — see [docs/workflows.md](docs/workflows.md) for the full
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

Current outputs from the checked-in raw data:

```
734,469 observations   196,791 queries   69 models   88 datasets
by metric : mc_accuracy 294,998 | judge_preference 218,202 | arena_preference 114,954 | accuracy 106,315
by source : routerbench 401,313 | gpt4_judge 218,202 | chatbot_arena 114,954
train / val / test queries : 157,271 / 19,672 / 19,767
cold-start models : claude-2.0, code-llama-34b-instruct, vicuna-13b, yi-34b-chat
NIRT observations : 328,347  (train 262,548 / val 32,706 / test 33,093; 9 warm models, 36,483 queries)
```

Dataset sources, schema, chance correction and split mechanics:
[docs/data_sources.md](docs/data_sources.md) and
[docs/data_structures.md](docs/data_structures.md).

## NIRT model

[`src/router/nirt/`](src/router/nirt/) implements a bilinear IRT core with
two orientations (`model.orientation` in [`configs/nirt.yaml`](configs/nirt.yaml)):
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
[docs/nirt_model.md](docs/nirt_model.md).

A separate, frozen Bernoulli/BCE control arm plus continuous
(Normal/Beta/ZOIB) response heads live in
[`src/training/nirt/baseline/`](src/training/nirt/baseline/) — see
[docs/baseline_control_arm.md](docs/baseline_control_arm.md). The IRT-Router
paper's own 20-LLM × 12-dataset benchmark is replicated as an isolated
parallel pipeline (`configs/irt_router.yaml`) — see
[docs/irt_router.md](docs/irt_router.md).

## Agentic router

[`src/router/agentic/`](src/router/agentic/) wraps a `Router` with an LLM
client registry and a triage step: a query is either answered with a single
routed model call, or decomposed into sub-tasks that are each routed and
answered independently, then synthesized. Triage is driven by the router's
own predicted quality (`HeuristicTriage`, default threshold 0.55) or an
`LLMTriage`; orchestration is `RecursiveOrchestrator` by default, or
`LangChainToolOrchestrator` (exposes `route_query` / `answer_with_model` /
`route_and_answer` as LangChain tools) with the `agentic` extra installed.
Full write-up: [docs/agentic_router.md](docs/agentic_router.md).

## Configuration

Everything experimental is in [`configs/phase0.yaml`](configs/phase0.yaml)
(paths, per-source locations, chance correction, embedding pathways,
taxonomy, anchor-judge, splits). Model-specific configs:
[`configs/nirt.yaml`](configs/nirt.yaml) (NIRT model + training),
[`configs/phase1.yaml`](configs/phase1.yaml) /
[`configs/phase2.yaml`](configs/phase2.yaml) (Bernoulli / continuous control
arms), [`configs/irt_router.yaml`](configs/irt_router.yaml) (isolated
paper-replication track), plus
[`configs/model_registry.yaml`](configs/model_registry.yaml),
[`configs/model_profiles.yaml`](configs/model_profiles.yaml) and
[`configs/model_prices.yaml`](configs/model_prices.yaml). Source code does
not hard-code these; pass `--config path/to.yaml` to any script.

## Tests

```bash
pytest -q
```

417 tests across `tests/router/{agentic,embeddings,nirt,routing}`,
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
  question — see [docs/data_sources.md](docs/data_sources.md) and
  [docs/anchor_judge.md](docs/anchor_judge.md), which investigates whether
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
  ([docs/irt_router.md](docs/irt_router.md)) is the pool where learned
  routing actually beats a fixed-model baseline on accuracy *and* cost.
* **Model registry cross-source merges are conservative** — several
  probably-safe merges are kept separate pending review. See
  [docs/data_sources.md](docs/data_sources.md).
* **Two intentionally similarly-named classes**: `router.llm.client.LLMClient`
  (model registry / provider adapters) and
  `router.agentic.llm_clients.LLMClient` (the agentic orchestrator's chat
  interface) are unrelated — see [docs/architecture.md](docs/architecture.md).
