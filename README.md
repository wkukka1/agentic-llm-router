# agentic-llm-router

Research code for a **cost-aware LLM router** that models each candidate LLM as a
test-taker with a multidimensional latent ability vector `theta_m`, each query as
an item with difficulty / discrimination / relevance `(b_q, a_q, r_q)`, and
observed benchmark performance as the response data (NIRT / IRT-Router style).

```
query → encoder → embedding → kNN warm-up / relevance lookup
      → IRT response model → P(model answers correctly) → cost-aware routing
```

**Foundation** (the data pipeline, originally "Phase 0") is complete: heterogeneous
benchmark sources are turned into one clean, reproducible dataset consumed through a single
facade (`router.data.phase1`). **Phase 2 — the NIRT baseline** — has started:
[`src/router/nirt/`](src/router/nirt/) holds a first, deliberately simple response model
(`theta_q = f_theta(e_q)`, `y_hat = sigmoid(a_m · theta_q − b_m)`). See
[docs/nirt_model.md](docs/nirt_model.md).

> **Scope of the current NIRT work.** v1 is a *predictor* only, evaluated on prediction
> quality. Still to come, in order: classical-IRT baseline, ranking / routing / cold-start
> evaluation, the 1D→2D→4D dimension ladder, then the multi-source (Arena / GPT-4-Judge)
> objectives. The pairwise / M-IRT path stays isolated from N-IRT. No bandits or serving
> router yet.

### Architecture this data feeds (Phase 1, not built)

```
 RouterBench ──correctness──┐
                            ├─►  multi-task AMIRT  ──►  θ_q (query latent) + a_m (model latent)
 Arena ──────pairwise──────┤        │                        │
 GPT-4 Judge ─pairwise─────┘        └── separate preference heads (λ_arena, λ_judge)
```

RouterBench correctness is the **primary** IRT signal; Arena (human) and GPT-4
Judge (LLM) preferences are **auxiliary pairwise** objectives with their own
heads — never folded into `R[q,m]`. Taxonomy/relevance vectors are discovered
*after* an initial NIRT baseline, so those stages are deferred (see below).

**Text pathway (NIRT-style).** A frozen `bert-base-uncased` encoder (`irt`
pathway) embeds **both** queries and per-model *profile* texts into one 768-d
space. Phase 1 projects the query vector to item parameters `(b_q, a_q, r_q)` and
the profile vector to an initial ability `theta_m` — so a cold-start model is
placed from its description alone. A separate `retrieval` pathway (MiniLM) serves
the kNN / FAISS bank. See [docs/embedding_pathways.md](docs/embedding_pathways.md)
and [docs/model_profiles.md](docs/model_profiles.md).

---

## Foundation goals (data pipeline)

| # | Goal | Status |
|---|------|--------|
| 1 | Canonical `(query, model, score, metric_type)` schema for all sources | **implemented** |
| 2 | RouterBench loader → canonical correctness observations | **implemented** |
| 3 | Chatbot Arena + GPT-4 Judge loaders → pairwise-preference observations (separate `metric_type`s) | **implemented** |
| 4 | lm-evaluation-harness ingestion interface + consumer | **implemented** (consumer + runner; you supply the eval runs) |
| 5 | Response matrix `responses/queries/models.parquet` + `R[q,m]` pivot | **implemented** |
| 6 | Multiple-choice chance correction (**no** guessing parameter) + per-model bias diagnostic | **implemented** |
| 7 | Data-quality checks + Phase 0 summary report | **implemented** |
| 8 | Deterministic, leakage-free train/val/test + cold-start-model split | **implemented** |
| 9 | Confidence-based canonical model registry (`configs/model_registry.yaml`) | **implemented** |
| 10 | Encoder **pathways** — `retrieval` (MiniLM) + `irt` (BERT), NIRT-style; **streaming + resumable** builds | **implemented** |
| 11 | **LLM profiles** (`model_profiles.parquet`) + profile embeddings for `theta_m` init | **implemented** |
| 12 | **NIRT dataset** (`nirt_observations.parquet` + `NIRTDataset` → `query_emb, model_emb, target`) | **implemented** |
| 13 | **Phase 1 data facade** (`router.data.phase1`) | **implemented** |
| 14 | UMAP → HDBSCAN clustering of the NIRT queries | **implemented** (`router.taxonomy`, 39 clusters) |
| 15 | Cluster → candidate ability taxonomy (family + top terms, no LLM) | **implemented** (`data/taxonomy/taxonomy.json`) |
| 16 | Representative queries + provider-agnostic cached LLM labeling | *deferred — opt-in refinement* |
| 17 | Relevance vectors `r_q` (`sum = 1`, `>= 0`) + FAISS query bank | **implemented** (`router.taxonomy` / `router.retrieval`) |

Stages 14/15/17 were built for the Phase 1 baseline (they condition `a_q` and
feed the warm-up blend); see [docs/query_representation.md](docs/query_representation.md).
Stage 16 (LLM cluster labelling) is still deferred as an opt-in refinement.

```bash
python -m router.phase0 --taxonomy --query-bank      # or the scripts/taxonomy + scripts/retrieval scripts
```

## Phase 2 — NIRT baseline

[`src/router/nirt/`](src/router/nirt/) + [`configs/nirt.yaml`](configs/nirt.yaml). Two
orientations of the same bilinear IRT core (`model.orientation`): `query_latent` (ours —
`theta_q = f(e_q)`, `(a_m,b_m)` on the model) and `model_latent` (IRT-Router, Song et al.
ACL'25 — `theta_m` ability on the LLM, `(a_q,b_q)` on the query). Full write-up, results and
roadmap: [docs/nirt_model.md](docs/nirt_model.md).

```bash
python scripts/nirt/train_nirt.py --config configs/nirt.yaml --dim 2 --model-params projected --name nirt-2d-projected
python scripts/nirt/compare.py        --run nirt-2d-projected --split test      # vs classical IRT + routing/cost
python scripts/nirt/ab_orientation.py --dim 2 --model-params projected          # query_latent vs model_latent
python scripts/nirt/coldstart.py      --run nirt-2d-projected --split test      # place a held-out LLM from its profile
```

| status | |
|--------|--|
| **implemented** | both orientations (`NIRTModel` / `IRTRouterModel`, 1..K-dim, `projected` / `free`), trainer + checkpoints, prediction / ranking eval, **classical-IRT baseline** (transductive cell-holdout ceiling + `theta=0` lower bound), **cost-aware routing** with a λ sweep, **cold-start experiment** |
| *findings* | NIRT (text only) matches the matrix-factorisation ceiling on prediction (test BCE ≈0.59); `model_latent` edges `query_latent` (better at 1D, better cost frontier). GPT-4-dominated pool → quality routing ≈ always-GPT-4; cost-aware routing gives ≈−54 % spend at −3 accuracy points. **Cold-start from a profile alone does not beat the `warm_mean` baseline** — 9 training LLMs is too few for the projection. See [docs/nirt_model.md](docs/nirt_model.md). |
| *next (part 2)* | OOD held-out-benchmark split; `Reward(α)` + AIQ metrics; in-house KNN/MLP router baselines; clustering → relevance `r_q` → relevance-masked N-IRT head + kNN query warm-up. Then multi-source Arena / GPT-4-Judge (M-IRT). |

### Phase 1 baseline — plain Bernoulli / BCE control arm

A deliberately simple, faithful control arm that the continuous response model
will replace. `model_latent` IRT core `logit = a_q·θ_m − b_q`, `p = sigmoid`,
BCE loss; `a_q` conditioned on the relevance vector `r_q`, optional FAISS
warm-up blend, an inert length head (no token labels in Phase 0). Lives in
[`src/router/nirt/`](src/router/nirt/) (`baseline.py`, `components.py`,
`response_head.py`, `losses.py`, `checkpoint.py`, `baseline_train.py`,
`baseline_eval.py`, `calibration.py`, `diagnostics.py`, `synthetic.py`) +
[`configs/phase1.yaml`](configs/phase1.yaml). Full write-up:
[docs/phase1_baseline.md](docs/phase1_baseline.md).

```bash
python scripts/synthetic.py                                             # synthetic multidim-IRT recovery gate
python scripts/train_baseline.py --config configs/phase1.yaml           # -> artifacts/phase1/baseline/
python scripts/evaluate.py --checkpoint artifacts/phase1/baseline --split test
python scripts/inspect_baseline.py --checkpoint artifacts/phase1/baseline
```

Binary label = `1[score_effective ≥ 0.5]` on RouterBench correctness only
(**Arena excluded**); original graded score preserved. Test: accuracy 0.71,
BCE 0.55, Brier 0.19, AUC 0.78, ECE 0.015, ΔBCE vs per-model-mean −0.076;
theta effective rank 6.5/8 (not collapsed). Synthetic recovery passes
(difficulty ρ 0.85, ability ρ 0.98).

### Phase 2 — continuous response model

Same representation, **different response likelihood** over the graded score
`y ∈ [0,1]`. Pluggable `ResponseHead`s (`src/router/nirt/response_head.py` +
`continuous_{normal,beta,zoib}.py`): heteroskedastic **Normal**
`y ~ N(z, σ_qm)`, **Beta** `μ=σ(z), y~Beta(μκ,(1−μ)κ)` (interior only), **ZOIB**
(explicit `π₀`,`π₁` boundary masses + Beta middle). `configs/phase2.yaml`,
`scripts/{train,evaluate,synthetic}_continuous.py`,
`scripts/{boundary_stats,compare_response_models}.py`. Full write-up:
[docs/phase2_response_model.md](docs/phase2_response_model.md).

```bash
python scripts/boundary_stats.py
python scripts/synthetic.py --response zoib
python scripts/train_continuous.py --response zoib
python scripts/evaluate.py --checkpoint artifacts/phase2/zoib
python scripts/compare_response_models.py
```

Boundary stats: **~79 % of graded scores are exactly 0 or 1** (≈37 % / ≈41 %),
~22 % interior — which is why plain Beta is starved and ZOIB's boundary masses
matter. `π₀`/`π₁` are literal probability mass at the score boundaries, **not a
3PL guessing parameter** (no IRT lower asymptote). All 3 synthetic recovery
gates pass.

### Phase 2 — kNN-imputed query representation

A separate NIRT variant that **replaces `e_q` with the similarity-weighted mean
of its `k` nearest training queries** (self-excluded, via the FAISS bank) before
the query head, snapping every query onto the training manifold. New
`data.query_pathway` key swaps only the query store;
`router.retrieval.knn_impute` builds it. Leakage-safe OOD bank drops the held-out
families. Full write-up + how to run:
[docs/knn_imputed_queries.md](docs/knn_imputed_queries.md).

```bash
python scripts/retrieval/build_query_bank.py --exclude-ood-families
python scripts/nirt/knn_impute_sweep.py --k 1,5,10,25
```

Finding (`K=2`, similarity-weighted): imputation is a **clearly better calibrated
predictor in-distribution** — test BCE 0.612 → 0.579, AUC 0.739 → 0.772,
Spearman 0.44 → 0.47, also beating the IRT-Router paper baseline — with gains
saturating by k≈10. Argmax routing is ~unchanged (GPT-4-dominated pool). **OOD
calibration gets worse** (BCE +0.05–0.08: a held-out math/code query imputed from
non-math/code neighbours is biased), though OOD ranking still nudges up.

---

## Installation

Requires Python ≥ 3.11. A `.venv/` already exists in this checkout.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on POSIX
pip install -e ".[dev]"           # router + torch / transformers / sentence-transformers / pytest
```

Core deps now include `matplotlib` + `faiss-cpu` / `umap-learn` / `hdbscan`
(query taxonomy + FAISS bank + Phase 1 plots). Optional extras:
`pip install -e ".[labeling]"` (OpenAI/Anthropic SDKs for the future LLM cluster
labelling stage), `pip install lm-eval` (to run the harness yourself).

---

## Dataset sources

| Source | `source` / `metric_type` | Origin | Notes |
|--------|--------------------------|--------|-------|
| **RouterBench** | `routerbench` / `accuracy`, `mc_accuracy` | HF `withmartian/routerbench` (git-LFS) → `routerbench/*.pkl` | 11 models × ~36.5k prompts, 86 `eval_name` tasks. Graded `performance` in `{0.0 … 1.0}` (not binary). Per-model **cost** but **no token counts**. `dev`/`test` marker parsed from `sample_id`. **Primary correctness signal.** |
| **Chatbot Arena** | `chatbot_arena` / `arena_preference` | HF `lmarena-ai/arena-human-preference-55k` → `data/raw/arena/arena-human-preference-55k/` | 57,477 human pairwise votes, ~60 models. **No timestamps / language** in this release → missing. Each battle → 2 observations. |
| **GPT-4 Judge** | `gpt4_judge` / `judge_preference` | HF `routellm/gpt4_judge_battles` → `data/raw/arena/gpt4_judge_battles/` | 109,101 LLM-judged battles, all `gpt-4-1106-preview` vs `mixtral-8x7b-instruct`. Separate head — not assumed calibrated like human votes. |
| **lm-evaluation-harness** | `lm_eval_harness` / `acc`, `acc_norm`, … | you run it → `data/raw/lm_harness/*samples*.jsonl` | Phase 0 does **not** run expensive evals. `scripts/data/run_lm_harness.py` emits the reproducible command; the loader consumes per-sample logs (`--log_samples`). Aggregate-only results are skipped with a warning. |

Pairwise interpretation and the separate-heads rationale:
[docs/arena_interpretation.md](docs/arena_interpretation.md). Model identity /
alias-merge policy: [docs/model_registry.md](docs/model_registry.md).

---

## Data schema

Canonical tables live in `data/processed/` (see
[`src/router/data/schemas.py`](src/router/data/schemas.py)).

### `responses.parquet` — one row per observation

`query_id, model_id, query, score, metric_type, dataset, split, source,
is_multiple_choice, n_choices, input_tokens, output_tokens, latency, cost,
metadata, corrected_score, chance_level, chance_corrected, content_hash`

* **`score`** is always the original source score on its original scale.
* **`metric_type`** ∈ `{accuracy, mc_accuracy, exact_match, f1, pass@1, acc,
  acc_norm, arena_preference, judge_preference}` — metrics are **never** merged
  onto one axis. `schemas.MetricType.CORRECTNESS` vs `.PAIRWISE` partition them.
* Missing `cost` / `*_tokens` / `latency` are **null**, never `0`.
* `corrected_score` is the chance-corrected value (MC only); the original is kept.

### `queries.parquet` — one row per `query_id`

`query_id, query, dataset, source, content_hash, n_observations, n_models,
cluster_id, metadata`  (`cluster_id` reserved for the deferred clustering stage).
Embeddings live in a dedicated store keyed by `query_id`, not inline.

### `models.parquet` — one row per `model_id`

`model_id, model_name, provider, family, version, profile_text, source_datasets,
n_observations, sources, metadata`. Identity comes from the human-reviewed
[`configs/model_registry.yaml`](configs/model_registry.yaml) — see
[docs/model_registry.md](docs/model_registry.md).

### `data/processed/model_profiles.parquet` — one row per `model_id`

`model_id, model_name, provider, family, version, has_curated, feature,
profile_text, profile_version, n_observations, n_datasets, structured (json),
empirical (json)`. `feature` = curated description; `profile_text` = curated +
empirical behaviour summary. See [docs/model_profiles.md](docs/model_profiles.md).

### `data/processed/nirt_observations.parquet` — one row per training example

`query_id, model_id, target, score_raw, metric_type, source, split, cost,
is_multiple_choice, n_choices`. Correctness signal only, **no embeddings** —
`NIRTDataset` joins those by id. See [docs/nirt_dataset.md](docs/nirt_dataset.md).

### `data/processed/embeddings/<kind>__<pathway>/`

`kind ∈ {query, model_profile}`, `pathway ∈ {retrieval, irt}`. Each dir:
`vectors.npy` (float32 `n × dim`, memmap), `ids.parquet` (`<query_id|model_id>
→ row`), `manifest.json` (backend, model, pooling, dim, `complete`, fingerprints).
Builds are streaming + **resumable** (`_progress.json` mid-run).

### `data/splits/*.json`

`train.json`, `validation.json`, `test.json` (explicit `query_ids` + params),
`cold_start_models.json` (held-out `model_ids`).

---

## Running Phase 0

### One shot (deterministic, non-LLM parts)

```bash
python -m router.phase0
```

### Stage by stage

```bash
# 1. obtain raw data (idempotent; download_arena.py also fetches judge battles)
python scripts/data/download_routerbench.py
python scripts/data/download_arena.py
python scripts/data/run_lm_harness.py                     # prints the eval command (default)

# 2. normalize everything → data/processed/*.parquet  (+ quality report)
python scripts/data/build_response_matrix.py
#    subset: --sources routerbench,chatbot_arena     strict: --strict

# 3. deterministic splits → data/splits/*.json  (verifies no leakage)
python scripts/data/build_splits.py

# 4. LLM profiles → data/processed/model_profiles.parquet
python scripts/models/build_model_profiles.py                  # --print to see them

# 5. NIRT training table → data/processed/nirt_observations.parquet
python scripts/data/build_nirt_dataset.py

# 6. embeddings → data/processed/embeddings/<kind>__<pathway>/  (streaming + resumable;
#    re-run the same command to continue an interrupted build)
python scripts/embeddings/build_query_embeddings.py   --pathway retrieval          # full corpus, MiniLM
python scripts/embeddings/build_query_embeddings.py   --pathway irt --from-nirt    # ~36.5k queries, BERT
python scripts/embeddings/build_profile_embeddings.py --all-pathways
```

Or run stages 1-5 (deterministic parts) at once:

```bash
python -m router.phase0 --embeddings        # omit --embeddings to skip the slow encode
```

### How the response matrix is built

`build_response_matrix.py` → loaders → `coerce_response_frame` (canonical
columns/dtypes) → `collapse_duplicate_observations` (exact
`(query_id, model_id, metric_type)` duplicates averaged, fan-in in
`metadata.collapsed_from`; pairwise metrics exempt) → `apply_chance_correction`
→ `queries` / `models` tables → `check_responses` (errors abort, warnings
continue) → parquet.

Build `R[q, m]` for a single metric (or use the Phase 1 facade below):

```python
from router.config import load_config
from router.data.response_matrix import read_responses, pivot
df = read_responses(load_config())
R = pivot(df, "mc_accuracy", score_column="corrected_score")   # rows=query_id, cols=model_id
```

### Phase 1 data facade

Phase 1 code imports **only** `router.data.phase1`:

```python
from router.config import load_config
from router.data.phase1 import load_phase1

d = load_phase1(load_config())
R      = d.correctness_matrix(split="train", combine_metrics=["accuracy", "mc_accuracy"])
long   = d.correctness(split="train", score_kind="effective")   # + score_raw / score_corrected
pairs  = d.pairwise(source="chatbot_arena", split="train")      # (q, m_a, m_b, winner, score_a)
judge  = d.pairwise(source="gpt4_judge", split="train")
q_emb  = d.query_embeddings(pathway="irt")                      # query_id -> BERT vector
m_emb  = d.profile_embeddings(pathway="irt")                    # model_id -> profile vector (theta_m init)
prof   = d.model_profiles()                                     # text + empirical facts per model
ds     = d.nirt_dataset(split="train", pathway="irt")           # -> {query_embedding, model_embedding, target}
for batch in ds.dataloader(batch_size=256): ...                 # torch DataLoader, collate provided
warm, cold = d.warm_model_ids(), d.cold_start_model_ids()
d.summary()
```

`score_kind`: `effective` (chance-corrected where applicable, else raw — the
default), `raw`, or `corrected`. `models`: `warm` (default, excludes cold-start),
`all`, or `cold`.

### Multiple-choice chance correction

**No guessing parameter, fitted or fixed** (confirmed 2026-08-27). Data-level
rescaling `corrected = (score − 1/n) / (1 − 1/n)`, clipped, applied per
observation only where `n_choices` is known; original `score` preserved;
unknown-`n` MC rows left alone with a warning. Full write-up + per-model bias
diagnostic: [docs/chance_correction.md](docs/chance_correction.md). Which tasks
are MC (and their option counts) is set in `configs/phase0.yaml → multiple_choice`.

### How train/test splits work

* Split unit is a **content-hash group**: all `query_id`s whose normalized text
  hashes equal are assigned to the same split, so identical queries from
  different datasets never straddle train/test.
* Assignment = deterministic hash of `(seed, group_key)` into fraction buckets.
* **Cold-start models**: `split.cold_start_model_fraction` of eligible models
  (≥ `cold_start_min_observations` rows) are held out of *all* of
  train/val/test, stratified per source, chosen by hashed rank. They exist only
  in `cold_start_models.json` for later profile-based `theta_m` eval.
* `router.data.splits.check_leakage` asserts empty train∩val∩test.

### Encoder pathways + LLM profiles

[`src/router/embeddings/encoder.py`](src/router/embeddings/encoder.py) — named
**pathways** under `configs/phase0.yaml → embedding.pathways`:

* **`retrieval`** — `sentence_transformer` / `all-MiniLM-L6-v2`, normalized. For
  kNN and the (deferred) FAISS bank.
* **`irt`** — `bert` / `bert-base-uncased`, mean-pooled, un-normalized. The
  NIRT text pathway: encodes **queries and LLM profile texts** into one 768-d
  space so Phase 1 derives `(b_q, a_q, r_q)` from the query vector and initialises
  `theta_m` from the profile vector.

Deterministic inference (eval, `no_grad`, seeded), **not** fine-tuned. Backends:
`bert` (`cls`/`mean` pooling, optional L2) and `sentence_transformer`.
[`src/router/models/profiles.py`](src/router/models/profiles.py) builds the
profile text (curated `configs/model_profiles.yaml` + optional empirical
behaviour summary). Details:
[docs/embedding_pathways.md](docs/embedding_pathways.md),
[docs/model_profiles.md](docs/model_profiles.md).

### Taxonomy / clustering / relevance / FAISS

Built for the Phase 1 baseline. `router.taxonomy` clusters the 36.5k NIRT queries
(UMAP→HDBSCAN, retrieval embeddings) into **39 ability clusters**, labels each
from its dominant benchmark family + top TF-IDF terms, and derives the relevance
vector `r_q ∈ R³⁹` (`softmax(cos(e_q, centroid)/τ)`, `sum=1`, `≥0`).
`router.retrieval` builds the FAISS **query bank** over the train split only
(`retrieval.k`) and the pre-computed warm-up representations. LLM cluster
labelling stays deferred as an opt-in refinement. Full write-up:
[docs/query_representation.md](docs/query_representation.md).

---

## Configuration

Everything experimental is in [`configs/phase0.yaml`](configs/phase0.yaml):
`seed`, `paths`, per-source locations, `multiple_choice` task map,
`chance_correction`, `embedding.pathways`, `profiles` (curated file +
`include_empirical` + `task_families`), `clustering` (UMAP + HDBSCAN),
`retrieval.k`, `split` fractions. Plus [`configs/model_registry.yaml`](configs/model_registry.yaml)
and [`configs/model_profiles.yaml`](configs/model_profiles.yaml). Source code
does not hard-code these. Pass `--config path/to.yaml` to any script.

The NIRT model has its own [`configs/nirt.yaml`](configs/nirt.yaml) (model + training
hyperparameters); the data pipeline stays in `phase0.yaml`.

---

## Tests

```bash
pytest -q
```

`tests/test_data.py` (schema, id determinism, alias-merge confidence policy,
quality checks, duplicate collapse, metric preservation, `R[q,m]` separation,
RouterBench smoke), `tests/test_chance_correction.py` (formula, non-MC
untouched, missing-`n` warns, score preserved & in range),
`tests/test_splits.py` (determinism, no leakage, content-identical queries
co-located), `tests/test_phase1.py` (facade), `tests/test_embeddings.py` (both
backends, determinism, **streaming build + resume-from-checkpoint**, memmap
load), `tests/test_profiles.py` (empirical stats, curated vs template,
`include_empirical` toggle), `tests/test_nirt.py` (observation schema has no
vectors, target `score_kind`, dataset joins by id without copying, drops
missing-embedding rows, `gather`/`collate`), `tests/test_nirt_model.py` (forward
shapes / range, softplus discrimination, known-value metrics, **synthetic
recovery** of `theta_q`, `fit` + checkpoint round-trip, determinism).
`tests/test_nirt_model.py` also covers the classical-IRT baseline and the routing report.
`tests/test_taxonomy.py` (UMAP→HDBSCAN determinism, `r_q` is a distribution,
temperature control), `tests/test_retrieval.py` (FAISS bank save/load/kNN,
self-exclusion), `tests/test_baseline.py` (BaselineNIRT shapes, IRT equation,
ablation switches, warm-up blend), `tests/test_training.py` (BCE decreases,
**synthetic multidim-IRT recovery**, seed reproducibility, no cold-start-model
leakage), `tests/test_metrics.py` (Brier / log-loss / reliability / ECE on toy
inputs), `tests/test_diagnostics.py` (theta spectrum collapse detection,
effective rank, ICC monotonicity). Phase 2 adds `tests/test_response_heads.py`
(shared `ResponseHead` API + factory + `BaselineNIRT` wiring),
`tests/test_continuous_{normal,beta,zoib}.py` (positivity, analytic likelihood,
boundary handling, synthetic recovery) and `tests/test_phase2_evaluation.py`.
**280+ tests** (`pytest -q`).

---

## Example end-to-end

```bash
pip install -e ".[dev]"
python scripts/data/build_response_matrix.py
python scripts/data/build_splits.py
pytest -q
python -c "from router.config import load_config; from router.data.response_matrix import read_responses; print(read_responses(load_config()).groupby('metric_type').size())"
```

Current outputs from the checked-in raw data:

```
734,469 observations   196,791 queries   69 models   88 datasets
by metric : mc_accuracy 294,998 | judge_preference 218,202 | arena_preference 114,954 | accuracy 106,315
by source : routerbench 401,313 | gpt4_judge 218,202 | chatbot_arena 114,954
correctness 401,313  |  pairwise 333,156
train / val / test queries : 157,271 / 19,672 / 19,767
cold-start models : claude-2.0, code-llama-34b-instruct, vicuna-13b, yi-34b-chat
LLM profiles : 69 (20 curated)   |   irt pathway: bert-base-uncased 768-d
NIRT observations : 328,347  (train 262,548 / val 32,706 / test 33,093; 9 warm models, 36,483 queries)
```

---

## Known limitations / decisions needing human review

* **Pairwise sources are relative, not absolute.** Arena + Judge kept as
  `*_preference` rows with the opponent in metadata and returned separately by
  the facade; how Phase 1's preference heads consume them (BT layer vs win-rate
  vs held-out eval) is an open modeling decision —
  [docs/arena_interpretation.md](docs/arena_interpretation.md).
* **GPT-4 Judge is one matchup.** All 109k battles are `gpt-4-1106-preview` vs
  `mixtral-8x7b-instruct`, so `judge_preference` only constrains those two
  models' relative placement per query.
* **RouterBench has no token counts** (only cost) → `input_tokens` /
  `output_tokens` null for 100% of rows. lm-harness runs would add real ones.
* **`performance` is graded** in `{0.0 … 1.0}`, not binary. Left as-is; the
  facade exposes `score_raw` / `score_corrected` / `score_effective` so Phase 1
  can ablate binary vs ordinal vs continuous (without ZOIB / heteroskedastic
  machinery, which is out of scope).
* **MC task map is partial.** Only tasks in `configs/phase0.yaml → multiple_choice`
  get correction; others warn. Review as tasks are added.
* **Model registry cross-source merges** are conservative — several probably-safe
  merges (`codellama-34b-instruct`, `wizardlm-13b`) are kept separate pending
  review. See [docs/model_registry.md](docs/model_registry.md).
* **Cold-start split** holds out whole models by hashed rank per source; a
  coverage-aware selection may be worth adding.
* **Embedding stores**: all four are built and `manifest.json → complete` — including
  `query__irt` (BERT, 36,483 × 768, scoped to the NIRT queries) and `model_profile__irt`
  (69 × 768). The BERT query pathway is slow on CPU (~1–2 h for the full corpus); the builder
  streams to a memmap and resumes from `_progress.json`, so re-running the same command
  continues an interrupted build.
* **Model profiles**: only 20 / 69 have curated descriptions (RouterBench
  candidates + high-traffic Arena models); the rest use a registry template +
  empirical addendum. No LLM-generated descriptions yet.
* **Profile empirical addendum** groups tasks with a coarse fixed
  `profiles.task_families` map — deliberately *not* the learned taxonomy.
