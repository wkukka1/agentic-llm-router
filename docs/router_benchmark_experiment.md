# Router bake-off: every router, same data, in- and out-of-domain

Status: **step 1 run** (2026-11-12/13): primary split, seed 0, plus the RouteLLM and LiteLLM
2-model sub-study. Results: `investigations/router_bakeoff/primary/headline.md`.

## Question

On identical training data, which router picks the best model per query at a
given cost, both in-domain and on datasets it never saw? Our routers (NIRT, the
Elo-manifold NIRT, the IRT-Router re-implementation, kNN, MLP, Bernoulli/ZOIB)
have only been compared against each other. This adds published routers and
scores everything with **routing** metrics, not BCE.

## Why not BCE

* The target is graded, not binary: `squad`, `hotpot_qa` (F1) and `mbpp`
  (pass rate) score in [0, 1], and multiple-choice targets are chance-corrected.
  BCE with soft targets is still a proper score, but its floor is the target's
  own entropy, not 0. So a BCE of 0.50 means different things on different
  datasets and can't be compared across them.
* Many external routers (kNN/SVM/MLP classifiers, RouterDC, GraphRouter) output
  a model **choice** or a class score, not P(correct). They have no BCE at all.
* What matters is the decision. A router can be better calibrated yet route
  worse (`docs/routing_evaluation.md`).

So the headline metrics are computed from **what the router chose** and the
**observed graded score** of that choice. Predictive metrics are secondary and
only reported for routers that output per-model quality.

## Data and splits (fixed for every router)

Primary benchmark: **IRT-Router** (`configs/irt_router.yaml`). It has 20 models,
graded performance and real per-call cost.

| split | source | queries | used for |
|---|---|---|---|
| train | `train.csv`, 90 % | 22 011 | fitting |
| validation | `train.csv`, 10 % | 2 403 | hyperparameters and early stopping, the **only** tuning data |
| test (ID) | `test1.csv` | 10 458 | in-domain report, 8 datasets |
| ood | `test2.csv` | 3 159 | out-of-domain report: `ceval, commonsense_qa, gsm8k, humaneval` |

The same query ids are used for every router. They come from
`data/splits/irt_router/*.json`, which external routers receive through an
exporter (below).

### Out-of-domain protocols

1. **Held-out datasets (primary).** Train on the 8 ID datasets; test on the 4
   OOD datasets. Nothing is tuned on OOD.
2. **Leave-one-dataset-out (LODO).** 8 folds: train on 7 ID datasets, test on the
   held-out one (its `test` rows). This gives 8 more OOD points, so OOD
   conclusions don't rest on 4 datasets. Validation rows of the held-out dataset
   are dropped too.
3. **Held-out model (cold start), separate track.** Only routers that take model
   features (NIRT projected / Elo manifold, GraphRouter) can do this. It is the
   existing `elo_manifold_lomo.py` protocol, scored with the metrics below.

Secondary benchmark (if time allows): **LLMRouterBench** perf-cost setting
(`docs/llmrouterbench.md`, 13 frontier models, 10 datasets), with LODO only.

## Routers

### Ours (in repo)

| id | what |
|---|---|
| `nirt` | NIRT `query_latent`, K=2, projected (`irtrouter-nirt-2d-projected`) |
| `nirt_elo_manifold` | `configs/nirt_elo_manifold.yaml` (Elo manifold + per-model discrepancy) |
| `irt_router_k25` | the paper's `model_latent` K=25 (our re-implementation) |
| `bernoulli`, `zoib` | continuous response models (`train_continuous.py`) |
| `knn`, `mlp` | `router.routing.routers.KNNRouter`, `MLPRouter` |

### External: `llmrouter-lib` 0.4.0 (ulab-uiuc LLMRouter, from PyPI)

GitHub is blocked in this environment, but PyPI isn't. The wheel is downloaded
and its source has been read. Runnable on 4 CPUs with no GPU:

| id | method | output |
|---|---|---|
| `ext_knn` | kNN over query embeddings | per-model score |
| `ext_svm` | SVM classifier | class score |
| `ext_mlp` | MLP classifier | class score |
| `ext_mf` | RouteLLM-style bilinear matrix factorization | per-model score |
| `ext_elo` | Elo from pairwise wins (query-blind) | per-model score |
| `ext_graphrouter` | GraphRouter (GNN over query/task/model graph; needs `torch-geometric`) | per-model score |
| `ext_routerdc` | RouterDC (dual contrastive) **with the encoder frozen**. Full encoder fine-tuning only if a GPU becomes available. | per-model score |

Excluded, with reasons:
* `causallm_router` and `router_r1`: need GPU / vLLM.
* `automix`: needs live LLM self-verification calls.
* `hybrid_llm`: strong/weak binary only. It goes to the 2-model sub-study below.
* `*multiround`, `personalizedrouter`, `gmtrouter`, `tsrouter`: a different task
  setting (multi-turn, per-user or online).

### External: RouteLLM 0.2.0 (PyPI), 2-model sub-study

RouteLLM routes between one strong and one weak model. It runs in its own venv
(`scripts/router_bench/routellm_scores.py` already does this) as
**gpt_4o vs gpt_4o_mini**, in two arms:
* (a) its **pretrained** Arena routers, off the shelf;
* (b) `ext_mf` and `hybrid_llm` retrained on our data.

Every 20-model router is also restricted to those two columns, so all are
compared on the same binary decision.

### Reference policies (no learning, or an upper bound)

* `oracle`: cheapest model with the best observed score.
* `random`.
* `best_single`: the best model on train, `deepseek_coder`.
* `cheapest`, and `gpt_4o` as the cost reference.
* `per_dataset_best`: picks the best train model for the query's dataset. This
  uses the dataset label, so it is a "task-aware" upper bound for any router
  that only learns which dataset a query belongs to.

## Fairness controls

1. **Same rows.** Every router trains on exactly the train split's (query,
   model, score, cost) rows and tunes on validation only.
2. **Same features.** All routers get the same 768-d BERT query embeddings
   (`query_embeddings.pkl`), so the comparison is between routing methods, not
   encoders. llmrouter-lib defaults to Longformer; its `text_dim=768` takes ours
   unchanged. A second arm lets each router use its native encoder, to show how
   much of any gap is the encoder.
3. **Same cost.** Per-model cost is the train-split mean cost per call
   (`routing._mean_train_costs`), and all routers share one utility:
   `OptimizationObjective.utility`, i.e. `pred − λ·C(m)`.
4. **Same tuning budget.** Each router gets 20 random-search trials on
   validation. The selection criterion is **validation AIQ**, not loss.
5. **Seeds.** 3 seeds per (router, protocol). The report gives the mean ± the
   spread across seeds.

### Turning each router into a cost/quality curve

* **Routers with per-model scores** (NIRT, IRT, kNN, MF, Elo, GraphRouter,
  RouterDC, Bernoulli/ZOIB): sweep λ over a shared grid of 41 log-spaced points
  from 0 to λ_max, where λ_max makes the cheapest model optimal for 99 % of
  queries. This gives the frontier from one fit.
* **Choice-only classifiers** (SVM, MLP-classifier): their training label is
  `argmax_m (y − λ·C)`. Retrain them at 5 λ values from the same grid; each fit
  is one point on the frontier.
* **RouteLLM / binary:** sweep the threshold (strong-model share 0–100 %).

## Metrics

> **Revised after step 1 (2026-11-13).** A *query-blind* policy gives every query each model's
> train-mean score, plus the same cost trade-off. It reaches 0.751 of the 20-model macro log-cost
> AIQ in-domain, against 0.759 for the best router and 0.920 for the oracle. So that AIQ mostly
> rewards knowing model averages and costs, not per-query routing. The headline is now **per-query
> skill over the query-blind baseline**:
> * `headroom` = (AIQ − AIQ_blind) / (AIQ_oracle − AIQ_blind), with a paired-bootstrap CI on AIQ − AIQ_blind;
> * Δ per-query Spearman / NDCG@3 against the blind ranking (the same model order for every query), with CIs;
> * the 2-model gpt_4o-vs-gpt_4o_mini **APGR** (RouteLLM's metric), where query-blind = random ≈ 0.5,
>   with CIs vs random.
>
> Raw 20-model AIQ and the metrics below remain as secondary columns. Report:
> `scripts/bench/bakeoff_report.py` → `headline.md`.

All are computed on the observed **graded** score `y(q, m)`, using the existing
`evaluation.routing.oracle` and `evaluation.nirt.routing` code.

**Primary (decision quality)**
1. **AIQ**: the area under the quality-vs-cost frontier, normalized over the
   pool's cost range (`routing.aiq`). This is the one-number summary.
2. **Quality at fixed cost**: mean graded score when spending (a) gpt_4o's
   cost, (b) 25 % of it, (c) 5 % of it, interpolated on the frontier.
3. **Cost at fixed quality**: the cost needed to match `gpt_4o`'s quality and
   `best_single`'s quality.
4. **Regret at λ=0**: `y(q, oracle) − y(q, chosen)`, reported as the mean and
   the p90, plus the **tie-aware best-hit rate**
   (`oracle_hit_rate_any_best`).

**Macro over datasets.** Each metric is computed per dataset, then averaged,
so `mmlu` / `cmmlu` (most of the rows) don't decide the result. The
row-weighted number is reported next to it.

**Secondary (ranking quality, score-producing routers only)**
* Per-query **Spearman / NDCG@3** of the predicted model ranking against the
  observed scores. This asks whether the router orders models correctly, which
  is what routing needs, and it works for graded targets.
* **Brier / MSE** on the graded target, plus a calibration plot. These replace
  BCE.

## Statistics

* **Paired bootstrap over queries** (2 000 resamples, stratified by dataset)
  for the difference between each router and the best of ours: 95 % CI on
  ΔAIQ and Δquality@cost. All routers are scored on the same queries, so
  pairing removes per-query difficulty noise.
* **Holm correction** across routers within a protocol.
* LODO: report the per-fold wins and a **Wilcoxon signed-rank** test over the
  8 folds.
* A difference counts as real only if its CI excludes 0 **and** it holds
  across all 3 seeds.

## Plumbing (no new `src/` classes)

* `scripts/bench/export_llmrouter.py`: writes the IRT-Router splits into
  llmrouter-lib's format:
  * `query_data_{train,test}.jsonl`
  * `routing_data_{train,test}.jsonl` (query, model, performance, cost)
  * `llm_data.json`
  * embeddings `.pt`

  It does this per protocol and fold.
* `scripts/bench/run_llmrouter.py`: runs in an **isolated venv**
  (`.venv-llmrouter`, `torch-geometric` + the lib). It is never imported by
  the repo, the same pattern as `routellm_scores.py`. For each router, fold
  and seed it writes a long table: `query_id, model_id, score` (per-model
  routers) or `query_id, lambda, chosen_model_id`.
* `scripts/bench/router_bakeoff.py`: loads our routers through
  `router.routing` and the external tables from disk, then computes every
  metric and bootstrap above. It writes
  `investigations/router_bakeoff/{results.csv, frontiers.csv, summary.md}`.

## Compute (4 CPUs, 15 GB, no GPU)

Rough estimates, to be confirmed by a 1-seed smoke run:

| item | per fit | fits (3 seeds × [1 primary + 8 LODO]) |
|---|---|---|
| NIRT / Elo manifold | ~45 min | 27 each → too much for all LODO folds; run LODO at 1 seed |
| IRT K=25, Bernoulli, ZOIB | 10–30 min | 27 each |
| kNN / MF / Elo / SVM / MLP (ext + ours) | < 5 min | trivial |
| GraphRouter, RouterDC-frozen | ~15–30 min | 27 each |

Order of work:
1. Exporter + venv + 1-seed primary (ID/OOD) for all routers, which gives the
   first comparison table.
2. Then 3 seeds.
3. Then LODO.
4. Then the RouteLLM sub-study.

Background jobs are capped at 2 h, so every runner writes a resumable CSV, as
`elo_manifold_lomo.py` does.

## What would change our conclusions

* If `per_dataset_best` is within the CI of the learned routers on ID, the
  routers are mostly learning the dataset label. Then OOD and LODO are the
  only informative tests.
* If ranking metrics favour one router but AIQ favours another, the cost term
  dominates. Report quality at fixed cost to separate the two.
