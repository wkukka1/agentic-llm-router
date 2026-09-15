# NIRT model + capacity workstream (Phase 2)

[`src/router/nirt/`](../src/router/nirt/) — the response model, kept separate from
[`src/router/data/nirt.py`](../src/router/data/nirt.py) (the training
*representation*, see
[query_and_model_representation.md](query_and_model_representation.md#3-nirt-training-representation)).
Config: [`configs/nirt.yaml`](../configs/nirt.yaml). The frozen Bernoulli /
continuous control arm is separate again — see
[baseline_control_arm.md](baseline_control_arm.md).

---

## Formulation — two orientations

Same bilinear IRT core; `model.orientation` picks which side carries the latent
ability:

```
query_latent  (ours, default)          model_latent  (IRT-Router, Song et al. ACL'25)
  theta_q = f_theta(e_q)                  theta_m = [sigmoid] W_theta e_m
  (a_m, b_m) on the model                 a_q = W_a e_q ,  b_q = W_b e_q
  y = sigmoid(a_m . theta_q - b_m)        y = sigmoid(a_q . theta_m - b_q)
```

- **`query_latent`** ([`NIRTModel`](../src/router/nirt/model.py)) — the latent
  vector is amortised from the query text; the model side is a small `(a_m, b_m)`
  table (`model_params=free`) or a projection of the profile embedding
  (`projected`, cold-start capable). No explicit per-query difficulty.
- **`model_latent`** ([`IRTRouterModel`](../src/router/nirt/model.py)) — the
  classical-IRT reading: the LLM is the examinee with an **ability vector
  `theta_m`**, the query is the item with **discrimination `a_q` and difficulty
  `b_q`** (both amortised from `e_q`). Matches the README design and yields
  reusable per-LLM ability vectors. `free` == the paper's M-IRT.

`e_q` is a frozen `bert-base-uncased` mean-pooled embedding (irt pathway, 768-d).
Training minimises soft-label BCE against the graded RouterBench target
(`performance ∈ {0,.25,.5,.75,1}`). For `K = 1` discrimination is
softplus-constrained (`> 0`, like 2PL); unconstrained for `K > 1`.

### A/B result (test split, dim 2, projected — `scripts/nirt/ab_orientation.py`)

| orientation | test BCE | AUC | Spearman | ranking optimal-rate | cost @ −3 acc pts ($/1k) |
|---|--:|--:|--:|--:|--:|
| classical IRT 2D (ceiling) | 0.593 | 0.79 | 0.49 | 0.81 | 1.99 |
| **model_latent** (IRT-Router) | **0.592** | 0.75 | 0.45 | 0.81 | 2.18 |
| query_latent (ours) | 0.594 | 0.76 | 0.43 | 0.81 | 2.85 |

`model_latent` wins narrowly at 2D and clearly at 1D (BCE 0.603 vs 0.614) — the
explicit per-query difficulty `b_q` helps, and it gives a better cost/quality
frontier. Both sit ~0.02 BCE above the matrix-factorisation ceiling and produce
the *same* routing decision at λ=0 (always-GPT-4). Default stays `query_latent`;
the evidence favours switching.

## What v1 is / is not

v1 is a **predictor**, evaluated on prediction quality only:

| metric | meaning |
|--------|---------|
| `bce`, `mse`, `mae` | error vs the soft target |
| `pearson_r`, `spearman_r` | correlation of `y_hat` with observed correctness |
| `auc` | ranking of correct vs incorrect (target binarised at 0.5) |
| `ece` | calibration (10-bin, against the soft outcome rate) |
| `acc@0.5` | agreement after thresholding both sides at 0.5 |

Every run is compared against two **marginal baselines**: `global_mean` (overall
training correctness rate) and `model_mean` (each model's mean training
correctness, query-independent). Beating `model_mean` is the bar — it means the
query head learned something query-specific.

**Built** — dual orientation + A/B, classical-IRT baseline (both protocols),
per-query ranking metrics, cost-aware routing with a λ sweep, the cold-start
experiment, and the full capacity workstream (P1–P6, below).

## Config (`configs/nirt.yaml`)

| key | |
|-----|--|
| `data.pathway` | embedding pathway to join against (`irt` = BERT) |
| `data.query_pathway` | override only the *query* store (kNN-imputed representation) |
| `data.query_features` | structured per-query feature vector concatenated to `e_q` (`default` = the 10-d RouterBench set); `null` = off |
| `data.score_kind` | target column (`effective` = chance-corrected where MC) |
| `model.orientation` | `query_latent` (ours) \| `model_latent` (IRT-Router) |
| `model.dim` | `K` — latent ability dimensions |
| `model.model_params` | `projected` \| `free` |
| `model.difficulty` | `scalar` (`b_m ∈ R`) \| `vector` (`b_m ∈ R^K`) |
| `model.query_hidden` | query-head hidden width; `null` → single linear layer |
| `model.query_head_{layers,norm,activation,dropout,residual}` | **P2a** query-head capacity. Defaults reproduce the legacy `Linear→ReLU→Linear` head bit-for-bit |
| `model.model_hidden` | **P2b** `projected` only: `a_m`/`b_m` via a 1-hidden-layer MLP; `null` → legacy linear |
| `model.interaction` / `model.interaction_hidden` | **P2c** `query_latent` only: `gamma · MLP([θ_q, a_m, θ_q·a_m])` added to the bilinear logit (`gamma` init 0); `false` → strictly bilinear |
| `model.constrain_discrimination` | `auto` → true iff `dim == 1` |
| `model.bound_ability` | `model_latent` only: `theta_m = sigmoid(W_theta e_m)` |
| `train.loss` | `soft_bce` \| `hard_bce` \| `mse` \| `pairwise` \| `listwise` \| `bce_pairwise` |
| `train.sampler` | `cell` (legacy per-obs shuffle) \| `query` (whole query groups; auto-forced by a grouped loss) |
| `train.val_metric` | `bce` \| `regret` (early-stop on mean per-query routing regret) |
| `train.{lr,weight_decay,batch_size,epochs,patience}` | Adam + early stop |
| `train.grad_clip` / `train.lr_schedule` | **P2** max grad-norm (`0` = off), schedule (`none` \| `cosine` \| `plateau`) |
| `train.preference` | auxiliary Bradley-Terry loss over Arena / Judge battles (off by default — see below) |
| `train.materialize` | gather the split into RAM once (fast on CPU) vs stream |
| `runs_dir` | `data/processed/nirt_runs` (gitignored) |

## Run

```bash
python scripts/nirt/train_nirt.py --config configs/nirt.yaml --dim 2 --model-params projected --name nirt-2d-projected
python scripts/nirt/eval_nirt.py  --run nirt-2d-projected --split test
python scripts/nirt/compare.py    --run nirt-2d-projected --split test    # vs classical IRT + routing/cost

# the other orientation (IRT-Router: theta on the LLM)
python scripts/nirt/train_nirt.py --orientation model_latent --dim 2 --model-params projected --name irt-2d-projected
python scripts/nirt/ab_orientation.py --dim 2 --model-params projected --reuse

# cold-start: place code-llama-34b / yi-34b from their profile text alone
python scripts/nirt/coldstart.py --run nirt-2d-projected --split test
```

Each run writes `<runs_dir>/<name>/`: `model.pt` (`state_dict` + config +
`model_index`), `run.json` (config, git sha, per-epoch history, best val metrics,
baselines), and — after `eval_nirt.py` — `eval_<split>.json`.

```python
from router.nirt import fit, load_run
res = fit(yaml.safe_load(open("configs/nirt.yaml")))          # RunResult
model, cfg, model_index = load_run("nirt-1d-projected")
```

## Classical IRT & routing

[`classical_irt.py`](../src/evaluation/baselines/classical_irt.py) fits a
classical (multidimensional) 2PL with **no query features** — free per-query
`theta_q`, per-model `a_m`, `b_m`:

* **transductive ceiling** (`protocol="cell"`) — fit on a random 85 % of the
  split's `(query, model)` cells, predict the held-out 15 %. The
  matrix-factorisation ceiling. `theta_q` is L2-regularised (`weight_decay=1e-3`).
* **`theta_q = 0` lower bound** (`protocol="main_effects"`) — fit `a_m`, `b_m` on
  train+val, predict test queries as `sigmoid(-b_m)` (no query information).

[`routing.py`](../src/evaluation/nirt/routing.py) turns any predictor into a policy
`m* = argmax_m (pred − λ · cost_norm)` and scores it on the dense test matrices.

### Results (test split — 3,677 queries, 9 warm RouterBench models)

**Prediction.** NIRT (text only, never sees a response) matches the transductive
MF ceiling on BCE:

| method | BCE | MSE | AUC | acc@0.5 | Spearman |
|--------|----:|----:|----:|--------:|---------:|
| NIRT-2D projected (inductive) | **0.594** | 0.162 | 0.762 | 0.702 | 0.43 |
| classical IRT 2D — transductive ceiling | 0.593 | 0.158 | 0.794 | 0.727 | 0.49 |
| classical IRT 2D — `theta=0` lower bound | 0.645 | 0.183 | 0.690 | 0.664 | 0.30 |
| baseline: per-model mean | 0.645 | 0.183 | — | — | — |
| baseline: global mean | 0.690 | 0.205 | — | — | — |

NIRT-1D lags slightly (BCE 0.614); 2D and 4D both land at ~0.592–0.594. Extra
dimensions past 2 do not help here.

**Routing + cost.** The warm pool is **GPT-4-dominated** — `gpt-4-1106-preview`
is the (a) best model on 84 % of test queries at ~1.4× the mid-tier Claudes — so
**quality-only routing (λ=0) collapses to "always GPT-4"** for *both* NIRT and
the classical ceiling. The value is on the cost/quality frontier:

| policy | quality | accuracy | $/1k queries | vs GPT-4 |
|--------|--------:|---------:|-------------:|----------|
| oracle (best per query) | 0.907 | 0.958 | 0.83 | −76 % cost, +12 pts acc |
| fixed: GPT-4 | 0.772 | 0.837 | 3.45 | — |
| fixed: cheapest (mistral-7b) | 0.302 | 0.290 | 0.05 | −99 % cost, −55 pts acc |
| NIRT route, λ=0 | 0.772 | 0.837 | 3.44 | ≈ GPT-4 |
| NIRT route, λ=0.5 | 0.768 | 0.826 | 2.85 | −17 % cost, −1 pt acc |
| NIRT route, λ=1.0 | 0.759 | 0.807 | 1.60 | **−54 % cost, −3 pts acc** |
| NIRT route, λ=3.0 | 0.710 | 0.755 | 0.82 | −76 % cost, −8 pts acc |

**Takeaway.** On prediction the frozen-BERT query head already reaches the MF
ceiling — the bottleneck is not the IRT structure. On routing, a single dominant
model means little quality headroom; the win is a real but modest cost reduction
(≈half the spend at −3 accuracy points). RouterBench is not a useful routing
testbed — the capacity workstream below uses the IRT-Router 20-model suite (see
[irt_router.md](irt_router.md)) for the routing decision.

## Cold-start — placing a held-out LLM from its profile

`code-llama-34b-instruct` and `yi-34b-chat` are held out of *all* training. A
`projected` run predicts their per-query correctness on the test split from their
**profile-text embedding only**.

### Result — it does not work yet (test split, pooled over both cold LLMs)

| predictor | BCE | AUC |
|---|--:|--:|
| `warm_mean` (no NIRT) | **0.69** | 0.61 |
| `global_mean` (no NIRT) | 0.71 | 0.50 |
| NIRT `query_latent` projected | 0.71 | 0.59 |
| NIRT `model_latent` projected | 0.74 | 0.54 |
| `nearest_warm` (no NIRT) | 0.76 | 0.46 |

Both orientations land at or **below** the trivial `warm_mean` baseline. A
768→K linear projection trained on **9 LLMs** cannot learn "this description
implies high knowledge ability". IRT-Router's new-LLM result (0.68 vs 0.34) came
with ~20 training LLMs *and* by calibrating the new model's `theta_m` on a
handful of its actual responses. Description-only cold-start is the harder problem
and needs many more training LLMs, a fine-tuned encoder, or a few calibration
responses. Recorded as a negative result; the same wall shows in the baseline
control arm ([baseline_control_arm.md](baseline_control_arm.md)).

---

# Capacity workstream (P1–P6)

Goal: fix the gap between NIRT and "perfect", with the **routing decision**
(regret / oracle-hit on the IRT-Router test + ood splits) as the success metric,
not prediction BCE alone. Script: `scripts/nirt/capacity_diagnostics.py`.
Artifacts: `artifacts/{phase2,irt_router}/capacity_diagnostics.json`.

> **The headline correction (2026-09-04).** The workstream originally concluded
> the `query_latent` model *underfits* (train ≈ val BCE, flat across a param
> sweep). That was an **early-stopping artifact** — those numbers were all
> measured at the val minimum (epoch ~15), where train and val are close by
> construction. Trained to convergence with no early stopping the model
> **overfits** (RouterBench train BCE 0.59→0.36 while val 0.61→**1.06**). The
> architecture / capacity / encoder levers (P2, P4a) are dead not because the
> model lacks capacity but because it does not *generalise*. The exploitable
> gaps are **data quality** and **objective**.

## P1 — ceiling decomposition

Four blocks, each also reporting routing regret:

1. **Transductive per-query ceiling** (`fit_classical_irt(protocol="cell")`) —
   "how low BCE goes with an **oracle** query representation".
2. **IRT-free capacity bound** (`fit_mlp_router`) — a plain `e_q → R^M` MLP with
   dropout at widths 128 / 512 / 2048. MLP ≫ NIRT ⇒ bilinear logit is a
   bottleneck; MLP plateaus with NIRT ⇒ the frozen `e_q` is the wall.
3. **NIRT reference** — the shipped runs.
4. **Per-family** — BCE / AUC grouped by benchmark family.

### RouterBench (test, 40,447 obs, 9 models)

| predictor | BCE | routing regret |
|---|--:|--:|
| `model_mean` (no query info) | 0.6466 | — |
| **NIRT** `nirt-2d-projected` | 0.6117 | 0.1415 |
| NIRT + best P2 head (clean) | ~0.595 | ~0.140 |
| **MLP-router** (2048-wide + dropout, IRT-free) | **0.5947** | 0.135 |
| **transductive per-query ceiling** (oracle rep) | **0.5362** | — |

**Gap decomposition (NIRT → ceiling = 0.076 BCE):** −0.017 bilinear structure (a
wide MLP beats `a_m·θ_q − b_m` → **P2c** interaction term); −0.059 query
representation (even the 2048-MLP is 0.06 above the ceiling — both see a frozen
mean-pool `e_q` → **P4** stronger encoder, the bigger lever); the ceiling itself
(0.536) is well above 0 → genuine aleatoric noise (graded scores, MC guessing,
0-/5-shot format variance).

**Per-family — where the model is blind:** code BCE 0.714 / **AUC 0.56**
(≈chance), reasoning 0.663 / 0.67, math 0.663 / 0.75, knowledge 0.562 / 0.80,
zh 0.356 / 0.80. Code and reasoning are the underfit families (BERT does not
encode code-task difficulty).

Routing regret is **flat ~0.135–0.142 for every predictor including the 2048-MLP**
— the GPT-4-dominated 9-model pool makes the per-query arg-max
representation-independent.

### IRT-Router 20-model suite (the routing testbed)

**ID test (10,458 q):** `model_mean` BCE 0.561; **NIRT (ours, K=2)** 0.511 /
regret 0.146 / oracle-hit 0.843; **MLP-router (2048)** 0.500 / 0.137; **IRT-Router
paper (K=25)** 0.499 / **0.136** / 0.854; **transductive ceiling** 0.372. Same
shape (MLP > NIRT-ours by 0.011, still 0.13 above the ceiling), but **unlike
RouterBench routing regret is *not* flat** — the K=25 model_latent and the big
MLP route ~1 point better. On a non-dominated pool, capacity *can* help the
routing decision, modestly.

**OOD (3,159 q, held-out datasets):** the learned predictors are **worse than
`model_mean` on BCE** (NIRT 0.606 vs 0.547), and a wider MLP overfits OOD harder
(0.542 → 0.566). Routing regret ~0.12 for all. OOD calibration is a robustness
problem raw capacity does not fix → shrinkage (below).

## Trained-to-convergence behaviour — it OVERFITS

`model_params=free`, K=16, 2-layer head, **no early stopping**, minibatched:

| epoch | RB train | RB val | IRT-R train | IRT-R val |
|---|--:|--:|--:|--:|
| ~15 (early-stop point) | 0.59 | **0.61** (min) | 0.49 | **0.50** (min) |
| ~60 | 0.48 | 0.70 | 0.47 | 0.50 |
| ~120–200 | **0.36** | **1.06** | 0.45 | 0.51 |

RouterBench val BCE *explodes*; IRT-Router's barely moves. Early stopping at
epoch ~15 is near-optimal and is the operative regulariser. Regularisation swept
to convergence (dropout 0.3–0.5, wd 1e-4–1e-3) moves the RouterBench val minimum
by **−0.003 at best**; heavy reg hurts. So val BCE ~0.60 (RB) / ~0.50 (IRT-R) is
a **generalisation ceiling**, not a capacity or optimisation floor.
`head(e_q)` is *not* information-bottlenecked (an earlier probe suggesting this
had an obs-vs-query indexing bug): with correct indexing a plain head fits 100
training queries to BCE 0.072.

## The RouterBench 0-/5-shot confound — the biggest single lever

Pool-expansion E2 made 0-shot and 5-shot versions of a question **separate items
carrying the identical question text** (hence identical `e_q`) with different
outcomes, and shot count is not a feature. The model memorises one twin on train
and its twin contradicts it in val — the mechanism behind the val explosion.

| RouterBench training data / features | best val BCE |
|---|--:|
| 0-shot + 5-shot, `e_q` only (default) | 0.602 |
| **0-shot only** | **0.557** (−0.045) |
| 0-shot + 5-shot, **`e_q` + 1 shot-indicator feature** | **0.569** (−0.033, all data) |

**One binary feature recovers ~all of the 0-shot-only performance and makes the
5-shot regime predictable** — bigger than P2 (−0.016), kNN-impute (−0.002),
encoder swap (0), or regularisation (−0.003) combined. IRT-Router has no such
twins.

**Shipped** (`training.data.query_features`,
`scripts/embeddings/build_query_features.py`, `data.query_features` in
`configs/nirt.yaml`, `NIRTDataset(feature_store=...)`): a small (10-d for
RouterBench) deterministic per-query feature vector — coarse family one-hot,
`is_5shot`, prompt length, `n_choices` — built from `queries.parquet` /
`responses.parquet` alone (no labels, leakage-free) and concatenated to `e_q`.
Default `None` → byte-identical to before. Real-pipeline validation
(`rb-nirt-qf-free`, K=16 free, `query_features: default`): **val BCE 0.604 →
0.572 (−0.032)**, matching the probe.
`train scripts/nirt/train_nirt.py --query-features default`.

## P2 — architecture capacity vs the routing decision

Three orthogonal levers added to `NIRTModel`, **each off by default and
bit-identical to the pre-P2 architecture** (checkpoints load under `strict=True`;
`test_query_head_default_identical` / `test_interaction_off_identical` /
`test_model_hidden_default_and_mlp`):

- **P2a — query head** (`_query_head` / `_MLPHead`): stack depth, LayerNorm, GELU,
  dropout, residual connections.
- **P2b — model side** (`model_hidden`): the `a_m` / `b_m` heads were a linear
  projection of the frozen profile embedding; a 1-hidden-layer MLP is optional.
- **P2c — non-additive interaction** (`interaction`): a gated residual MLP on
  `[θ_q, a_m, θ_q·a_m]` added to the strictly-bilinear logit.

Trained on the IRT-Router suite (`route_compare.py --config configs/irt_router.yaml
--reference gpt_4o --lam 0.3`):

| run | val BCE | ID reward@0.8 | OOD reward@0.8 |
|---|--:|--:|--:|
| `irtrouter-nirt-2d-projected` (K=2 baseline) | 0.513 | 0.640 | 0.684 |
| `irtrouter-nirt-cx-p2` (K=16, 3-layer) | 0.510 | 0.639 | 0.686 |
| `irtrouter-nirt-cx-p2c` (+ interaction) | 0.510 | 0.640 | 0.685 |

**Flat.** P2 architecture capacity does not move the routing decision on *either*
suite (RouterBench regret 0.139–0.141 across 24 configs; IRT-Router reward@0.8
within 0.001 across K=2 / K=16 / K=16+interaction). A clean P2 head/capacity
sweep (`complexity_search.py --n 24`) buys **~−0.016 BCE / +0.02 AUC** on
RouterBench prediction (best config `K16 h128 scalar, 3-layer`, test BCE 0.595)
— the P1 hint that "K=25 routes 1 pt better" is the `model_latent`
**orientation**, not capacity. Latent width `K` beyond ~8 adds nothing
measurable; the nonlinear query head is the whole story.

## P4a — encoder swap

Added a `bge` pathway (`BAAI/bge-small-en-v1.5`, 384-d, MTEB ~62 vs
`bert-base-uncased` mean-pool ~30). `scripts/data/encode_irt_router_pathway.py`
encodes the IRT-Router `query.csv` / `llm.csv` with any pathway.
`all-mpnet-base-v2` was abandoned (5.9 h CPU encode).

| suite | run | test BCE | test regret |
|---|---|--:|--:|
| IRT-Router | `irt`, K=16 **free** | **0.494** | **0.134** |
| IRT-Router | `bge`, K=16 free | 0.494 | 0.138 |
| RouterBench | `irt`, K=16 free | **0.598** | 0.140 |
| RouterBench | `bge`, K=16 free | 0.605 | 0.138 |

**bge-small is a null on both suites.** A modern *small* (33M) sentence encoder
does not close the 0.13 BCE gap to the transductive ceiling. Open: a base-size
encoder (`bge-base`, 109M) or fine-tuning (**P4b**, deprioritised — the model
already overfits).

**The one lever that helped both suites: `model_params: free`** — RouterBench BCE
0.612 → 0.598, IRT-Router 0.511 → 0.494 / regret 0.146 → 0.134. With only 9–20
LLMs, projecting `θ_m` from a short description loses signal.

## kNN-imputed queries

`NIRTModel` (`query_latent`) turns a frozen `e_q` into `theta_q = f_theta(e_q)`.
For a query far from anything in the training set the head is extrapolating —
exactly where it is least trustworthy. This variant replaces `e_q`, **before the
query head sees it**, with a similarity-weighted mean of its `k` nearest
*training* queries:

```
e_q  :=  sum_i w_i * e_{n_i}      n_1..n_k = k nearest TRAIN queries of q
         w_i = max(cos(q, n_i), 0),  normalised   (self-excluded)
```

- **Search / averaging** reuse the Phase-0 retrieval stack
  (`training.retrieval.query_bank.QueryBank`, train-only FAISS, self-exclusion; the
  mean is taken in the `irt` / BERT space).
- **Builder**: `training.retrieval.knn_impute.build_knn_imputed_store(cfg, k=...)`
  writes an ordinary query `EmbeddingStore` under a synthetic pathway name
  (`knn10w` = k 10 similarity-weighted; `knn10u` = uniform; `-ood` suffix for the
  leakage-safe variant, held-out families dropped from the bank).
- **Consumption**: `data.query_pathway` (`configs/nirt.yaml`) overrides **only**
  the query store; threaded through `NIRTDataset.from_config`, `train.fit`,
  `ood.ood_datasets`, `evaluate.predict_matrix`, `route_compare.py`,
  `scripts/nirt/compare.py`. `null` → unchanged.

```bash
python scripts/retrieval/build_query_bank.py
python scripts/retrieval/build_query_bank.py --exclude-ood-families
python scripts/nirt/knn_impute_sweep.py --k 1,5,10,25   # -> artifacts/phase2/knn_impute_sweep.json
```

### Result

> **Correction (2026-09-03).** The original headline — "kNN imputation is a
> clearly better calibrated predictor, BCE 0.612 → 0.579, AUC 0.739 → 0.772" —
> was **largely a training-data confound**: `query__retrieval` (MiniLM) had never
> been built for RouterBench's `:5shot` ids, so the `nirt-knn*` runs trained on
> 0-shot observations only while the baselines trained on 0-shot + 5-shot. After
> rebuilding every store over the full 72,966-query set and **retraining every
> run on the identical observation set** (`knn_impute_sweep.py --retrain`):

| k | ID test BCE (clean) | ID AUC (clean) | OOD BCE (clean) |
|---|--:|--:|--:|
| 0 raw | **0.6117** | 0.739 | 0.692 |
| 10 | **0.6093** | 0.745 | 0.711 |
| 25 | 0.6103 | 0.742 | 0.705 |

**Clean effect: BCE −0.002, AUC +0.006 at k≈10** (was −0.033 / +0.033), and it
still *hurts* OOD calibration (imputing a math/code query from non-math/code
neighbours biases the probability). Routing regret flat everywhere (~0.141).
**Retrieval smoothing of `e_q` is not a meaningful lever** — and as a routing
policy it slightly *hurts* (compresses the between-model spread the arg-max lives
on). Use kNN-imputed `e_q` only when the deliverable is a calibrated
per-(query, model) probability, not the routing decision.

## P5 — routing-aware training

P1 / P2 established that **per-cell BCE capacity does not reach the routing
arg-max**. P5 (`src/router/nirt/train.py`) trains the *decision* instead:

- **`train.sampler: query`** — batches are whole query groups, so a within-query
  loss sees the complete per-query model list.
- **`train.loss`** gains: `pairwise` (RankNet / BT over model pairs within a
  query), `listwise` (ListNet top-1), `bce_pairwise` (`soft_bce +
  loss_aux_weight · pairwise` — calibrated probabilities *and* sharper ranking).
- **`train.val_metric: regret`** — early-stop on mean per-query routing regret,
  so a ranking loss is not penalised for raising BCE.

All default off; `sampler: cell` + `loss: soft_bce` reproduces the legacy path.

### Results — IRT-Router 20-model suite (`model_params=free`, K=16)

| run | test BCE | **test regret** | oracle-hit | ood regret |
|---|--:|--:|--:|--:|
| baseline `irtrouter-nirt-2d-projected` (per-cell BCE, projected) | 0.511 | 0.146 | 0.843 | 0.122 |
| `+ model_params=free` (per-cell BCE) | 0.494 | 0.134 | 0.856 | 0.116 |
| IRT-Router paper K=25 (projected) | 0.499 | 0.136 | 0.854 | 0.124 |
| **`loss=pairwise`** | 0.56 | **0.127** | **0.863** | 0.123 |
| `loss=listwise` | 0.58 | 0.126 | 0.864 | 0.128 |
| **`loss=bce_pairwise`** (w=1) | **0.498** | 0.132 | 0.859 | **0.121** |

Pairwise / listwise give the lowest routing regret and highest oracle-hit — the
first lever that moves the routing decision, ~2 points better than baseline and
better than the paper's K=25 model — but pure ranking losses **wreck calibration**
(BCE 0.56–0.58). **`bce_pairwise` is the practical pick**: BCE 0.498 *and* regret
0.132 / best OOD regret 0.121. On RouterBench, regret stays
representation-independent (0.138 vs 0.140) — P5 is an IRT-Router lever.

**Recommended IRT-Router routing model:** `model_params: free`, `dim: 16`,
`loss: bce_pairwise`, `sampler: query`, `val_metric: regret`.

## Shrinkage-to-`model_mean` for OOD (Item 4)

`router.nirt.shrinkage` (`novelty_weight` + `shrink_predictions`,
`scripts/nirt/shrinkage_eval.py`): a purely evaluation-time blend of the raw
prediction toward the per-model training rate, weighted by how novel a query's
embedding is (cosine to its nearest training query). No training change, no
leakage.

Real validation, IRT-Router OOD split (`irtrouter-nirt-2d-projected`), midpoint
tuned to the bert-base cosine scale (OOD-vs-train max-cosine mean 0.91,
train-vs-train 0.95):

| variant | BCE | AUC | regret |
|---|--:|--:|--:|
| raw prediction | 0.6060 | 0.7257 | 0.1215 |
| `model_mean` alone | 0.5474 | 0.7488 | 0.1199 |
| **shrink(midpoint=0.92)** | **0.5433** | **0.7581** | 0.1202 |

At the right midpoint shrinkage beats *both* the raw prediction and pure
`model_mean` on every metric — it keeps the per-query ranking signal while
correcting the systematic OOD miscalibration. **Trade-off:** the same midpoint
costs a little on the in-distribution test split (BCE 0.511 → 0.532). Use it when
a query is *known* to be from a held-out task type, or tune the midpoint on a
mixed validation set.

## Auxiliary pairwise (Arena / Judge) signal (Item 2)

`training.nirt.pairwise` (`build_pairwise_arrays`, `pairwise_loss`) +
`train.preference` in `configs/nirt.yaml`: an auxiliary Bradley-Terry loss over
Chatbot Arena / GPT-4-Judge battles, reusing the model's own bilinear score
`z(q,m) = a_m·θ_q(e_q) − b_m` for both battle participants against one shared
`θ_q` — more supervision pulling on the SAME query head, never folded into the
correctness matrix. Needs `model_params: projected`. Off by default;
`test_preference_*` in `test_nirt_model.py` + `test_pairwise.py` cover the wiring.

**Full-scale result (128,119 train-split Arena+Judge queries, GPU-encoded, all
133,213 battles): clearly HARMFUL, with no sweet spot.**

| `train.preference.weight` | val BCE | AUC | **regret** |
|---|--:|--:|--:|
| 0 (no preference) | 0.6047 | 0.7331 | **0.1181** |
| 0.02 | 0.6305 | 0.7238 | 0.1230 |
| 0.10 | 0.6914 | 0.5710 | 0.3206 |
| 1.00 | 0.6608 | 0.6815 | 0.3178 |

Even the smallest tested weight (0.02) makes routing worse; damage is monotonic
and plateaus near random-ish routing by weight ~0.1. The pairwise loss itself
trains cleanly throughout — the auxiliary objective is learned, it just isn't
complementary to correctness. **Verdict: do not enable `train.preference` for
correctness prediction as currently built** (mechanism and code stay, off by
default).

> **Correction (2026-09-04):** the "two notions of quality" read is *not*
> supported by this experiment as run. The Arena/Judge and RouterBench query sets
> have **zero query_id overlap** (72,966 correctness ids vs 160,308 preference
> ids, disjoint namespaces), and the table is computed **only** against the
> correctness-side validation split. So the curve is equally explained by "adding
> 160k out-of-distribution training queries hurts in-distribution eval". The
> negative **operational** result stands; the causal explanation is retracted
> pending a query-matched test — see [anchor_judge.md](anchor_judge.md).

**Infrastructure win, independent of the result:** the encode was blocked on CPU
throughput (~7.3 texts/s) until the machine's idle NVIDIA RTX 3050 Ti was found —
the installed `torch` was a CPU-only build. `torch==2.13.0+cu130` jumped encoding
to **83 texts/s (11×)**, cutting a ~5 h encode to ~30 min. The GPU is now
available for any encoding-heavy step.

## Conclusions (2026-09-04, corrected)

The model does **not** underfit — trained to convergence it overfits, and early
stopping at epoch ~15 is near-optimal. Capacity, architecture (P2), encoder swap
(P4a), and regularisation are **all near-dead levers for prediction BCE**. The
exploitable gaps are **data quality** and **objective**:

1. **RouterBench: the shot-count feature** — `query_features` recovers **−0.032
   val BCE** (0-/5-shot twins), bigger than every model-complexity lever combined.
2. **More training queries.** ~29k RouterBench / ~22k IRT-Router queries over a
   768-d input is data-starved — the model overfits within ~5 epochs. → more
   benchmarks, pooling the two suites.
3. **Routing objective + parameterisation** (done): `model_params: free` (−0.017
   BCE both suites) and P5 `bce_pairwise` (regret → 0.132, oracle-hit 0.843 →
   0.863) generalise better than raw BCE.
4. **IRT-Router is near its ceiling** — val 0.50 vs transductive ceiling 0.37, no
   twin problem, gentle overfitting. Little prediction headroom; routing is the
   lever.
5. **Fine-tuning the encoder (P4b): deprioritised.** The model already overfits.
6. **OOD calibration** regresses below `model_mean` — shrinkage (above) is the
   lever, not capacity.

---

## Tests

[`tests/router/nirt/test_nirt_model.py`](../tests/router/nirt/test_nirt_model.py): forward shapes / range,
softplus discrimination, `from_config` auto-constraint, known-value metrics +
marginal baselines, **synthetic recovery** for *both* orientations (Pearson >
0.85 on held-out queries), `build_model` dispatch, `fit` + checkpoint round-trip,
determinism, the `free`-mode guard against scoring unseen models, classical IRT
beating the column-mean on held-out cells and reproducing the per-model rate, the
routing report, cold-start, the P2 bit-identical guards, `test_sampler_default_unchanged`,
and `test_preference_*`. `tests/training/nirt/test_pairwise.py` (10 tests) covers the BT loss;
`tests/router/nirt/test_shrinkage.py` the OOD blend.
