# NIRT model (Phase 2)

[`src/router/nirt/`](../src/router/nirt/) — the response model, kept separate from
[`src/router/data/nirt.py`](../src/router/data/nirt.py) (the training *representation*).
Config: [`configs/nirt.yaml`](../configs/nirt.yaml).

## Formulation — two orientations

Same bilinear IRT core, `model.orientation` picks which side carries the latent ability:

```
query_latent  (ours, default)          model_latent  (IRT-Router, Song et al. ACL'25)
  theta_q = f_theta(e_q)                  theta_m = [sigmoid] W_theta e_m
  (a_m, b_m) on the model                 a_q = W_a e_q ,  b_q = W_b e_q
  y = sigmoid(a_m . theta_q - b_m)        y = sigmoid(a_q . theta_m - b_q)
```

- **`query_latent`** ([`NIRTModel`](../src/router/nirt/model.py)) — the latent vector is amortised
  from the query text; the model side is a small `(a_m, b_m)` table (`model_params=free`) or a
  projection of the profile embedding (`projected`, cold-start capable). No explicit per-query
  difficulty.
- **`model_latent`** ([`IRTRouterModel`](../src/router/nirt/model.py)) — the classical-IRT
  reading: the LLM is the examinee with an **ability vector `theta_m`**, the query is the item
  with **discrimination `a_q` and difficulty `b_q`** (both amortised from `e_q`). Matches this
  repo's README design and yields reusable per-LLM ability vectors. `free` == the paper's M-IRT.

`e_q` is a frozen `bert-base-uncased` mean-pooled embedding (irt pathway, 768-d). Training
minimises soft-label BCE against the graded RouterBench target (`performance ∈ {0,.25,.5,.75,1}`).
For `K = 1` discrimination is softplus-constrained (`> 0`, like 2PL); unconstrained for `K > 1`.

### A/B result (test split, dim 2, projected — `scripts/nirt/ab_orientation.py`)

| orientation | test BCE | AUC | Spearman | ranking optimal-rate | cost @ −3 acc pts ($/1k) |
|---|--:|--:|--:|--:|--:|
| classical IRT 2D (ceiling) | 0.593 | 0.79 | 0.49 | 0.81 | 1.99 |
| **model_latent** (IRT-Router) | **0.592** | 0.75 | 0.45 | 0.81 | 2.18 |
| query_latent (ours) | 0.594 | 0.76 | 0.43 | 0.81 | 2.85 |

`model_latent` wins narrowly at 2D and clearly at 1D (BCE 0.603 vs 0.614) — the explicit
per-query difficulty `b_q` helps, and it gives a better cost/quality frontier. Both orientations
sit ~0.02 BCE above the matrix-factorisation ceiling and produce the *same* routing decision at
λ=0 (always-GPT-4). Default stays `query_latent`; the evidence favours switching.

## What v1 is / is not

v1 is a **predictor**, evaluated on prediction quality only:

| metric | meaning |
|--------|---------|
| `bce`, `mse`, `mae` | error vs the soft target |
| `pearson_r`, `spearman_r` | correlation of `y_hat` with observed correctness |
| `auc` | ranking of correct vs incorrect (target binarised at 0.5) |
| `ece` | calibration (10-bin, against the soft outcome rate) |
| `acc@0.5` | agreement after thresholding both sides at 0.5 |

Every run is compared against two **marginal baselines**: `global_mean` (overall training
correctness rate) and `model_mean` (each model's mean training correctness, query-independent).
Beating `model_mean` is the bar — it means the query head learned something query-specific.

**Built** — dual orientation + A/B (above), classical-IRT baseline (both protocols), per-query
ranking metrics, cost-aware routing with a λ sweep (*Classical IRT & routing* below), and the
cold-start experiment (*Cold-start* below).

**Not built yet** (part 2): OOD held-out-benchmark split; `Reward(α)` + AIQ metrics; in-house
KNN-router / MLP-router baselines; clustering → relevance vector `r_q` → the IRT-Router
relevance-masked N-IRT head; kNN query-embedding warm-up. Then multi-source Arena / GPT-4-Judge
(M-IRT), kept isolated. The dimension ladder is a `--dim` sweep, not new code.

## Config (`configs/nirt.yaml`)

| key | |
|-----|--|
| `data.pathway` | embedding pathway to join against (`irt` = BERT) |
| `data.score_kind` | target column (`effective` = chance-corrected where MC) |
| `model.orientation` | `query_latent` (ours) \| `model_latent` (IRT-Router) |
| `model.dim` | `K` — latent ability dimensions |
| `model.model_params` | `projected` \| `free` |
| `model.query_hidden` | query-head hidden width; `null` → single linear layer |
| `model.constrain_discrimination` | `auto` → true iff `dim == 1` |
| `model.bound_ability` | `model_latent` only: `theta_m = sigmoid(W_theta e_m)` |
| `train.loss` | `soft_bce` \| `hard_bce` \| `mse` |
| `train.{lr,weight_decay,batch_size,epochs,patience,val_metric}` | Adam + early stop |
| `train.materialize` | gather the split into RAM once (fast on CPU) vs stream |
| `runs_dir` | `data/processed/nirt_runs` (gitignored) |

## Run

```bash
python scripts/nirt/train_nirt.py --config configs/nirt.yaml --dim 2 --model-params projected --name nirt-2d-projected
python scripts/nirt/eval_nirt.py  --run nirt-2d-projected --split test
python scripts/nirt/compare.py    --run nirt-2d-projected --split test    # vs classical IRT + routing/cost

# the other orientation (IRT-Router: theta on the LLM)
python scripts/nirt/train_nirt.py --orientation model_latent --dim 2 --model-params projected --name irt-2d-projected
python scripts/nirt/ab_orientation.py --dim 2 --model-params projected --reuse   # side-by-side

# cold-start: place code-llama-34b / yi-34b from their profile text alone
python scripts/nirt/coldstart.py --run nirt-2d-projected --split test
```

Each run writes `<runs_dir>/<name>/`: `model.pt` (`state_dict` + config + `model_index`),
`run.json` (config, git sha, per-epoch history, best val metrics, baselines), and — after
`eval_nirt.py` — `eval_<split>.json`.

Programmatic:

```python
from router.nirt import fit, load_run, evaluate_split
res = fit(yaml.safe_load(open("configs/nirt.yaml")))          # RunResult (.model, .val_metrics, ...)
model, cfg, model_index = load_run("nirt-1d-projected")
evaluate_split(model, model_index, "test")
```

## Classical IRT & routing

[`baselines.py`](../src/router/nirt/baselines.py) fits a classical (multidimensional) 2PL
with **no query features** — free per-query `theta_q`, per-model `a_m`, `b_m`:

* **transductive ceiling** (`protocol="cell"`) — fit on a random 85 % of the split's
  `(query, model)` cells, predict the held-out 15 %. The matrix-factorisation ceiling.
  `theta_q` is L2-regularised (`weight_decay=1e-3`); without it the sparse per-query fit
  overfits badly.
* **`theta_q = 0` lower bound** (`protocol="main_effects"`) — fit `a_m`, `b_m` on
  train+val, predict test queries as `sigmoid(-b_m)` (no query information at all).

[`routing.py`](../src/router/nirt/routing.py) turns any predictor into a policy
`m* = argmax_m (pred − λ · cost_norm)` and scores it on the dense test matrices
(`true` = graded RouterBench `performance`, `cost` = USD/query). Run it all:

```bash
python scripts/nirt/compare.py --run nirt-2d-projected --split test
```

### Results (test split — 3,677 queries, 9 warm RouterBench models)

**Prediction.** NIRT (text only, never sees a response) matches the transductive
matrix-factorisation ceiling on BCE:

| method | BCE | MSE | AUC | acc@0.5 | Spearman |
|--------|----:|----:|----:|--------:|---------:|
| NIRT-2D projected (inductive) | **0.594** | 0.162 | 0.762 | 0.702 | 0.43 |
| classical IRT 2D — transductive ceiling | 0.593 | 0.158 | 0.794 | 0.727 | 0.49 |
| classical IRT 2D — `theta=0` lower bound | 0.645 | 0.183 | 0.690 | 0.664 | 0.30 |
| baseline: per-model mean | 0.645 | 0.183 | — | — | — |
| baseline: global mean | 0.690 | 0.205 | — | — | — |

NIRT-1D lags slightly (BCE 0.614); 2D and 4D both land at ~0.592–0.594. Extra dimensions
past 2 do not help here.

**Routing + cost.** The warm pool is **GPT-4-dominated** — `gpt-4-1106-preview` is the (a)
best model on 84 % of test queries and only ~1.4× the price of the mid-tier Claudes — so
**quality-only routing (λ=0) collapses to "always GPT-4"** for *both* NIRT and the classical
ceiling. The value is on the cost/quality frontier:

| policy | quality | accuracy | $/1k queries | vs GPT-4 |
|--------|--------:|---------:|-------------:|----------|
| oracle (best per query) | 0.907 | 0.958 | 0.83 | −76 % cost, +12 pts acc |
| fixed: GPT-4 | 0.772 | 0.837 | 3.45 | — |
| fixed: cheapest (mistral-7b) | 0.302 | 0.290 | 0.05 | −99 % cost, −55 pts acc |
| NIRT route, λ=0 | 0.772 | 0.837 | 3.44 | ≈ GPT-4 |
| NIRT route, λ=0.5 | 0.768 | 0.826 | 2.85 | −17 % cost, −1 pt acc |
| NIRT route, λ=1.0 | 0.759 | 0.807 | 1.60 | **−54 % cost, −3 pts acc** |
| NIRT route, λ=3.0 | 0.710 | 0.755 | 0.82 | −76 % cost, −8 pts acc |

**Takeaway for the roadmap.** On prediction, the frozen-BERT query head already reaches the
MF ceiling — so the bottleneck is not the IRT structure. On routing, a single dominant model
means there is little quality headroom to capture; the win is a real but modest cost
reduction (≈half the spend at −3 accuracy points). Adding latent dimensions past 2 buys
nothing. The next lever is a stronger query representation and/or a less GPT-4-dominated
candidate pool (the Arena/Judge models), not more IRT dimensions.

`compare.py` writes `<runs_dir>/<run>/comparison_<split>.json` with the full tables + λ sweep.

## Cold-start — placing a held-out LLM from its profile

[`evaluate.cold_start_eval`](../src/router/nirt/evaluate.py) /
[`scripts/nirt/coldstart.py`](../scripts/nirt/coldstart.py). `code-llama-34b-instruct` and
`yi-34b-chat` are in [`cold_start_models.json`](../data/splits/cold_start_models.json) — held
out of *all* training. A `projected` run predicts their per-query correctness on the test
split from their **profile-text embedding only** (a projection the model never trained on).

References without NIRT: `global_mean` (overall train rate), `warm_mean` (mean of the 9 warm
models' predictions on that query), `nearest_warm` (the warm model whose profile embedding is
closest, cosine → use its prediction).

### Result — it does not work yet (test split, pooled over both cold LLMs)

| predictor | BCE | AUC |
|---|--:|--:|
| `warm_mean` (no NIRT) | **0.69** | 0.61 |
| `global_mean` (no NIRT) | 0.71 | 0.50 |
| NIRT `query_latent` projected | 0.71 | 0.59 |
| NIRT `model_latent` projected | 0.74 | 0.54 |
| `nearest_warm` (no NIRT) | 0.76 | 0.46 |

Both orientations land at or **below** the trivial `warm_mean` baseline. The by-family
breakdown shows why: `yi-34b-chat` is genuinely strong on knowledge (0.67) and reasoning
(0.73), but the projection predicts ~0.23 / ~0.40 — it reads yi as weak everywhere and ranks
it near-worst (predicted rank 8.7 of 10 vs true 1.3). `code-llama` is over-predicted on
knowledge (0.38 vs true 0.04).

**Why.** A 768→K linear projection trained on **9 LLMs** cannot learn "this description implies
high knowledge ability". IRT-Router's new-LLM result (0.68 vs 0.34) came with ~20 training
LLMs *and* by calibrating the new model's `theta_m` on a handful of its actual responses —
not from description alone. Description-only cold-start is the harder problem and needs either
many more training LLMs, a fine-tuned encoder, or a few calibration responses. Recorded as a
negative result; `coldstart_<split>.json` has the full breakdown.

## Tests

[`tests/test_nirt_model.py`](../tests/test_nirt_model.py): forward shapes / range, softplus
discrimination, `from_config` auto-constraint, known-value metrics + marginal baselines,
**synthetic recovery** for *both* orientations (the query/model head recovers the latent
vector and the true `P(correct)` to Pearson > 0.85 on held-out queries), `build_model`
dispatch, `fit` + checkpoint round-trip, determinism, the `free`-mode guard against scoring
unseen models, classical IRT beating the column-mean on held-out cells (`cell` protocol) and
reproducing the per-model rate (`main_effects`), the routing report (oracle bound,
fixed-model rows, λ→cheaper), and cold-start (a `projected` model scores a held-out model
better than `global_mean` on the synthetic where the profile embedding encodes ability;
`free` runs are refused).
