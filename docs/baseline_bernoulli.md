# Phase 1 -- plain Bernoulli / BCE NIRT baseline

A deliberately simple, interpretable **control arm**. Phase 2 replaces the
response head (heteroskedastic Gaussian / Beta / ZOIB ...); it does **not**
replace this IRT core. Nothing here is tuned to hit the reference ~0.67.

## 1. Mathematical formulation

`model_latent` orientation (examinee = LLM has ability; item = query has
discrimination + difficulty):

```
theta_m  in R^K      per-model ability      (free params, or W_theta @ e_m for cold-start)
a_q      in R^K      query discrimination   a_q = softplus(f_a(e_q')) * sigmoid(W_r r_q)   (>= 0)
b_q      in R        query difficulty       b_q = f_b(e_q')     (larger b_q = harder)

z_qm = a_q^T theta_m - b_q            base IRT score  (always exposed for diagnostics)
logit = z_qm + gamma * g(a_q, theta_m, r_q)          interaction residual, gamma = 0 at init
p_qm  = sigmoid(logit)
L     = BCEWithLogits(logit, y_qm)  [+ regularisers]
```

`e_q'` is the (optionally) warm-up-blended query embedding
(`e_q' = (1-alpha) e_q + alpha * neighbour_mean`).

Sign convention: `theta_m >= 0` (softplus/sigmoid), `a_q >= 0` (softplus),
`b_q` subtracted => **large `b_q` = hard query**. Consistent everywhere.

## 2. Dataset

Phase 0 `nirt_observations.parquet` -- the RouterBench absolute-correctness
signal only (`accuracy` + `mc_accuracy`), 9 warm models, deterministic
content-hash query splits. Joined at access time to the frozen `irt`-pathway
(BERT-768) query + profile embeddings, the `r_q` relevance store, and the
warm-up store.

| split | observations | queries |
|---|---|---|
| train | 262 548 | 29 172 |
| validation | ~29 k | 3 634 |
| test | ~33 k | 3 677 |

Every query is scored by all 9 models (dense 9-wide). Positive rate ~0.575
(near balanced -- see §10).

## 3. Binary-label construction

`y_qm = 1[ score_effective >= binary_threshold ]`, `binary_threshold = 0.5`
(config `response.binary_threshold`).

* `score_effective` = the chance-corrected score where available (MC), else the
  raw score.
* `mc_accuracy` observations are already in {0, 1}.
* RouterBench free-form `accuracy` is graded in {0, .25, .5, .75, 1}; it is
  **thresholded**, not cast with `int()`. ~20 % of `accuracy` rows are fractional.
* The original graded score is preserved as `y_soft` in the arrays and
  `score`/`corrected_score` in the Phase 0 tables -- Phase 0 data is never
  overwritten.

## 4. Arena handling

`use_arena = false`. Arena / GPT-4-Judge are pairwise preferences, not
independent binary correctness, and are already kept out of
`nirt_observations.parquet`. A pairwise auxiliary head is Phase 2+ work. This is
documented in `docs/arena_interpretation.md`.

## 5. Model architecture (`router.nirt.baseline.model.BaselineNIRT`)

Composed from `router.nirt.components`:

* `WarmupBlender` -- `e_q'` blend (off by default).
* `DiscriminationHead` -- `f_a` (1-hidden-layer MLP, width 64) -> softplus; times
  the relevance gate `sigmoid(W_r r_q)` (`W_r`: `C -> K`, init bias 2 so the gate
  is ~0.88 => enabling relevance starts near pass-through).
* `DifficultyHead` -- `f_b` (MLP width 64) -> scalar.
* `InteractionLayer` -- returns `(logit, base_irt)`; `logit = base_irt + gamma *
  MLP([a_q, theta_m, a_q*theta_m, r_q])`, `gamma` **initialised at 0** so turning
  the interaction on is a no-op start.
* `LengthHead` -- independent `f_len(e_q) -> scalar`. **Loss disabled**: Phase 0
  has no output-token counts for any source (`input_tokens`/`output_tokens` are
  all null), so there is no length target. The module/interface exists because
  Phase 3 couples length and correctness; we do not fabricate labels.
* `BernoulliResponseHead` -- identity on the logit; `predict_proba = sigmoid`.
  The piece Phase 2 swaps.

`forward` returns an `Output` exposing `logit, base_irt, a_q, b_q, theta_m,
relevance_gate, length_pred` for inspection.

## 6. `theta`, `a`, `b`

* `theta_m`: `nn.Embedding(n_models, K)` (`model_params="free"`), init `N(0,
  0.1)`. Belongs to the model, **never a function of the query**. Or
  `W_theta @ e_m` with a sigmoid bound (`model_params="projected"`) for
  cold-start.
* `a_q in R^K`: how strongly each latent dimension drives this item's response.
  `>= 0` via softplus. Inspectable per query.
* `b_q in R`: scalar difficulty, larger = harder.

`K` is `model.theta_dim` (default 8; **not** claimed optimal). It is the only
knob between the 1-D and multidimensional model.

## 7. Relevance vector `r_q`

See `docs/query_representation.md`. `r_q in R^39` (softmax over cosine similarity
to the 39 cluster centroids). Conditions `a_q` through the relevance gate so a
query only loads on the latent dimensions it is actually about. Ablate with
`ablation.use_relevance: false` / `--no-relevance`.

## 8. Warm-up mechanism

FAISS bank over train queries (Phase 0, never rebuilt) -> `k=5` nearest train
queries -> mean of their `irt` embeddings -> `e_q' = (1-alpha) e_q + alpha *
neighbour_mean`, `alpha = 0.5` (config, optionally learnable). Off by default
(`ablation.use_warmup: false`); enable with `--warmup`. Rows with no neighbour
mean fall back to `e_q`.

## 9. Loss

`torch.nn.functional.binary_cross_entropy_with_logits` on the raw logit (never
`sigmoid` -> `BCELoss`). `target: binary` (default) or `soft`. Optional
`pos_weight = #neg/#pos` (`class_weighting`, default off).

## 10. Regularization (`router.nirt.baseline.losses`)

```
+ theta_l2        * mean(theta_m^2)          = 1e-4   shrink ability scale
+ theta_center_l2 * sum(mean_m theta_m ^2)   = 1e-3   pin ability mean ~ 0
+ discrimination_l2 * mean(a_q^2)            = 0
+ difficulty_l2   * mean(b_q^2)              = 1e-4   shrink difficulty
```

Mild and documented. The theta regularisers touch **only the ability rows that
appear in training** -- a model absent from training gets no gradient (no
cold-start leakage). Not tuned to move a validation number.

## 11. Identifiability

`a^T theta - b` is invariant under `theta -> s theta + t` with matching changes
to `a`, `b`. The convention we fix:

* `theta` **mean pinned to ~0** (`theta_center_l2`);
* `theta` **scale controlled** (`theta_l2`), and bounded to [0, 1] in the
  projected variant (`bound_ability`);
* discrimination kept `>= 0` (softplus), matching a 2PL reading.

Runs are comparable under this convention. The numbered latent dimensions are
**not** claimed to carry absolute semantic meaning -- `diagnostics.theta_spectrum`
+ the taxonomy are how we interpret them.

## 12. Training configuration

`configs/phase1.yaml`. Adam, lr 1e-3, batch 4096, <=60 epochs, early stop on
validation BCE (patience 8), `torch.utils.data.DataLoader` over materialised
CPU arrays (never a block on GPU). CLI: `--seed --device --epochs --batch-size
--learning-rate --theta-dim --model-params --no-relevance --no-interaction
--warmup --checkpoint --resume`.

## 13. Evaluation metrics (`router.nirt.metrics`, `router.nirt.baseline.calibration`)

accuracy (@0.5), log loss (BCE), Brier (`mean((p-y)^2)`), AUC, Spearman,
reliability curve + ECE / MCE, per-model and per-benchmark-family breakdowns,
and cold-start metrics reported **separately**.

## 14. Cold-start protocol

Phase 0 cold-start models: `claude-2.0`, `code-llama-34b-instruct`,
`vicuna-13b`, `yi-34b-chat` -- held out of every split, their responses never in
parameter fitting (the identifiability regularisers touch only the ability rows
seen in training, so there is no gradient leakage). `model_params="projected"`
gives `theta_m = W_theta @ e_m` for an unseen model from its profile embedding
alone; scored against the "predict the training correctness rate" baseline. A
`free` run cannot place an unseen model and reports cold-start as skipped.

Result (`artifacts/phase1/baseline_projected`, K=8 projected, ~2 300 cold-start
observations across the 3 cold models with test-split rows in the query store):

| | accuracy | BCE |
|---|---|---|
| baseline NIRT (profile -> theta_m) | **0.58** | 0.74 |
| global-mean (predict training rate) | 0.44 | 0.72 |

The projection recovers *ranking* (accuracy 0.58 > 0.44) but is **not
calibrated** (BCE 0.74 > 0.72) -- consistent with the earlier NIRT finding
(`docs/nirt_model.md`): 9 training LLMs is too few for a 768->K "description ->
ability" projection. The projected run's theta effective rank is only **3.3 / 8**
(vs 6.5 for the free run) -- the profile bottleneck compresses ability into ~3
dimensions.

## 15. Synthetic recovery experiment

`scripts/nirt/baseline/synthetic.py` -- generate a well-specified multidimensional
IRT world (`theta_true`, `a_true`, `b_true`, `y ~ Bernoulli(sigma(a.theta-b))`),
train `BaselineNIRT` on it, check recovery up to the unidentifiable
rotation/scale:

| criterion | threshold | result (K=4, Q=1800, M=12) |
|---|---|---|
| BCE gap vs Bayes-optimal | < 0.05 | **0.035** |
| difficulty ordering (Spearman) | > 0.6 | **0.85** |
| ability ordering (Spearman, aligned) | > 0.6 | **0.98** |
| ICC curves | monotone, non-saturated | sane |

The implementation recovers a synthetic IRT problem. `artifacts/phase1/synthetic/recovery.json`.

## 16. Results

Default config (`K=8`, `model_params=free`, relevance on, interaction on, warm-up
off, `constrain_discrimination=true`), seed 42, early stop at epoch 14 (~180 s CPU).

| metric | validation | test |
|---|---|---|
| n | 32 706 | 33 093 |
| accuracy @0.5 | 0.718 | 0.712 |
| log loss (BCE) | 0.538 | 0.548 |
| Brier | 0.183 | 0.186 |
| AUC | 0.787 | 0.782 |
| Spearman (per-obs) | 0.490 | 0.483 |
| ECE (15 bins) | 0.011 | 0.015 |
| ΔBCE vs per-model-mean baseline | -- | **-0.076** |

* **Beats the per-model-mean baseline by 0.076 nats** -- the query head is doing
  real work, not just learning each model's average correctness.
* **Well calibrated out of the box** (ECE ~0.015, no temperature scaling).
* **theta spectrum**: singular values `[0.58, 0.49, 0.47, 0.40, 0.29, 0.24,
  0.12, 0.003]`, **effective rank 6.5 / 8**, not collapsed -- the latent ability
  space is genuinely multidimensional; the last dimension is unused.
* No pathology flags (theta norms bounded, discrimination >= 0, difficulty
  well-distributed).
* Reference point ~0.67 cold-start accuracy: we get **0.58** (§14). Before
  reading this as failure: only 9 warm LLMs (the reference likely had more), a
  different held-out-model set, and binary-at-0.5 labels vs the reference's
  construction. The earlier NIRT experiment hit the same wall from the same
  cause. This phase is a reproduction of that limitation, not a regression to
  fix by adding layers.

ICC curves (`plots/icc/`) for easy / medium / hard / high- and low-discrimination
queries are monotone increasing in ability and non-saturated -- qualitatively
sane 2PL-style curves.

Reproducibility: a second run at the same seed reproduces validation BCE to
< 1e-3 (`artifacts/phase1/baseline_repro/`).

## 17. Known limitations

* **39-cluster taxonomy is coarse and MiniLM-space-dependent** -- many clusters
  are MCQ boilerplate ("answer, correct, single"). Good enough to condition
  `a_q`; not a validated ability ontology.
* **Length head is inert** -- no token-count labels anywhere in Phase 0.
* **Cold-start** is only meaningful in `projected` mode; the default run is
  `free`. The earlier NIRT experiment found profile-only cold-start *worse* than
  a warm-mean baseline with 9 training LLMs -- expect the same here.
* **Arena excluded** -- the baseline sees only RouterBench correctness.
* `K = 8` is a default, not a tuned choice; the theta spectrum (below) shows how
  many dimensions are actually used.
* Interaction residual + relevance gate start as near-no-ops by construction --
  if they help, it is a small effect.

## Equations (summary)

```
z_qm = a_q^T theta_m - b_q
p_qm = sigmoid(z_qm)
L    = BCE(y_qm, z_qm)                    (BCEWithLogits)
a_q  = softplus(f_a(e_q)) * sigmoid(W_r r_q)
b_q  = f_b(e_q)
```
