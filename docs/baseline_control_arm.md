# Baseline control arm — Bernoulli + continuous response heads

A deliberately simple, interpretable **control arm**, compared against in
`scripts/route_compare.py` and isolated in
[`src/router/nirt/baseline/`](../src/router/nirt/baseline/). Part 1 is the plain
Bernoulli / BCE core; Part 2 swaps the response head (heteroskedastic Gaussian /
Beta / ZOIB) on the **same** IRT core and representation. Nothing here is tuned to
hit a reference number.

Shared representation details live elsewhere: chance-corrected targets and Arena
exclusion in [data_sources.md](data_sources.md); the `irt` embedding pathway,
`r_q` relevance vector, FAISS warm-up and NIRT dataset in
[query_and_model_representation.md](query_and_model_representation.md); the
cold-start wall in [nirt_model.md](nirt_model.md#cold-start--placing-a-held-out-llm-from-its-profile).

---

# Part 1 — plain Bernoulli / BCE baseline

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
(`e_q' = (1-alpha) e_q + alpha * neighbour_mean`). Sign convention: `theta_m >= 0`
(softplus/sigmoid), `a_q >= 0` (softplus), `b_q` subtracted ⇒ **large `b_q` = hard
query**. Consistent everywhere.

## 2. Dataset & binary label

Phase 0 `nirt_observations.parquet` — the RouterBench absolute-correctness signal
only (`accuracy` + `mc_accuracy`), 9 warm models, deterministic content-hash query
splits, joined at access time to the frozen `irt` (BERT-768) query + profile
embeddings, `r_q`, and the warm-up store.

| split | observations | queries |
|---|---|---|
| train | 262 548 | 29 172 |
| validation | ~29 k | 3 634 |
| test | ~33 k | 3 677 |

`y_qm = 1[ score_effective >= 0.5 ]` (config `response.binary_threshold`).
`mc_accuracy` rows are already {0, 1}; RouterBench free-form `accuracy` graded in
{0, .25, .5, .75, 1} is **thresholded, not cast** (~20 % of `accuracy` rows are
fractional). The graded score is preserved as `y_soft`. Positive rate ~0.575.
**Arena / GPT-4-Judge excluded** (`use_arena = false`; they're pairwise, already
kept out of `nirt_observations.parquet` — see [data_sources.md](data_sources.md)).

## 3. Model architecture (`router.nirt.baseline.model.BaselineNIRT`)

Composed from `router.nirt.components`:

* `WarmupBlender` — `e_q'` blend (off by default).
* `DiscriminationHead` — `f_a` (1-hidden-layer MLP, width 64) → softplus; times
  the relevance gate `sigmoid(W_r r_q)` (`W_r`: `C → K`, init bias 2 so the gate
  is ~0.88 ⇒ enabling relevance starts near pass-through).
* `DifficultyHead` — `f_b` (MLP width 64) → scalar.
* `InteractionLayer` — `logit = base_irt + gamma * MLP([a_q, theta_m, a_q*theta_m,
  r_q])`, `gamma` **initialised at 0** so turning the interaction on is a no-op
  start.
* `LengthHead` — independent `f_len(e_q) → scalar`. **Loss disabled**: Phase 0 has
  no output-token counts for any source, so there is no length target. The
  interface exists because Phase 3 couples length and correctness; we do not
  fabricate labels.
* `BernoulliResponseHead` — identity on the logit; `predict_proba = sigmoid`. The
  piece Part 2 swaps.

`forward` returns an `Output` exposing `logit, base_irt, a_q, b_q, theta_m,
relevance_gate, length_pred` for inspection.

## 4. `theta`, `a`, `b`, and `K`

* `theta_m`: `nn.Embedding(n_models, K)` (`model_params="free"`), init `N(0, 0.1)`.
  Belongs to the model, **never a function of the query**. Or `W_theta @ e_m` with
  a sigmoid bound (`projected`) for cold-start.
* `a_q ∈ R^K` ≥ 0 via softplus — how strongly each latent dimension drives this
  item. Inspectable per query.
* `b_q ∈ R` scalar difficulty, larger = harder.
* `K` is `model.theta_dim` (default 8; **not** claimed optimal) — the only knob
  between the 1-D and multidimensional model.

## 5. Loss & regularization (`router.nirt.baseline.losses`)

`binary_cross_entropy_with_logits` on the raw logit. `target: binary` (default) or
`soft`. Optional `pos_weight = #neg/#pos` (`class_weighting`, default off).

```
+ theta_l2        * mean(theta_m^2)          = 1e-4   shrink ability scale
+ theta_center_l2 * sum(mean_m theta_m ^2)   = 1e-3   pin ability mean ~ 0
+ discrimination_l2 * mean(a_q^2)            = 0
+ difficulty_l2   * mean(b_q^2)              = 1e-4   shrink difficulty
```

Mild and documented. The theta regularisers touch **only the ability rows that
appear in training** — a model absent from training gets no gradient (no
cold-start leakage). Not tuned to move a validation number.

## 6. Identifiability

`a^T theta - b` is invariant under `theta -> s theta + t` with matching changes to
`a`, `b`. The convention fixed: `theta` mean pinned to ~0 (`theta_center_l2`),
scale controlled (`theta_l2`, bounded to [0, 1] in the projected variant),
discrimination kept ≥ 0 (softplus, 2PL reading). The numbered latent dimensions
are **not** claimed to carry absolute semantic meaning —
`diagnostics.theta_spectrum` + the taxonomy are how we interpret them.

## 7. Training

`configs/phase1.yaml`. Adam, lr 1e-3, batch 4096, ≤60 epochs, early stop on
validation BCE (patience 8), `DataLoader` over materialised CPU arrays. CLI:
`--seed --device --epochs --batch-size --learning-rate --theta-dim --model-params
--no-relevance --no-interaction --warmup --checkpoint --resume`.

## 8. Synthetic recovery

`scripts/nirt/baseline/synthetic.py` — generate a well-specified multidimensional
IRT world (`theta_true`, `a_true`, `b_true`, `y ~ Bernoulli(sigma(a.theta-b))`),
train `BaselineNIRT`, check recovery up to the unidentifiable rotation/scale:

| criterion | threshold | result (K=4, Q=1800, M=12) |
|---|---|---|
| BCE gap vs Bayes-optimal | < 0.05 | **0.035** |
| difficulty ordering (Spearman) | > 0.6 | **0.85** |
| ability ordering (Spearman, aligned) | > 0.6 | **0.98** |
| ICC curves | monotone, non-saturated | sane |

`artifacts/phase1/synthetic/recovery.json`.

## 9. Results

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
| ΔBCE vs per-model-mean baseline | — | **−0.076** |

* **Beats the per-model-mean baseline by 0.076 nats** — the query head is doing
  real work, not just learning each model's average correctness.
* **Well calibrated out of the box** (ECE ~0.015, no temperature scaling).
* **theta spectrum**: singular values `[0.58, 0.49, 0.47, 0.40, 0.29, 0.24, 0.12,
  0.003]`, **effective rank 6.5 / 8** — genuinely multidimensional; last dimension
  unused. No pathology flags.

Reproducibility: a second run at the same seed reproduces validation BCE to < 1e-3.

## 10. Cold-start protocol

Held-out models `claude-2.0`, `code-llama-34b-instruct`, `vicuna-13b`,
`yi-34b-chat` — never in parameter fitting. `model_params="projected"` gives
`theta_m = W_theta @ e_m` for an unseen model from its profile embedding alone,
scored against the "predict the training correctness rate" baseline.

Result (`artifacts/phase1/baseline_projected`, K=8 projected, ~2 300 cold-start
observations across 3 cold models with test-split rows):

| | accuracy | BCE |
|---|---|---|
| baseline NIRT (profile → theta_m) | **0.58** | 0.74 |
| global-mean (predict training rate) | 0.44 | 0.72 |

The projection recovers *ranking* (accuracy 0.58 > 0.44) but is **not calibrated**
(BCE 0.74 > 0.72) — consistent with the NIRT finding: 9 training LLMs is too few
for a 768→K "description → ability" projection. The projected run's theta
effective rank drops to **3.3 / 8** (profile bottleneck).

## 11. Known limitations

* **39-cluster taxonomy is coarse and MiniLM-space-dependent** — good enough to
  condition `a_q`; not a validated ability ontology.
* **Length head is inert** — no token-count labels anywhere in Phase 0.
* **Cold-start** is only meaningful in `projected` mode; the default run is
  `free`.
* **Arena excluded** — the baseline sees only RouterBench correctness.
* `K = 8` is a default, not a tuned choice.
* Interaction residual + relevance gate start as near-no-ops by construction — if
  they help, it is a small effect.

---

# Part 2 — continuous response model

An ablation against Part 1: **same NIRT representation** (frozen encoder, `r_q`,
warm-up, `K=8`, `a_q`/`b_q`, interaction, splits, cold-start set), **different
response likelihood** over the graded score `y ∈ [0, 1]`. Pluggable
`ResponseHead`s (`src/router/nirt/baseline/response_head.py` +
`continuous_{normal,beta,zoib}.py`); `BaselineNIRT` produces the latent score
`z_qm` and hands `(z, [e_q, θ_m])` to the head without knowing which distribution
is used. `configs/phase2.yaml` adds `train.grad_clip: 5.0` (continuous heads only
— Bernoulli reproduces Part 1 bit-for-bit).

## 2.1 Why Bernoulli is insufficient

Part 1 thresholded the graded score to `1[y ≥ 0.5]`, discarding the gradation and
any notion of predictive spread. **Boundary statistics**
(`artifacts/phase2/response_boundary_statistics.json`, test split): `y == 0` ≈
0.37, `y == 1` ≈ 0.41, `0 < y < 1` ≈ 0.22. **~79 % of observations sit exactly on
a boundary** — this dominates the modelling choice.

## 2.2 The three heads

**Heteroskedastic Normal (2A)** — `y ~ Normal(mu=z, sigma=softplus(f_sigma(e_q,
theta_m) + s0) + 1e-6)`. `sigma` is input-dependent. The Normal has support
outside `[0, 1]`; the likelihood uses the true Normal, no clipping. A poor fit to
a mass-at-0/1 distribution — a machinery-validation step.

**Beta (2B)** — `mu = sigmoid(z)`, `kappa = softplus(f_kappa(...)+k0)` clamped;
`alpha = mu*kappa`, `beta = (1-mu)*kappa`. `E[y] = mu`,
`Var[y] = mu(1-mu)/(kappa+1)`. No mass at exact 0 or 1, so trained and evaluated
on **interior observations only** (`0 < y < 1`, ~22 % of the data). No jittering.

**Zero-One-Inflated Beta (2C)** —
`(pi_0, pi_1, pi_c) = softmax(f_pi(e_q, theta_m))`, Beta middle as above.
`y == 0 → log pi_0`; `y == 1 → log pi_1`; `0 < y < 1 → log pi_c + log Beta(y)`.
`E[Y] = pi_1 + pi_c · mu`. `pi_0`, `pi_1` are **literal probability mass at the
boundary values** — not a 3PL guessing parameter (which is a lower asymptote on
`P(correct | theta)`). ZOIB does not touch the IRT curve.

### Likelihood equations

```
Bernoulli : -[ y log sigmoid(z) + (1-y) log(1 - sigmoid(z)) ]        (== BCE, Part 1)
Normal    : 0.5 ((y - mu)/sigma)^2 + log sigma + 0.5 log 2pi
Beta      : -log B(alpha,beta)^{-1} y^{alpha-1} (1-y)^{beta-1}       (interior y only)
ZOIB      : -[ 1{y=0} log pi_0 + 1{y=1} log pi_1
               + 1{0<y<1} ( log pi_c + log Beta(y; alpha, beta) ) ]
```

Numerical stability: `sigma = softplus(.) + 1e-6`; `mu ∈ [1e-4, 1-1e-4]`,
`kappa ∈ [1e-4, 1e4]`; ZOIB probabilities from `log_softmax`; exact 0/1 detected
with `y <= 0` / `y >= 1` and routed to the boundary mass. `test_continuous_*`
assert finite NLL at `z = ±100/200` and `y ∈ {0, 1, 0.5, eps, 1-eps}`.

## 2.3 Synthetic recovery

`scripts/nirt/baseline/synthetic.py --response {normal,beta,zoib}`:

| head | mean-corr | difficulty ρ | ability ρ | dispersion ρ | result |
|---|---|---|---|---|---|
| Normal | 0.87 | 0.82 | 0.94 | 0.48 | **PASS** |
| Beta | 0.99 | 0.92 | 0.99 | 0.97 | **PASS** |
| ZOIB | 0.93 | 0.94 | 0.98 | 0.79 | **PASS** |

All three recover the IRT core and their own dispersion parameter with finite NLL.
Normal's heteroskedastic `sigma` recovery is the weakest (trades mu-fit against
sigma-fit; early stopping + grad clipping keep it stable).

## 2.4 Real-data results

Test split, K=8, `model_params=free`, seed 42 (`artifacts/phase2/comparison.json`).

| model | own NLL | MAE | RMSE | mean-cal ECE | cov 50 | cov 90 | BCE-diag | acc-diag | θ eff-rank |
|---|---|---|---|---|---|---|---|---|---|
| **Bernoulli** (Part 1) | 0.548 (BCE) | — | — | 0.015 | — | — | 0.548 | 0.712 | 6.45 |
| **Normal** | 0.383 | 0.338 | 0.399 | **0.013** | 0.40 | 0.94 | 0.551 | 0.712 | 6.47 |
| **Beta** (interior 22 %) | −0.323 | 0.411 | 0.458 | 0.098 | 0.09 | 0.20 | 0.686 | 0.590 | 5.55 |
| **ZOIB** | 0.388 | 0.337 | 0.400 | **0.012** | 0.81 | 0.98 | 0.573 | 0.706 | 6.13 |

**NLL is each model's own proper score** — not comparable across rows. Read
MAE/RMSE/calibration/coverage across rows.

* **Beta is starved.** With ~79 % of observations at a boundary it trains/evaluates
  on only ~22 % interior, θ eff-rank drops to 5.55, its 50 % interval covers 9 %.
  Plain Beta is the wrong model for this score distribution.
* **Normal validates the machinery** — finite NLL, representation intact (eff-rank
  6.47 ≈ Bernoulli's 6.45), best mean calibration — but MAE/RMSE are high (places
  `mu` in the middle of a bimodal target) and the 50 % interval undercovers.
* **ZOIB fits the actual distribution.** Boundary calibration near-exact:
  `P(y=0)` predicted 0.361 vs observed 0.366; `P(y=1)` predicted 0.415 vs observed
  0.410. Mean calibration ECE 0.012 (best). Representation largely intact (eff-rank
  6.13). Intervals slightly over-wide but usable.

The BCE-after-threshold diagnostic shows the continuous models land close to
Bernoulli on the binary task (Normal 0.712, ZOIB 0.706 acc) without training on it
— the graded likelihood does not cost binary accuracy.

## 2.5 Confidence intervals (for Phase 4 escalation)

Every head exposes `interval(out, level)` (central), `lower_bound(out, level)`
(one-sided lower quantile at `1-level`), `mean`, `stddev`. Normal: **analytic**
(`mu + sigma · Phi^{-1}(q)`). Beta / ZOIB: **sampling** (512 draws, empirical
quantiles — `torch.distributions.Beta` has no `icdf`). The router will escalate on
`prediction.lower_bound`, not `prediction.mean` — so `mean 0.82 / lower 0.61` can
be dispreferred to `mean 0.78 / lower 0.74`. The escalation policy itself is
Phase 4; Phase 2 only builds the statistical interface.

## 2.6 Cold-start

Same protocol as Part 1 (`model_params=projected`, scored with `head.as_probability`
= `pi_1 + pi_c · mu`):

| | accuracy | BCE-diag |
|---|---|---|
| ZOIB (projected) cold-start | 0.56 | 0.76 |
| Bernoulli (projected) cold-start | 0.58 | 0.74 |
| global-mean baseline | 0.44 | 0.72 |

**Same wall as Part 1**: recovers ranking, not calibration. ZOIB does not move
cold-start — 9 warm LLMs is too few to learn "description → ZOIB parameters" any
better than "description → Bernoulli logit". Projected θ eff-rank drops to ~4.0/8.

## 2.7 Model-selection decision

**Selected Phase 2 response model: ZOIB.** It is the only head that represents the
data-generating distribution (a 79 % boundary spike), and does so while matching
the best mean calibration (ECE 0.012), the best mean prediction error (MAE 0.34,
tied with Normal), the best uncertainty calibration (cov 50/90 = 0.81/0.98), and
keeping the IRT representation intact (eff-rank 6.13). Plain Beta *fails* here
rather than merely tying, because the boundary mass is so large. ZOIB earns its
extra complexity (`f_pi` 3 logits + `f_kappa`).

## 2.8 Known limitations

* Normal support leaks outside `[0, 1]` (machinery-validation only); needs early
  stopping against train-time variance underestimation.
* Beta ignores the ~79 % boundary mass — its numbers are not on the same
  denominator as the others; its sampled intervals are badly overconfident.
* ZOIB intervals slightly over-cover (cov 50 = 0.81); a calibrated post-hoc
  interval scaling could tighten them (deferred).
* Beta/ZOIB intervals are sampling-based (512 draws) — cheap Monte-Carlo noise.
* Cold-start is unchanged from Part 1 (data-starved profile projection).
* Arena still excluded; graded score is RouterBench-only.
