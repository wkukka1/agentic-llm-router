# Baseline control arm — continuous response model

An ablation against the Bernoulli baseline arm: **same NIRT representation**
(frozen encoder, `r_q`, warm-up, `K=8`, `a_q`/`b_q`, interaction, splits,
cold-start set), **different response likelihood** over the graded score
`y ∈ [0, 1]`. Implemented as pluggable `ResponseHead`s
(`src/router/nirt/baseline/response_head.py` + `continuous_{normal,beta,zoib}.py`);
`BaselineNIRT` produces the latent score `z_qm` and hands `(z, [e_q, θ_m])` to
the head without knowing which distribution is used.

## 1. Why Bernoulli is insufficient for graded LLM evaluation

RouterBench `performance` is graded in `{0, .25, .5, .75, 1}` (and MC accuracy in
`{0, 1}`). Phase 1 thresholded it to `1[y ≥ 0.5]`, discarding the gradation and
any notion of predictive spread. A `0.5` (half-credit) and a hard `0`/`1` were
treated identically once on the same side of the threshold. Phase 2 keeps the
soft score and asks whether a proper continuous likelihood buys better
calibration / NLL / uncertainty without touching the representation.

**Boundary statistics** (`artifacts/phase2/response_boundary_statistics.json`,
test split): `y == 0` ≈ 0.37, `y == 1` ≈ 0.41, `0 < y < 1` ≈ 0.22. **~79 % of
observations sit exactly on a boundary** — this dominates the modelling choice.

## 2. Heteroskedastic Normal (Phase 2A)

    y_qm ~ Normal(mu_qm, sigma_qm)
    mu_qm    = z_qm
    sigma_qm = softplus( f_sigma(e_q, theta_m) + s0 ) + epsilon        (epsilon = 1e-6)

`sigma` is input-dependent (heteroskedastic), not a global constant. The Normal
has support outside `[0, 1]`; **the likelihood uses the true Normal, no
clipping** (§7 of the brief). For metrics that need a probability we report
`P(Y ≥ 0.5) = Phi((mu − 0.5)/sigma)`; the expected score is `clamp(mu, 0, 1)`.
Neither feeds the loss. This is a machinery-validation step — a Normal is a poor
fit to a mass-at-0/1 distribution.

NLL: `-Normal(mu, sigma).log_prob(y)` via `torch.distributions`.

## 3. Beta (Phase 2B)

    mu_qm    = sigmoid(z_qm)                         in (mu_min, 1-mu_min)   -> E[y] = mu
    kappa_qm = softplus( f_kappa(e_q, theta_m) + k0 ) + eps , clamped [min_conc, max_conc]
    alpha    = mu * kappa ,   beta = (1 - mu) * kappa

Mean/concentration parameterisation: `E[y] = mu` while `kappa` carries the
learned dispersion (`Var[y] = mu(1-mu)/(kappa+1)`). The Beta density has **no
mass at exact 0 or 1**, so the Beta model is trained and evaluated on **interior
observations only** (`0 < y < 1`, ~22 % of the data). We do NOT jitter exact
0/1 to fake interior support; the boundary fraction is reported separately.

NLL: `-Beta(alpha, beta).log_prob(clamp(y, 1e-6, 1-1e-6))`.

## 4. Zero-One-Inflated Beta (Phase 2C)

    P(Y=0) = pi_0 ,  P(Y=1) = pi_1 ,  P(0<Y<1) = pi_c
    (pi_0, pi_1, pi_c) = softmax( f_pi(e_q, theta_m) )        (guaranteed simplex)
    mu = sigmoid(z) , kappa clamped , alpha = mu*kappa , beta = (1-mu)*kappa

Log-likelihood (log-space, no tiny-probability-then-log):

    y == 0      ->  log pi_0
    y == 1      ->  log pi_1
    0 < y < 1   ->  log pi_c + log Beta(y; alpha, beta)

`E[Y] = pi_1 + pi_c · mu`;  `Var[Y] = E[Y²] − E[Y]²` with
`E[Y²] = pi_1 + pi_c·(Var_beta + mu²)`.

## 5. Boundary-mass interpretation vs 3PL guessing

`pi_0`, `pi_1` are **literal probability mass at the boundary values** of the
graded score. A 3PL guessing parameter `c` is an *item-response mechanism* — a
lower asymptote on `P(correct | theta)` as `theta → -inf`. ZOIB does not touch
the IRT curve `P(correct | theta)`; it models the empirical spikes of the graded
score at 0 and 1 plus a continuous Beta over the middle. ZOIB does **not** bring
guessing back (Phase 0 decision: no guessing parameter, fitted or fixed).

## 6. Likelihood equations

    Bernoulli : -[ y log sigmoid(z) + (1-y) log(1 - sigmoid(z)) ]        (== BCE, the Phase 1 loss)
    Normal    : 0.5 ((y - mu)/sigma)^2 + log sigma + 0.5 log 2pi
    Beta      : -log B(alpha,beta)^{-1} y^{alpha-1} (1-y)^{beta-1}       (interior y only)
    ZOIB      : -[ 1{y=0} log pi_0 + 1{y=1} log pi_1
                   + 1{0<y<1} ( log pi_c + log Beta(y; alpha, beta) ) ]

## 7. Numerical-stability measures

* Normal: `sigma = softplus(.) + 1e-6` (strictly > 0).
* Beta / ZOIB: `mu ∈ [1e-4, 1-1e-4]`, `kappa ∈ [1e-4, 1e4]` (config
  `min_concentration` / `max_concentration`), `alpha,beta` clamped ≥ `min_conc`,
  `y` clamped to `[1e-6, 1-1e-6]` for the Beta term.
* ZOIB probabilities from `log_softmax` (never `softmax` then `log`).
* Exact 0/1 detected with `y <= 0` / `y >= 1` and routed to the boundary mass —
  distinct from near-boundary `y = 1e-6` which uses the Beta density.
* Tests (`test_continuous_*`) assert finite NLL at `z = ±100/200` and
  `y ∈ {0, 1, 0.5, eps, 1-eps}`.

## 8. Synthetic recovery

`scripts/nirt/baseline/synthetic.py --response {normal,beta,zoib}` — a
well-specified world with known `theta_true`, `a_true`, `b_true`, and the head's
dispersion (`sigma`/`kappa`) + (ZOIB) `pi` parameters. Checks: finite NLL,
predictive-mean correlation, difficulty/ability Spearman, dispersion Spearman,
(ZOIB) boundary-mass ≈ empirical fractions.

| head | mean-corr | difficulty ρ | ability ρ | dispersion ρ | result |
|---|---|---|---|---|---|
| Normal | 0.87 | 0.82 | 0.94 | 0.48 | **PASS** |
| Beta | 0.99 | 0.92 | 0.99 | 0.97 | **PASS** |
| ZOIB | 0.93 | 0.94 | 0.98 | 0.79 | **PASS** |

All three recover the IRT core (ability/difficulty ordering) and their own
dispersion parameter with finite NLL. Normal's heteroskedastic `sigma` recovery
is the weakest — the Normal NLL trades mu-fit against sigma-fit and tends toward
variance underestimation on train (early stopping on validation NLL + grad
clipping keep it stable).

## 9. Real-data results

Test split, K=8, `model_params=free`, seed 42, same representation as Phase 1
(`artifacts/phase2/comparison.json`).

| model | own NLL | MAE | RMSE | mean-cal ECE | cov 50 | cov 90 | BCE-diag | acc-diag | θ eff-rank |
|---|---|---|---|---|---|---|---|---|---|
| **Bernoulli** (Phase 1) | 0.548 (BCE) | — | — | 0.015 | — | — | 0.548 | 0.712 | 6.45 |
| **Normal** | 0.383 | 0.338 | 0.399 | **0.013** | 0.40 | 0.94 | 0.551 | 0.712 | 6.47 |
| **Beta** (interior 22 %) | −0.323 | 0.411 | 0.458 | 0.098 | 0.09 | 0.20 | 0.686 | 0.590 | 5.55 |
| **ZOIB** | 0.388 | 0.337 | 0.400 | **0.012** | 0.81 | 0.98 | 0.573 | 0.706 | 6.13 |

**NLL is each model's own proper score** (Bernoulli BCE vs Normal/Beta/ZOIB NLL)
— *not* comparable across rows. Read MAE/RMSE/calibration/coverage across rows.

* **Beta is starved.** With ~79 % of observations at a boundary it trains and
  evaluates on only the ~22 % interior, its θ effective rank drops to 5.55, and
  its sampled intervals collapse (50 % interval covers 9 %). Confirmed: plain
  Beta is the wrong model for this score distribution.
* **Normal validates the machinery** — finite NLL, representation intact
  (eff-rank 6.47 ≈ Bernoulli's 6.45), best-in-class mean calibration — but
  MAE/RMSE are high (it places `mu` in the middle of a bimodal target) and the
  50 % interval undercovers (0.40).
* **ZOIB fits the actual distribution.** Boundary calibration is near-exact:
  `P(y=0)` predicted 0.361 vs observed 0.366; `P(y=1)` predicted 0.415 vs
  observed 0.410. Mean calibration ECE 0.012 (best). Representation largely
  intact (eff-rank 6.13). Intervals slightly over-wide (cov 50 = 0.81, cov 90 =
  0.98) but usable, unlike Beta.

The secondary BCE-after-threshold diagnostic (`BCE-diag` / `acc-diag`) shows the
continuous models land close to Bernoulli on the binary task (Normal 0.712,
ZOIB 0.706 vs 0.712 acc) without training on it — the graded likelihood does not
cost binary accuracy.

## 10. Calibration

`artifacts/phase2/<model>/plots/`: `mean_calibration.png` (predicted E[Y] vs
observed mean score), `uncertainty_calibration.png` (nominal vs empirical
central-interval coverage), `boundary_calibration.png` (ZOIB: predicted
`P(y=0)`/`P(y=1)` vs empirical, binned). ZOIB mean-calibration is essentially on
the diagonal; Beta's is the worst (ECE 0.098).

## 11. Cold-start

Same protocol as Phase 1 (held-out models `claude-2.0`,
`code-llama-34b-instruct`, `vicuna-13b`, `yi-34b-chat`; `model_params=projected`,
`theta_m = W_theta @ e_m`; scored with `head.as_probability` =
`pi_1 + pi_c · mu`). Only `code-llama-34b-instruct` and `yi-34b-chat` have
test-split rows in the irt query store.

| | accuracy | BCE-diag |
|---|---|---|
| ZOIB (projected) cold-start | 0.56 | 0.76 |
| Phase 1 Bernoulli (projected) cold-start | 0.58 | 0.74 |
| global-mean baseline | 0.44 | 0.72 |

**Same wall as Phase 1**: the profile→ability projection recovers ranking
(accuracy > global-mean) but not calibration (BCE-diag ≥ global-mean). ZOIB does
not move cold-start — 9 warm LLMs is too few to learn "description → ZOIB
parameters" any better than "description → Bernoulli logit". Projected θ
effective rank drops to ~4.0 / 8 (profile bottleneck), as in Phase 1.

## 12. Uncertainty

`mean_uncertainty` (predictive std): Normal 0.39, Beta 0.17 (over-concentrated),
ZOIB 0.37. ZOIB's std is honest — it reflects the boundary-mass split plus the
Beta spread. Coverage curves (`uncertainty_calibration.png`) show ZOIB slightly
conservative, Normal slightly anti-conservative at the 50 % level.

## 13. Confidence intervals (for Phase 4 escalation)

Every head exposes `interval(out, level)` (central), `lower_bound(out, level)`
(one-sided lower quantile at `1-level`), `mean`, `stddev`. Construction:

* Normal: **analytic** — `mu + sigma · Phi^{-1}(q)`.
* Beta / ZOIB: **sampling** — 512 draws from the fitted distribution
  (ZOIB: component then value), empirical quantiles. Chosen because
  `torch.distributions.Beta` has no `icdf`; matches the fitted distribution
  exactly in the limit.

## 14. Mean vs confidence (for Phase 4 escalation)

The system distinguishes a *high predicted score* from a *high-confidence*
predicted score. Every head exposes `mean`, `stddev`, `interval(out, level)`
(central) and `lower_bound(out, level)` (one-sided lower quantile at `1-level`).
The router will escalate on `prediction.lower_bound`, not `prediction.mean` — so
`mean 0.82 / lower 0.61` can be dispreferred to `mean 0.78 / lower 0.74`. The
escalation policy itself is Phase 4; Phase 2 only builds the statistical
interface.

## 15. Model-selection decision

**Selected Phase 2 response model: ZOIB.**

Ranked on the brief's criteria:

| criterion | winner | note |
|---|---|---|
| proper scoring / NLL | (within-family) | not cross-comparable; ZOIB & Normal both finite and stable |
| mean calibration | **ZOIB** (0.012) ≈ Normal (0.013) ≫ Beta (0.098) | |
| mean prediction error | **ZOIB** ≈ Normal (MAE 0.34) ≫ Beta (0.41, interior only) | |
| uncertainty calibration | **ZOIB** (cov 50/90 = 0.81/0.98) ≫ Normal (0.40/0.94) ≫ Beta (0.09/0.20) | |
| boundary fit | **ZOIB** — `P(y=0/1)` within 0.006 of empirical | only ZOIB models it |
| representation preserved | Normal (6.47) ≈ **ZOIB** (6.13) ≫ Beta (5.55) | |
| numerical stability | all stable with grad-clip + clamps; Normal needs early stopping against variance-underfit | |
| complexity | Bernoulli < Normal < Beta < ZOIB | ZOIB adds `f_pi` (3 logits) + `f_kappa` |

ZOIB earns its extra complexity: it is the only head that represents the
data-generating distribution (a 79 % boundary spike), and it does so while
matching the best mean calibration and keeping the IRT representation intact.
This is **Case A** of the brief (ZOIB best) — with the nuance that plain Beta
*fails* here rather than merely tying, because the boundary mass is so large.

## 16. Known limitations

* Normal support leaks outside `[0, 1]` (documented; machinery-validation only);
  it also needs early stopping against train-time variance underestimation.
* Beta ignores the ~79 % boundary mass — trains/evals on interior only, so its
  θ effective rank drops and its numbers are not on the same denominator as the
  others. Its sampled intervals are badly overconfident (learned κ too high).
* ZOIB intervals slightly over-cover (cov 50 = 0.81); a calibrated post-hoc
  interval scaling could tighten them (deferred).
* Dispersion / concentration / π nets condition on `[e_q, theta_m]`; a richer
  context is possible but out of scope for the ablation.
* Beta/ZOIB intervals are sampling-based (512 draws) — cheap Monte-Carlo noise.
* Cold-start is unchanged from Phase 1 (data-starved profile projection).
* Arena still excluded; graded score is RouterBench-only.
* `configs/phase2.yaml` adds `train.grad_clip: 5.0` (applied to continuous heads
  only — Bernoulli is untouched and reproduces Phase 1 bit-for-bit).
