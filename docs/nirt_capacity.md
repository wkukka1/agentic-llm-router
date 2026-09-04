# NIRT capacity / ceiling decomposition (P1)

Part of the capacity workstream (`start-researching-on-how-velvet-thunder` plan).

> **CORRECTION (2026-09-04).** This doc originally concluded the `query_latent`
> model *underfits* (train ≈ val BCE, flat across a param sweep). That was an
> **early-stopping artifact** — those numbers were all measured at the val
> minimum (epoch ~15), where train and val are close *by construction*. Trained to
> convergence with no early stopping the model **overfits**: RouterBench train BCE
> 0.59→0.36 while val 0.61→**1.06**; IRT-Router train 0.55→0.45 while val
> 0.50→0.51. See "Trained-to-convergence behaviour" and "The RouterBench 0-/5-shot
> confound" below. The architecture / capacity / encoder levers (P2, P4a) are dead
> for a different reason than first thought: the model already has enough capacity,
> it does not *generalise*.

The 2026-09-03 head search saturated latent width `K` and MLP width **at the
early-stop point**. This doc decomposes the gap between NIRT and "perfect".

Script: `scripts/nirt/capacity_diagnostics.py`.
Artifacts: `artifacts/phase2/capacity_diagnostics.json` (RouterBench),
`artifacts/irt_router/capacity_diagnostics.json` (IRT-Router).

## Method

Four blocks, each also reporting **routing regret** (the workstream's success
metric):

1. **Transductive per-query ceiling** — `fit_classical_irt(protocol="cell")`:
   free per-query `theta_q` (no text), per-model `a_m`/`b_m`, rank-`K` MF, fit on
   85 % of the split's `(query, model)` cells, scored on the held-out 15 %. This
   is "how low BCE goes with an **oracle** query representation".
2. **IRT-free capacity bound** — `fit_mlp_router`: a plain `e_q -> R^M` MLP with
   dropout at widths 128 / 512 / 2048. MLP ≫ NIRT ⇒ the bilinear logit is a
   bottleneck. MLP plateaus with NIRT ⇒ the frozen `e_q` is the wall.
3. **NIRT reference** — the shipped runs.
4. **Per-family** — BCE / AUC grouped by benchmark family (`family_of_query`).

## Results — RouterBench (test, 40,447 obs, 9 models)

| predictor | BCE | routing regret |
|---|--:|--:|
| `model_mean` (no query info) | 0.6466 | — |
| **NIRT** `nirt-2d-projected` | 0.6117 | 0.1415 |
| NIRT + best P2 head (`complexity_search.py`, clean) | ~0.595 | ~0.140 |
| **MLP-router** (2048-wide + dropout, IRT-free) | **0.5947** | 0.135 |
| **transductive per-query ceiling** (oracle rep) | **0.5362** | — |

**Gap decomposition (NIRT → ceiling = 0.076 BCE):**

- **−0.017 : bilinear structure.** A plain wide MLP beats the IRT bilinear form
  `a_m·θ_q − b_m`. → **P2c** (gated non-additive interaction term).
- **−0.059 : query representation.** Even the 2048-wide MLP is 0.06 above the
  ceiling — it and NIRT both see a frozen `bert-base-uncased` mean-pool `e_q`; the
  ceiling has an oracle query rep. → **P4** (stronger / fine-tuned encoder). The
  bigger lever.
- The ceiling (0.536) is itself well above 0 → genuine aleatoric noise (graded
  scores, MC guessing, 0-/5-shot format variance sharing one question).

**Per-family — where the model is blind:** code BCE 0.714 / **AUC 0.56** (≈chance
ranking), reasoning 0.663 / 0.67, math 0.663 / 0.75, knowledge 0.562 / 0.80,
zh 0.356 / 0.80. Code and reasoning are the underfit families — a representation
problem (BERT does not encode code-task difficulty).

Routing regret is **flat ~0.135–0.142 for every predictor including the 2048-MLP**
— the GPT-4-dominated 9-model pool makes the per-query arg-max
representation-independent. RouterBench is not a useful routing testbed.

## Results — IRT-Router 20-model suite (the routing testbed)

### ID test (10,458 q, 209,160 obs)

| predictor | BCE | routing regret | oracle-hit\* |
|---|--:|--:|--:|
| `model_mean` | 0.5610 | — | — |
| **NIRT** (ours, `query_latent` K=2) | 0.5107 | 0.1462 | 0.843 |
| **MLP-router** (2048-wide) | **0.5003** | 0.1366 | — |
| IRT-Router paper (`model_latent` K=25) | 0.4986 | **0.1362** | 0.854 |
| **transductive ceiling** | **0.3719** | — | — |

Same shape: MLP-router (0.500) > NIRT-ours (0.511) by 0.011, still **0.13 above the
ceiling** (0.372). **Unlike RouterBench, routing regret is *not* flat** — the K=25
model_latent and the big MLP route ~1 point better (regret 0.136 vs 0.146). So on a
non-dominated pool, capacity *can* help the routing decision, modestly.

### OOD (3,159 q, held-out datasets)

| predictor | BCE | routing regret |
|---|--:|--:|
| `model_mean` | 0.5474 | — |
| NIRT (ours) | 0.6060 | 0.1215 |
| IRT-Router (K=25) | 0.5899 | 0.1240 |
| MLP-router h128 | 0.5423 | 0.1208 |
| MLP-router h2048 | 0.5656 | 0.1164 |

**The learned predictors are *worse than `model_mean`* on OOD BCE**, and a wider
MLP overfits OOD harder (0.542 → 0.566). math OOD family BCE 0.779. Routing regret
~0.12 for all. OOD calibration is a robustness problem that raw capacity does not
fix.

## P2 architecture capacity vs the routing decision (2026-09-03)

Trained on the IRT-Router suite: `irtrouter-nirt-cx-p2` (K=16, `query_hidden=128`,
3-layer head, `grad_clip=1`, cosine) and `irtrouter-nirt-cx-p2c` (+ interaction,
`interaction_hidden=64`). `route_compare.py --config configs/irt_router.yaml
--reference gpt_4o --lam 0.3`:

| run | val BCE | ID reward@0.8 | OOD reward@0.8 |
|---|--:|--:|--:|
| `irtrouter-nirt-2d-projected` (K=2 baseline) | 0.513 | 0.640 | 0.684 |
| `irtrouter-nirt-cx-p2` (K=16, 3-layer) | 0.510 | 0.639 | 0.686 |
| `irtrouter-nirt-cx-p2c` (+ interaction) | 0.510 | 0.640 | 0.685 |

**Flat.** P2 architecture capacity does not move the routing decision on *either*
suite (RouterBench regret 0.139–0.141 across 24 configs; IRT-Router reward@0.8
within 0.001 across K=2 / K=16 / K=16+interaction). The P1 hint that "K=25 routes
1 pt better" is the `model_latent` **orientation**, not capacity — a K=16
`query_latent` head did not capture it.

## P4a — encoder swap (2026-09-03/04)

Added a `bge` pathway (`BAAI/bge-small-en-v1.5`, 384-d, MTEB ~62 vs
`bert-base-uncased` mean-pool ~30) as a drop-in for the shared query+profile role.
`scripts/data/encode_irt_router_pathway.py` (new) encodes the IRT-Router
`query.csv` / `llm.csv` with any pathway. `all-mpnet-base-v2` was abandoned (5.9 h
CPU encode); bge-small is ~26 min RouterBench / ~13 min IRT-Router.

| suite | run | test BCE | test AUC | test regret |
|---|---|--:|--:|--:|
| IRT-Router | `irt` (paper vectors), K=2 projected | 0.511 | 0.808 | 0.146 |
| IRT-Router | `irt`, K=16 **free** | **0.494** | — | **0.134** |
| IRT-Router | `bge`, K=16 free | 0.494 | — | 0.138 |
| RouterBench | `irt`, K=16 free | **0.598** | 0.758 | 0.140 |
| RouterBench | `bge`, K=16 free | 0.605 | 0.753 | 0.138 |

**bge-small is a null on both suites** — BCE unchanged (IRT-Router) or slightly
worse (RouterBench), AUC flat/down, routing unchanged. A modern *small* (33M)
sentence encoder does not close the 0.13 BCE gap to the transductive ceiling. Open
question: a base-size encoder (`bge-base`, 109M) or fine-tuning (**P4b**), or the
gap is the profile side / aleatoric noise rather than the query encoder.

**The one lever that helped both suites: `model_params: free`** (a free per-model
ability vector instead of a linear projection of the profile-text embedding) —
RouterBench BCE 0.612 → 0.598, IRT-Router BCE 0.511 → 0.494 / regret 0.146 →
0.134. With only 9–20 LLMs, projecting `θ_m` from a short description loses signal.

## Trained-to-convergence behaviour (2026-09-04) — it OVERFITS

`model_params=free`, K=16, 2-layer head, **no early stopping**, minibatched:

| epoch | RB train | RB val | IRT-R train | IRT-R val |
|---|--:|--:|--:|--:|
| ~15 (early-stop point) | 0.59 | **0.61** (min) | 0.49 | **0.50** (min) |
| ~60 | 0.48 | 0.70 | 0.47 | 0.50 |
| ~120–200 | **0.36** | **1.06** | 0.45 | 0.51 |

RouterBench val BCE *explodes*; IRT-Router's barely moves. The model has ample
capacity — early stopping at epoch ~15 is near-optimal and is the operative
regulariser. Regularisation swept to convergence (dropout 0.3–0.5, wd 1e-4–1e-3)
moves the RouterBench val minimum by **−0.003 at best** (0.6055 → 0.6030); heavy
reg hurts. So val BCE ~0.60 (RB) / ~0.50 (IRT-R) is a **generalisation ceiling**,
not a capacity or optimisation floor.

`head(e_q)` is *not* information-bottlenecked (an earlier probe that suggested this
had an obs-vs-query indexing bug): with correct per-query indexing a plain head
fits 100 training queries to BCE 0.072. The issue is transfer to unseen queries.

## The RouterBench 0-/5-shot confound (2026-09-04) — the biggest single lever

RouterBench pool-expansion E2 made 0-shot and 5-shot versions of a question
**separate items that carry the identical question text** (hence identical `e_q`)
with different outcomes, and shot count is not a feature. The model memorises one
twin on train and its twin contradicts it in val — the mechanism behind the val
explosion above.

| RouterBench training data / features | best val BCE |
|---|--:|
| 0-shot + 5-shot, `e_q` only (default) | 0.602 |
| **0-shot only** | **0.557** (−0.045) |
| 0-shot + 5-shot, **`e_q` + 1 shot-indicator feature** | **0.569** (−0.033, all data) |
|   ↳ 0-shot regime | 0.558 (was 0.593) |
|   ↳ 5-shot regime | 0.580 (was 0.611) |

**One binary feature recovers ~all of the 0-shot-only performance and makes the
5-shot regime predictable too** — bigger than P2 (−0.016), kNN-impute (−0.002),
encoder swap (0), or regularisation (−0.003) combined. IRT-Router has no such
twins.

## Conclusions (2026-09-04, corrected)

The model does **not** underfit — trained to convergence it overfits, and early
stopping at epoch ~15 is near-optimal. Capacity, architecture (P2), encoder swap
(P4a), and regularisation are **all near-dead levers** for prediction BCE — each
tested directly. The exploitable gaps are **data quality** and **objective**, not
model complexity:

1. **RouterBench: add a shot-count feature.** The 0-/5-shot twin items
   (identical text, different labels, no shot feature) are the largest single
   loss: **−0.033 val BCE** from one binary feature, or −0.045 by dropping 5-shot.
   → build a structured per-query feature vector (shot count, task family, prompt
   length, `n_choices`) and concat it to `e_q`.
2. **More training queries.** ~29k RouterBench / ~22k IRT-Router queries over a
   768-d input is data-starved — the model overfits within ~5 epochs. → P6
   (Arena/judge adds ~50k queries + a related signal), more benchmarks, pooling
   the two suites.
3. **Routing objective + parameterisation** (already done): `model_params: free`
   (−0.017 BCE both suites, IRT-Router regret 0.146→0.134) and P5 `bce_pairwise`
   (regret →0.132, oracle-hit 0.843→0.863). These generalise better than raw BCE.
4. **IRT-Router is near its ceiling** — val 0.50 vs transductive ceiling 0.37, no
   twin problem, gentle overfitting. Little prediction headroom; routing is the
   lever.
5. **Fine-tuning the encoder (P4b)**: deprioritised. The model already overfits;
   a trainable encoder overfits harder and needs careful regularisation for an
   uncertain payoff. Only revisit if (1)+(2) plateau.
6. **OOD calibration regresses below `model_mean`** — a shrinkage-to-`model_mean`
   prior (blend `p̂` toward the per-model rate for low-confidence / far-from-train
   queries) is the lever, not capacity.
