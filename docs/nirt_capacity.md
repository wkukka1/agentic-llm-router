# NIRT capacity / ceiling decomposition (P1)

Part of the capacity workstream (`start-researching-on-how-velvet-thunder` plan).
The 2026-09-03 head search saturated latent width `K` and MLP width with **no
overfitting at 5× params** — the `query_latent` model *underfits*. This doc
decomposes the gap between NIRT and "perfect" so the capacity work targets the
real bottleneck.

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

## Conclusions → where the capacity goes

1. **Representation is the dominant bottleneck** — ~0.13–0.14 BCE above the
   oracle-rep ceiling on *both* suites, and the underfit families (code,
   reasoning, math) are exactly where a frozen general-text encoder is weakest.
   **→ P4a (swap encoder) is the highest-value next step.**
2. **The bilinear form is a real but smaller bottleneck** (MLP-router beats
   NIRT-ours by ~0.01–0.017 on both suites). **→ P2c (interaction term)** is worth
   a focused sweep, done cheaply alongside P4.
3. **Head width / depth / dropout / `model_hidden` (P2a/P2b)** buy ~−0.016 BCE and
   do not touch routing — keep the best config but do not invest more there.
4. **Routing responds to capacity on IRT-Router (~1 pt), not on RouterBench.**
   Evaluate everything from here on the IRT-Router `test` + `ood` splits.
5. **OOD calibration regresses below `model_mean`** for learned predictors — P5
   (routing-aware / robust loss) and a shrinkage-to-`model_mean` prior are the
   levers there, not raw capacity.
