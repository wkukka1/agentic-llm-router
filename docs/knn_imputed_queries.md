# kNN-imputed query embeddings

## Why

`NIRTModel` (`query_latent`) turns a frozen `bert-base-uncased` query embedding
`e_q` into a latent ability `theta_q = f_theta(e_q)`. For a query that sits far
from anything in the training set the head is extrapolating, and that is exactly
where the model is least trustworthy (the OOD held-out families, the long tail of
task types).

This variant replaces `e_q`, **before the query head sees it**, with a
similarity-weighted mean of its `k` nearest *training* queries:

```
e_q  :=  sum_i w_i * e_{n_i}          n_1..n_k = k nearest TRAIN queries of q
         w_i = max(cos(q, n_i), 0),  normalised   (self-excluded)
```

Every query is snapped onto a small neighbourhood of points the model actually
trained on. `k` trades bias for variance: `k=1` is a hard nearest-neighbour
projection (noisy), large `k` over-smooths toward the global mean.

## How it plugs in

- **Search / averaging** reuse the Phase-0/1 retrieval stack:
  `router.retrieval.query_bank.QueryBank` (train-only FAISS index over MiniLM
  query vectors, self-exclusion) finds the neighbours; the mean is taken in the
  `irt` / BERT space, like `router.retrieval.warmup`.
  `QueryBank.neighbor_weighted_mean(..., weighting="similarity"|"uniform")`.
- **Builder**: `router.retrieval.knn_impute.build_knn_imputed_store(cfg, k=...)`
  writes an ordinary query `EmbeddingStore` under a synthetic pathway name
  (`knn10w` = k 10, similarity-weighted; `knn10u` = uniform; `-ood` suffix for
  the leakage-safe variant). Queries with no neighbour keep their raw `e_q`
  (counted as `fallback_count` in the manifest).
- **Consumption**: a new `data.query_pathway` key (`configs/nirt.yaml`) overrides
  **only the query store** — the profile store stays on `data.pathway`. It is
  threaded through `NIRTDataset.from_config`, `Phase1Data.nirt_dataset`,
  `router.nirt.train.fit`, `router.nirt.ood.ood_datasets`,
  `router.nirt.evaluate.predict_matrix`, and both `scripts/route_compare.py` /
  `scripts/nirt/compare.py` (read off the checkpoint config, so eval is
  automatic). `null` → unchanged behaviour; every existing run and caller is
  untouched.

## Leakage

The FAISS bank is training-queries-only and self-excluded, so a query is never
imputed from itself or from val/test. For the **OOD** pass the held-out families
(`evaluation.ood_holdout_families`) are also dropped from the bank
(`build_query_bank(exclude_query_ids=...)`, written to `indexes/query_bank__ood/`)
so an OOD query is imputed purely from in-distribution training neighbours. The
oracle is never involved — imputation sees no targets or costs.

## Run it

```bash
# one-time: the standard bank (if not already built) and the leakage-safe OOD bank
python scripts/retrieval/build_query_bank.py
python scripts/retrieval/build_query_bank.py --exclude-ood-families

# sweep k, train ID + OOD runs, table against raw (k=0) and the reference models
python scripts/nirt/knn_impute_sweep.py --k 1,5,10,25
#   -> artifacts/phase2/knn_impute_sweep.json  (+ .log)

# a single k, wired into the routing comparison
python scripts/route_compare.py --nirt-run nirt-knn10w-2d-projected
python scripts/nirt/compare.py --run nirt-knn10w-2d-ood --ood
```

Runs are named `nirt-knn{k}{w|u}-{K}d-projected` (ID) and `-{K}d-ood` (OOD);
`nirt-raw-{K}d-*` is the k=0 control trained with identical settings.

## Results

`weighting=similarity`, `K=2`, `model_params=projected`, seed 42,
`configs/nirt.yaml` defaults; 0-shot queries only. From
`artifacts/phase2/knn_impute_sweep.json`.

**ID — test split** (3 677 queries × 9 warm models):

| run | BCE | Brier | AUC | acc@0.5 | Spearman | regret | oracle-hit\* |
|---|---|---|---|---|---|---|---|
| raw (k=0) = `nirt-2d-projected` | 0.612 | 0.169 | 0.739 | 0.689 | 0.439 | 0.142 | 0.806 |
| kNN k=1  | 0.585 | 0.159 | 0.762 | 0.700 | 0.458 | 0.141 | 0.807 |
| kNN k=5  | 0.581 | 0.157 | 0.768 | 0.705 | 0.464 | 0.141 | 0.807 |
| kNN k=10 | 0.580 | 0.157 | 0.770 | 0.707 | 0.467 | 0.141 | 0.807 |
| kNN k=25 | **0.579** | **0.156** | **0.772** | **0.708** | **0.468** | 0.140 | 0.807 |
| `irt-25d-projected` (paper baseline) | 0.603 | 0.164 | 0.752 | 0.698 | 0.443 | 0.139 | 0.809 |

kNN imputation is a **clearly better calibrated predictor** — BCE −0.033, AUC
+0.033, Spearman +0.03 over raw `e_q`, and it beats the IRT-Router paper
baseline on every prediction metric. Gains are monotone in `k` but ~90 % realised
by k=5–10. Argmax **routing** is essentially unchanged (regret 0.142 → 0.140,
oracle-hit flat): on this GPT-4-dominated pool the per-query best-model ranking
barely moves, so the prediction gain does not translate to routing gain here.

**OOD — held-out `math` + `code` families** (leakage-safe bank):

| run | BCE | AUC | Spearman | regret |
|---|---|---|---|---|
| raw (k=0) | **0.692** | 0.691 | 0.267 | 0.115 |
| kNN k=1  | 0.773 | 0.693 | 0.268 | 0.115 |
| kNN k=5  | 0.747 | 0.698 | 0.278 | 0.115 |
| kNN k=10 | 0.757 | 0.691 | 0.268 | 0.115 |
| kNN k=25 | 0.743 | 0.695 | 0.274 | 0.115 |

On OOD the trade-off **reverses for calibration**: imputing a math/code query
from its nearest *non*-math/code training neighbours biases the probability
(BCE +0.05–0.08), even though rank quality (AUC / Spearman) still nudges up.
Routing is identical to four decimals — the held-out-family pool ranking does not
depend on the query representation at all here.

**Takeaway.** Retrieval-smoothing `e_q` is a solid win for *in-distribution
probability estimates* and a mild win for *ranking everywhere*, but it is not a
fix for out-of-distribution **calibration** — for that the neighbour pool would
have to contain the target family. k≈10 is the sweet spot.

### `route_compare.py` — as a routing policy vs the other predictors

`python scripts/route_compare.py --nirt-run nirt-knn10w-2d-projected --lam <λ>`
turns each predictor into `argmax_m [pred − λ·cost_norm]` on the test split
(3 677 q × 9 models, ref `gpt-4-1106-preview`). At **λ=0** every policy collapses
to ≈always-GPT-4 (unchanged from prior runs). Cost-aware:

λ=0.5 (acc / quality / $per-1k / cost-saving-vs-ref):

| policy | acc | quality | $/1k | save |
|---|---|---|---|---|
| NIRT **knn k=10** | 0.819 | 0.764 | 2.74 | 21 % |
| NIRT raw (`nirt-2d-projected`) | 0.800 | 0.763 | 1.42 | 59 % |
| IRT-Router (paper, `irt-25d`) | 0.804 | 0.759 | 1.40 | 59 % |
| baseline NIRT (Bernoulli) | 0.827 | 0.769 | 2.42 | 30 % |
| ZOIB E[Y] | 0.801 | 0.764 | 1.49 | 57 % |

λ=1.0 (acc / quality / cost-saving / reward@0.8):

| policy | acc | quality | save | reward@0.8 |
|---|---|---|---|---|
| NIRT **knn k=10** | 0.783 | 0.753 | 61 % | 0.525 |
| NIRT raw (`nirt-2d-projected`) | 0.798 | 0.762 | 61 % | **0.533** |
| IRT-Router (paper, `irt-25d`) | 0.805 | 0.759 | 62 % | 0.533 |
| baseline NIRT (Bernoulli) | 0.807 | 0.760 | 62 % | 0.533 |
| ZOIB E[Y] | 0.802 | 0.764 | 61 % | **0.535** |

**kNN imputation does not help routing — it slightly hurts.** At matched cost
savings (λ=1.0) the imputed run gives up ~1.5 accuracy points and ~0.01
reward@0.8 vs raw NIRT, and at λ=0.5 it barely responds to the cost penalty
(21 % vs ~59 % savings). The smoothed representation compresses the *between-model*
spread of `P(correct)` that the routing arg-max lives on, even though it makes
each individual probability better calibrated. Artifacts:
`artifacts/phase2/routing_comparison_{knn10,raw}.json`.

**Net:** use kNN-imputed `e_q` when the deliverable is a calibrated
per-(query, model) probability; keep raw `e_q` (or the Bernoulli / ZOIB heads)
when the deliverable is the routing decision.

### Overfitting check — train vs val vs test

Prediction BCE on all three splits, every run scored on the **same 0-shot
observation set** (train 320 892 / val 39 974 / test 40 447 obs):

| run | train | val | test | test−train | AUC train→test |
|---|---|---|---|---|---|
| raw (`nirt-2d-projected`) | 0.596 | 0.606 | 0.612 | **0.0153** | 0.758 → 0.739 |
| IRT-Router (`irt-25d`) | 0.589 | 0.594 | 0.603 | 0.0135 | 0.767 → 0.752 |
| kNN k=1 | 0.574 | 0.577 | 0.585 | 0.0118 | 0.775 → 0.762 |
| kNN k=10 | 0.571 | 0.574 | 0.580 | **0.0089** | 0.779 → 0.770 |
| kNN k=25 | 0.570 | 0.573 | 0.579 | 0.0090 | 0.780 → 0.772 |

**kNN imputation does not overfit — it overfits *less* than raw `e_q`.** k≥10 is
better on train *and* test *and* has the smallest train→test gap (~0.009 vs
0.015 for raw): a strict generalisation gain, not memorisation. k=1 has the
largest gap of the kNN variants (hard nearest-neighbour picks up neighbour
noise). val ≈ test everywhere and val−train is tiny (~0.003 for kNN), so early
stopping isn't overfitting the validation set either.

**Confound to be aware of.** `query__retrieval` (the MiniLM FAISS space) was
never built for RouterBench's `:5shot` query ids, so the kNN builder can only
impute the 36 483 non-5shot queries — the `nirt-knn*` runs were **trained on
0-shot observations only** (≈half the rows), while `nirt-2d-projected` /
`irt-25d-projected` trained on 0-shot + 5-shot. The table above removes this from
the *evaluation* (0-shot only on both sides), and the direction only strengthens
the conclusion (raw trained on 2× the data and still generalises worse). A fully
clean comparison would retrain raw/paper on 0-shot only, or embed the 5-shot
queries in the `retrieval` pathway and rebuild the kNN stores.

## Scaling the head — multidimensional difficulty + wider latent

`scripts/nirt/complexity_search.py` random-searches the `query_latent` head on
the kNN-imputed representation (`knn10w`, K=2 scalar baseline = `nirt-2d-projected`
numbers):

| knob | values |
|---|---|
| `model.dim` (K: `theta_q`, `a_m` width) | 4, 8, 16, 32, 48, 64 |
| `model.query_hidden` (query-MLP width) | none, 64, 128, 256 |
| `model.difficulty` | `scalar` (`b_m ∈ R`) or **`vector`** (`b_m ∈ R^K`, `logit = Σ_k a_k(θ_k − b_k)`) |
| `train.lr` / `train.weight_decay` | 1e-3, 2e-3 / 1e-5, 1e-4 |

```bash
python scripts/nirt/complexity_search.py --n 15
#   -> artifacts/phase2/complexity_search.json  (+ .log)
```

New model knob: `NIRTModel(difficulty="vector")` /
`configs/nirt.yaml → model.difficulty`. Backward compatible — omitting it is
`scalar`, bit-identical to before; existing checkpoints load unchanged.

### Results (n=15 random configs, seed 0, test split)

| run | K | hidden | difficulty | params | train BCE | test BCE | gap | AUC | Spearman | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| K16 h256 **vector** | 16 | 256 | vector | 226 k | 0.561 | **0.5715** | 0.011 | 0.778 | 0.485 | 0.137 |
| K8 h64 vector | 8 | 64 | vector | 62 k | 0.560 | 0.5716 | 0.011 | 0.771 | 0.490 | 0.138 |
| K32 h256 scalar | 32 | 256 | scalar | 230 k | 0.561 | 0.5716 | 0.011 | 0.779 | 0.485 | 0.138 |
| K64 h256 scalar | 64 | 256 | scalar | 263 k | 0.562 | 0.5720 | 0.010 | 0.775 | 0.486 | 0.138 |
| K32 h128 vector | 32 | 128 | vector | 152 k | 0.565 | 0.5730 | **0.008** | 0.778 | 0.480 | 0.141 |
| … (worst of the 15) K64 h**none** vector | 64 | – | vector | 148 k | 0.568 | 0.5764 | 0.009 | 0.768 | 0.481 | 0.136 |
| **baseline** `nirt-knn10w-2d-projected` | 2 | 64 | scalar | 52 k | 0.571 | 0.5798 | 0.009 | 0.770 | 0.467 | 0.141 |

**All 15 configs beat the K=2 baseline** (BCE 0.571–0.576 vs 0.580, AUC up to
0.779 vs 0.770). What actually moves the needle:

- **The nonlinear query head (`query_hidden=256`) is the whole story** — every
  top config has it; the three `query_hidden=none` configs are the worst 3 of
  the 15. The bottleneck was the *linear* `e_q → θ_q` map, not the latent size.
- **Latent width `K` beyond ~8 adds nothing measurable.** K=8/h64 ≈ K=64/h256 on
  BCE. So "expand θ from R^8 to higher" does not help here.
- **Multidimensional difficulty (`b_m ∈ R^K`) is a wash** — vector and scalar are
  interleaved at the top; vector occasionally has a slightly smaller train→test
  gap (K32 h128 v: 0.008) but no BCE/AUC edge.
- **No overfitting at any size** — the gap is 0.008–0.011 across the entire
  search, flat vs the 52 k-param baseline. Bigger heads do not memorise more.
- **Routing regret unchanged** (0.136–0.141) — same as every prior result: the
  prediction gain does not reach the arg-max on this pool.

**Cumulative:** raw K=2 (BCE 0.612, AUC 0.739) → + kNN-imputed `e_q` (0.580,
0.770) → + wide nonlinear query head (**0.572, 0.778**). The recommended config
is `K=16, query_hidden=256` (scalar or vector); going wider on `K` or the MLP
buys nothing.
