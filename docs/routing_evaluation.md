# Oracle-routing evaluation

The NIRT / ZOIB model is a **response model**, not an oracle classifier. It
estimates `E[Y | q, m]` (expected RouterBench performance). The **router** then
picks `m_hat(q) = argmax_m E[Y | q, m]`. The **oracle** `m*(q) = argmax_m y(q, m)`
is an **evaluation target only**.

```
RESPONSE MODELING        ROUTING DECISION       ACTUAL TEST OUTCOME
   P(Y | q, m)      ->   selected model    ->   y(q, m_hat)   ->  compare vs oracle
```

Predictive quality (NLL, calibration) and routing quality are reported
**separately** — a model can predict probabilities better yet route worse, so
`lower NLL != better router`.

| concern | where |
|---|---|
| predictive: ZOIB NLL / calibration | `scripts/evaluate.py`, `router.nirt.continuous_eval` |
| predictive: Bernoulli BCE / AUC | `scripts/nirt/eval_nirt.py`, `router.nirt.evaluate.evaluate_split` |
| **routing: oracle hit / regret / cost-aware** | **`scripts/nirt/route_eval.py`, `router.nirt.routing_eval`** |
| routing policy table + λ frontier + AIQ | `scripts/route_compare.py`, `router.nirt.routing` |

## Oracle labels (`routing_eval.oracle_labels`)

Given a dense `[query_id × model_id]` matrix of the **actual** observed target
(the chance-corrected graded score `y_soft` — the same target the router is
trained and evaluated against — §12), for every `(query, model)`:

```
oracle_score     = max_m  actual_score(q, m)          # within THIS candidate pool
oracle_flag      = int(actual_score >= oracle_score - tol)   # tie-INCLUSIVE
oracle_rank      = dense rank of actual_score, 1 = best, ties share a rank
oracle_model_id  = deterministic single winner: cheapest flagged model,
                   then lexicographically smallest model_id
n_oracle_ties    = number of models attaining the best score
```

**Ties** (§3): `oracle_flag` is 1 for *every* model that attains the best score,
so multiple rows per query can have `oracle_flag = 1`. The single
`oracle_model_id` is a documented deterministic tie-break — never left to pandas
sort order or dict iteration order.

**Pool-relative** (§2): the oracle is `argmax` over exactly the models in
`true_df.columns`. E0 / E1 / E2 each get their own oracle by passing their own
model subset (`eval_matrices(d, split, models=pool)`), *not* a global "best LLM".

## Routing metrics (`routing_eval.routing_evaluation`)

For each test query: predict `ŷ(q, m)` for every eligible candidate, select
`m_hat(q) = argmax_m ŷ(q, m)`, look up the **actual** `y(q, m_hat)`, compare to
`y(q, m*)`.

| metric | meaning |
|---|---|
| `oracle_hit_rate` | `mean( m_hat == oracle_model_id )` — exact match to the (cost-broken) oracle |
| `oracle_hit_rate_any_best` | `mean( y(q, m_hat) >= oracle_score )` — tie-aware; picked *a* best model |
| `mean/median/p90_regret` | `regret(q) = y(q, m*) − y(q, m_hat)` (clipped at 0) |
| `zero_regret_rate` | fraction with `regret <= 1e-9` |
| `within_tolerance_rate` | fraction with `regret <= tolerance` (configurable, default 0.01) |
| `mean_selected_quality` / `mean_oracle_quality` | achieved vs ceiling |
| `n_candidates` | mean / min / max eligible models per query |

**Hit rate and regret are different** (§6). With heavily-tied targets (binary MC
correctness) the router routinely picks a model that is *tied-best but not the
cheapest-best* → `oracle_hit = 0`, `regret ≈ 0`. Read `oracle_hit_rate_any_best`
and the regret distribution, not `oracle_hit_rate` alone. On the IRT-Router test
split the K=25 model scores `oracle_hit_rate` **2 %** but `any_best` **85 %** with
median regret 0.

## Cost-aware oracle (`§8`)

Kept out of the quality oracle. A separate utility

```
U(q, m) = y(q, m) − λ · C(m)          C(m) = mean cost of model m
m*_λ(q) = argmax_m U(q, m)
```

`routing_evaluation(..., lam=λ)` selects cost-aware (`argmax_m [ŷ − λ C(m)]`) and
reports, side by side, the `quality_oracle` (quality + its cost) and the
`cost_aware_oracle` (quality + its cost), plus `mean_utility_regret`. `λ` is a CLI
argument, never baked into the model.

## Strategy comparison (`routing_eval.compare_routing_strategies`, `§9`)

| id | strategy | learnable? |
|---|---|---|
| A | NIRT predicted-quality router | yes (this is the NIRT run) |
| B | NIRT cost-aware router (`− λ C(m)`) | yes |
| C | hard oracle upper bound (route on `y`) | no — max achievable within the pool |
| D | direct hard-oracle **classifier** (`routing_eval.oracle_classifier_matrix`) | separate baseline — multinomial logistic on `e_q`, labels = train-split `argmax_m y_train`; **train split only** |
| E | soft-oracle distillation target (`routing_eval.soft_oracle_targets`, `p*(m|q) = softmax(y/τ)`) | helper provided; model not trained here |

## Leakage rules (`§11`) — critical

The oracle is derived from **observed outcomes**. It must never flow backward:

* `oracle_flag` / `oracle_model_id` are **never** in the query/model feature
  tensors — `NIRTDataset[i]` and `build_arrays` expose only `e_q, e_m, target,
  cost, metric, source` (asserted in `tests/test_routing_eval.py`).
* test-set oracle outcomes are **not** used in training, embeddings,
  hyper-parameter selection, or candidate selection.
* the oracle-classifier baseline (D) is fit on **train-split** oracle labels
  only; validation / test oracle labels are for evaluation.
* `oracle_labels_<split>.parquet` is an evaluation artifact — it is **not** read
  by any training path.

## Running it

```bash
python scripts/nirt/route_eval.py \
    --run irtrouter-irt-25d-projected --split test \
    --config configs/irt_router.yaml \
    --lam 0.3 --tolerance 0.02 \
    --zoib artifacts/irt_router/zoib --bernoulli artifacts/irt_router/bernoulli \
    --oracle-classifier
```

Writes, under `<runs_dir>/<run>/`:

* `route_eval_<split>.json` — all metrics + the strategy comparison
* `route_eval_<split>_per_query.csv` — one row per query: `selected_model_id,
  selected_predicted_score, selected_actual_score, oracle_model_id, oracle_score,
  oracle_hit, oracle_regret` (+ cost-aware columns)
* `oracle_labels_<split>.parquet` — the full tie-aware oracle-label table

### Example (IRT-Router `irt-25d-projected`, test split, 20 models, 10 458 queries)

```
Quality routing (argmax predicted score)
  Selected quality:   0.8164     Oracle quality:  0.9525
  Oracle hit rate:     2.0%      (any-best 85.4%)
  Mean regret:        0.1362     Median: 0.0000     P90: 1.0000
  Zero-regret rate:  85.4%
Cost-aware routing (lambda = 0.300)
  Selected cost:     $0.478 / 1k     Utility regret: 0.1362
  quality-oracle:    q 0.9525  $0.020/1k
  cost-aware-oracle: q 0.9525  $0.022/1k

strategy comparison (regret / any-best-hit / selected quality)
  NIRT (quality)                 0.136 / 85.4% / 0.816
  ZOIB E[Y] (quality)            0.134 / 85.7% / 0.819
  Bernoulli (quality)            0.127 / 86.4% / 0.826
  oracle-classifier (baseline D) 0.145 / 84.4% / 0.807
  hard oracle (upper bound)      0.000 / 100 % / 0.953
  random                         0.348 / 63.9% / 0.605
```
