# Candidate-pool expansion — plan + results

**Question this workstream answers:** the Phase 2 routing analysis showed the
router's cost savings are capped by the *candidate pool*, not the predictor. The
9 "warm" RouterBench models are GPT-4-dominated (GPT-4 test acc 0.84; next best
0.69; the two models priced ≤ 1/30 of GPT-4 score 0.29 / 0.45). A *perfect*
router on that pool saves only **−76 %** vs always-GPT-4, so the "1/30 the cost"
(−97 %) framing is unreachable regardless of router quality.

So we expand the pool in **isolated phases** and re-run the *identical* evaluation
battery each time, recording how the pool's routing ceiling and the ZOIB router's
achievable savings move. The predictor (frozen ZOIB head + Bernoulli control) is
**never retuned** — the pool is the only variable. Orthogonal to the Phase 3
(correctness / length coupling) track.

---

## Discipline (every phase)

1. **One variable per phase.** Add exactly the models / data the phase specifies.
   Do not also change the model config, `theta_dim`, encoder, regularisation,
   split methodology, or the training block.
2. **Same model, same seed.** Train the Bernoulli baseline and the ZOIB head with
   the *unchanged* `configs/phase1.yaml` / `configs/phase2.yaml` and `seed: 42`.
3. **Identical eval battery** (below) re-run verbatim each phase.
4. **Ablation within the phase.** Also evaluate the pool *without* the newly added
   model(s) so each addition's marginal contribution is attributable.
5. **No leakage on new data.** New models get a warm/cold split by the existing
   deterministic hash; new *sources* get their queries assigned by the existing
   content-hash group split. Verify `training.data.splits.check_leakage` is clean
   and no new-source query collides with an existing test query.
6. **Cost comparability.** RouterBench carries a precomputed per-(model, query)
   USD cost. A model added another way needs a documented price basis
   (`configs/model_prices.yaml`, dated snapshot) or it is excluded from the
   *routing* (not the *prediction*) eval, and this is recorded.
7. **Reproducibility.** Every phase writes a provenance block and a second-seed
   sanity check on the ZOIB run.
8. **Record before moving on.** Append the phase's ledger row and fill its results
   section before starting the next phase.

## The fixed evaluation battery

Run **all** of this, identically, at the end of every phase, on the **test**
split, written under `artifacts/pool_expansion/<phase>/`:

* **1a — pool description** (`pool.json`): `model_id | sources | n_obs | test acc
  | $/1k | warm`, plus GPT-4:cheapest ratio and the count of models priced ≤ 1/30
  of the most expensive and their mean accuracy.
* **1b — prediction quality** (`prediction.json`): ZOIB (NLL / MAE / RMSE /
  mean-ECE / 50-90 % coverage / boundary calibration / θ eff-rank / per-family) +
  Bernoulli control. `scripts/nirt/baseline/evaluate.py --checkpoint <ck> --split test`.
* **1c — routing** (`routing.json`, the headline): `scripts/route_compare.py
  --split test` — the policy table (oracle / best-fixed / cheapest / NIRT /
  baseline-NIRT / ZOIB E[Y] / ZOIB LCB) with accuracy, graded quality, $/1k,
  `cost_savings_vs_ref`, `d_accuracy_vs_ref`; the ZOIB λ frontier; and AIQ.
* **1d — derived headline numbers**: `oracle_cost_saving_ceiling = 1 -
  oracle_cost / best-single-model_cost` (oracle = cheapest model attaining each
  query's max true score, `routing.oracle_choice`, ties broken by cost);
  `router_saving_at_minus{1,3}pt` — λ is chosen on the **validation** split (cheapest
  validation-frontier point within {1,3} acc pts of the best single model's validation
  accuracy) and the saving/accuracy are then measured on test at that λ, so the number
  is one a deployed router could have picked in advance. The same rule applied directly
  to the test frontier is kept as `router_saving_at_minus{1,3}pt_test_ceiling` — an
  optimistic ceiling, not a headline. Also: the fraction of queries the oracle routes off
  the best model + its model mix; ZOIB escalation rate at λ=0 and at the −3 pt point
  (validation-chosen λ, with the test-chosen λ as `*_test_ceiling`).
  `naive_argmax_oracle` (the pre-fix `true.argmax(1)` oracle) is recorded for
  comparison but does not feed the ledger.
* **1e — cold-start status** (`cold_start.json`): if a cold arm exists, a
  `projected` ZOIB run + whether it beats the global-mean BCE baseline.
* **1f — ablation** (`ablation.json`): 1c re-run with the phase's addition removed
  (predictor unchanged), giving the Δ in oracle ceiling and ZOIB −3 pt saving.
* **provenance.json** — git sha, seed, pool composition, and hashes of the config the
  battery ran under (file + resolved content), the split files under that config's
  `paths.splits` (incl. `ood.json` when present), and the checkpoints. **battery.json** — all of the above in one document.

Each run appends one row to `artifacts/pool_expansion/ledger.json`, re-rendering
[pool_expansion_results.md](pool_expansion_results.md)
(`evaluation.pool_expansion.render_ledger_md`). The trajectory of
`oracle_cost_saving_ceiling` and `zoib_saving_at_minus3pt` across rows **is the
deliverable**.

## Running a phase

```bash
python scripts/pool_expansion/run_phase.py --phase E0
python scripts/pool_expansion/run_phase.py --phase E1 \
    --added-models yi-34b-chat code-llama-34b-instruct --sources routerbench \
    --notes "un-hold the two RouterBench cold-start models"
```

### Planned phases

| phase | change |
|---|---|
| **E0** | lock the 9-warm-model baseline (no new models) |
| **E1** | un-hold the two RouterBench cold-start models (`yi-34b-chat`, `code-llama-34b-instruct`) via `split.force_warm` |
| **E2** | RouterBench 5-shot (`sources.routerbench.shots: ["0shot","5shot"]`) — 0-/5-shot as separate items sharing a split group |
| **E3** | *(superseded — see below)* modern cheap-competent models via lm-eval-harness |
| **E4** | an external benchmark source (RouteLLM / MixInstruct / the `evaluations/LLMRouterBench` submodule) |
| **E5** | profile-only cold-start models (only once `n_warm ≥ ~20`) |

---

## E0 — the frozen reference (2026-08-31)

9 warm RouterBench models, no retraining. Prediction numbers reproduce the Phase 2
report exactly.

| metric | value |
| --- | --- |
| ZOIB test NLL / MAE / mean-ECE | 0.388 / 0.337 / 0.012 |
| GPT-4 : cheapest cost ratio | 73× |
| **oracle cost-saving ceiling** | **−91 %** (oracle $0.31 vs GPT-4 $3.45 / 1k) |
| &nbsp;&nbsp;naive-argmax oracle (pre-fix) | −76 % ($0.83 / 1k) |
| ZOIB saving @ within −1 pt | −15 % (acc 0.829, λ 0.1) |
| ZOIB saving @ within −3 pt | −49 % (acc 0.808, λ 0.35) |
| ZOIB AIQ improvement over linear | +0.213 |
| cold-start (projected) beats global-mean BCE? | no — 0.757 vs 0.722 |

**Oracle tie-break fix (2026-09-01).** The original oracle was `true.argmax(1)`,
which on the many queries where several models score 1.0 picks the *first column*
(`claude-instant-v1`, $0.28 / 1k), not the cheapest — inflating oracle cost.
`routing.oracle_choice` now breaks ties by cost, matching RouterBench's own
oracle. Same oracle *quality*; the ceiling goes −76 % → **−91 %**.

So the existing pool's honest ceiling is already close to the "1/30" (−97 %)
claim — the remaining gap is the hardest queries where only GPT-4 is right. The
ZOIB router still only realises −49 % at −3 pt: its escalation decisions are
imperfect, not the pool.

## E1 — un-hold the two RouterBench cold-start models (2026-09-01)

`split.force_warm = [yi-34b-chat, code-llama-34b-instruct]`. Splits + NIRT dataset
rebuilt (RouterBench **test query set bit-identical to E0**: 3677 queries;
leakage clean). Bernoulli + ZOIB retrained on the unchanged configs, seed 42;
second-seed ZOIB (123) val NLL 0.3730 vs 0.3738 — reproduces. Pool 9 → **11**.

| metric | E0 (9) | E1 (11) |
| --- | --- | --- |
| ZOIB test NLL / MAE | 0.388 / 0.337 | **0.372 / 0.321** |
| oracle cost-saving ceiling | −91.1 % | **−92.6 %** |
| ZOIB saving @ within −3 pt | −48.6 % (λ 0.35) | −46.1 % (λ 0.30) |
| ZOIB AIQ improvement | +0.213 | **+0.222** |

New models on the cost/quality map: **`yi-34b-chat` acc 0.674 @ $0.19 / 1k** —
the cheap-and-good model the pool lacked; oracle routes it 520 / 3677 queries
(14 %). `code-llama-34b-instruct` acc 0.21 @ $0.18 — weak, oracle uses it 44
times.

**Ablation (leave-one-out on the routing matrices, predictor unchanged):**
un-holding `yi-34b-chat` is the whole of E1's movement (+1.2 pp of ceiling,
~+0.8 pp of achievable router saving, plus a real predictor-quality gain NLL
0.388 → 0.372 from ~2× the correctness observations). `code-llama-34b`
contributes essentially nothing.

## E2 — RouterBench 5-shot (2026-09-01)

`sources.routerbench.shots: ["0shot", "5shot"]`. **Item-identity policy:** a
question's 0-shot and 5-shot runs are *separate items* — the 5-shot `query_id`
gets a `:5shot` suffix — but both carry the **underlying question text** (the
0-shot prompt), because the model has no shot input. Consequences: 0-/5-shot of
one question share a content hash → the same split group (no straddle); every
0-shot `query_id` is byte-identical to E0/E1; the predictor sees 2× observations
of each question's difficulty while the routing matrices carry the true per-shot
scores. Same 11 models. Data: 802 626 obs, 72 966 queries.

| metric | E1 | E2 |
| --- | --- | --- |
| test queries | 3 677 | 7 354 (0-shot + 5-shot) |
| ZOIB test NLL / MAE | 0.372 / 0.321 | **0.413 / 0.351** |
| **oracle cost-saving ceiling** | −92.6 % | **−93.5 %** |
| ZOIB saving @ within −1 pt | −11.2 % | **−15.6 %** (λ 0.15) |
| ZOIB AIQ improvement | +0.222 | +0.180 |

Few-shot dramatically lifts the weak open models (code-llama +29.6 pt,
llama-2-70b +27.5 pt; the Claude models actually *lose* accuracy — claude-v2
−12.9 pt) and *slightly* raises the ceiling. The effect is modest because GPT-4 is
*also* lifted by few-shot (0.837 → 0.869), so the hardest queries still need it;
and the predictor **degrades** (NLL 0.372 → 0.413) — the model cannot see the shot
count, so the same question with two different outcomes is irreducible noise to
it. A shot feature would recover that, but that is a model change, out of scope
here (it later shipped as `query_features` — see
[nirt_model.md](nirt_model.md#the-routerbench-0-5-shot-confound--the-biggest-single-lever)).

**Ablation** — routing on the 0-shot subset only: oracle ceiling **−92.6 %**
(= E1 exactly), so the 5-shot observations contribute **+0.9 pp** of ceiling.

## E3 — modern cheap-competent models via lm-eval-harness (SUPERSEDED 2026-09-02)

> **Superseded by the IRT-Router benchmark suite** (`configs/irt_router.yaml`,
> [irt_router.md](irt_router.md)). Rather than run 5 lm-eval models against
> RouterBench and join on question identity, we adopt the IRT-Router paper's
> published 20-LLM × 12-dataset response data wholesale (modern cheap/competent
> models already included) with a proper in-distribution / out-of-distribution
> split. The E3 scaffolding (`src/router/data/benchmark_id.py`,
> `scripts/data/run_lm_harness.py`, `configs/model_prices.yaml` E3 rows) is left
> in place, parked — not deleted.

Original plan (not executed): target the empty cost/quality quadrant with 5
open-weight models (`Llama-3.1-8B`, `Qwen2.5-7B/14B`, `Gemma-2-9B`,
`Mistral-Nemo`) run through lm-evaluation-harness on GPU. Cross-source join
(`src/router/data/benchmark_id.py`) on **question identity, not prompt text**
(`benchmark:{task}:{index}`), valid only for RouterBench's standard-benchmark
subset (MMLU / HellaSwag / WinoGrande / ARC-Challenge ≈ 27k of 36k 0-shot
questions). Cost basis via `training.data.pricing`
(`cost = in_tok·price_in + out_tok·price_out` from a dated snapshot; a model with
no price is dropped from routing, kept in prediction).
