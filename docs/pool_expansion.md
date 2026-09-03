# Candidate-pool expansion

**Question this workstream answers:** the Phase 2 routing analysis showed the
router's cost savings are capped by the *candidate pool*, not the predictor. The
9 "warm" RouterBench models are GPT-4-dominated (GPT-4 test acc 0.84; next best
0.69; the two models priced <= 1/30 of GPT-4 score 0.29 / 0.45). A *perfect*
router on that pool saves only **-76 %** vs always-GPT-4, so the "1/30 the cost"
(-97 %) framing is unreachable regardless of router quality.

So we expand the pool in **isolated phases** and re-run the *identical*
evaluation battery each time, recording how the pool's routing ceiling and the
ZOIB router's achievable savings move. The predictor (frozen ZOIB head +
Phase 1 Bernoulli control) is **never retuned** -- the pool is the only variable.

This is orthogonal to the Phase 3 (correctness / length coupling) track.

## Running a phase

```
python scripts/pool_expansion/run_phase.py --phase E0
python scripts/pool_expansion/run_phase.py --phase E1 \
    --added-models yi-34b-chat code-llama-34b-instruct --sources routerbench \
    --notes "un-hold the two RouterBench cold-start models"
```

Each run writes `artifacts/pool_expansion/<phase>/`:

| file | content |
| --- | --- |
| `pool.json`        | **1a** model table: `model_id \| sources \| n_obs \| test acc \| $/1k \| warm`, plus GPT-4:cheapest ratio and the cheap-model quadrant |
| `prediction.json`  | **1b** ZOIB (NLL / MAE / RMSE / mean-ECE / 50-90 % coverage / boundary calibration / theta eff-rank / per-family) + Bernoulli control |
| `routing.json`     | **1c** policy table (oracle / best-fixed / cheapest / NIRT / baseline-NIRT / ZOIB E[Y] / ZOIB LCB) + ZOIB lambda frontier + AIQ, and **1d** derived headline numbers |
| `cold_start.json`  | **1e** projected-ZOIB cold-start vs the global-mean baseline (BCE), projected theta eff-rank |
| `ablation.json`    | **1f** the routing block re-run with the phase's addition removed -- newly added models (E1, leave-one-out) or a query subset (E2, `:5shot`) -- predictor unchanged |
| `shot_breakdown.json` | (multi-shot phases) per-model test accuracy on the 0-shot vs 5-shot subset |
| `provenance.json`  | git sha, seed, pool composition, config + split + checkpoint hashes |
| `battery.json`     | all of the above in one document |

and appends one row to `artifacts/pool_expansion/ledger.json`, re-rendering
`docs/pool_expansion_results.md`.

## Derived headline numbers (1d)

* **`oracle_cost_saving_ceiling`** = `1 - oracle_cost / best-single-model_cost`,
  where oracle = cheapest model attaining each query's max true score
  (`routing.oracle_choice`; ties broken by cost, not column order). Same oracle
  as `scripts/route_compare.py`. This is the trajectory the workstream tracks.
* **`router_saving_at_minus{1,3}pt`** = the largest cost-saving vs the reference
  model on the ZOIB lambda frontier whose accuracy is within {1, 3} points of the
  best single model. `null` if no frontier point qualifies.
* **`naive_argmax_oracle`** (secondary) = the pre-fix `true.argmax(1)` oracle,
  recorded so the cost of the column-order tie-break stays visible. Does **not**
  feed the ledger.
* **oracle off-best fraction** + **oracle model mix**; **ZOIB escalation rate**
  (share not sent to the best model) at lambda=0 and at the -3 pt operating point.

## E0 -- the frozen reference (2026-08-31)

9 warm RouterBench models, no retraining. Prediction numbers reproduce the
Phase 2 report exactly.

| metric | value |
| --- | --- |
| ZOIB test NLL / MAE / mean-ECE | 0.388 / 0.337 / 0.012 |
| GPT-4 : cheapest cost ratio | 73x |
| models priced <= 1/30 of GPT-4 | 2 (mean acc ~0.37) |
| **oracle cost-saving ceiling** | **-91 %** (oracle $0.31 vs GPT-4 $3.45 / 1k) |
| &nbsp;&nbsp;naive-argmax oracle (pre-fix, for comparison) | -76 % ($0.83 / 1k) |
| ZOIB saving @ within -1 pt | -15 % (acc 0.829, lambda 0.1) |
| ZOIB saving @ within -3 pt | -49 % (acc 0.808, lambda 0.35) |
| ZOIB saving @ within -3.7 pt (band optimum) | -63 % (acc 0.801, lambda 1.5) |
| ZOIB frontier point at lambda 0.5 (Phase 2 report) | -57 % (acc 0.801) |
| ZOIB AIQ improvement over linear | +0.213 |
| ZOIB escalation off GPT-4 @ lambda 0 / -3 pt | 9 % / 23 % |
| cold-start (projected) beats global-mean BCE? | no -- 0.757 vs 0.722 (2 cold models, unchanged wall) |

**Oracle tie-break fix (applied 2026-09-01).** The original oracle
(`scripts/route_compare.py`, `routing.routing_report`) was `true.argmax(1)`,
which on the many queries where several models all score 1.0 picks the *first
column* (`claude-instant-v1`, $0.28 / 1k), not the cheapest -- inflating the
oracle cost. `routing.oracle_choice` now breaks the tie by cost (cheapest model
attaining the per-query max score), matching RouterBench's own oracle. Same
oracle *quality*; the ceiling goes -76 % -> **-91 %**. The pre-fix number is
still recorded per phase as `naive_argmax_oracle`.

So the existing pool's honest ceiling is already close to the "1/30" (-97 %)
claim -- the remaining gap is the hardest queries where only GPT-4 is right. The
ZOIB router still only realises -49 % at -3 pt: its escalation decisions are
imperfect, not the pool. E1+ test whether adding cheap-and-competent models both
lifts the ceiling further and makes the router's job easier.

## E1 -- un-hold the two RouterBench cold-start models (2026-09-01)

`split.force_warm = [yi-34b-chat, code-llama-34b-instruct]`. Splits + NIRT dataset
rebuilt (RouterBench **test query set bit-identical** to E0: 3677 queries, 0
added/removed; leakage clean). Bernoulli + ZOIB + ZOIB-projected retrained on the
unchanged `configs/phase1.yaml` / `phase2.yaml`, seed 42; second-seed ZOIB
(seed 123) val NLL 0.3730 vs 0.3738 -- reproduces. Pool 9 -> **11**.

| metric | E0 (9) | E1 (11) |
| --- | --- | --- |
| ZOIB test NLL / MAE | 0.388 / 0.337 | **0.372 / 0.321** |
| oracle cost-saving ceiling | -91.1 % | **-92.6 %** |
| ZOIB saving @ within -1 pt | -14.7 % | -11.2 % |
| ZOIB saving @ within -3 pt | -48.6 % (lam 0.35) | -46.1 % (lam 0.30, -2.75 pt) |
| ZOIB AIQ improvement | +0.213 | **+0.222** |
| ZOIB escalation off GPT-4 @ lam 0 | 9 % | 16 % |
| cold-start (projected) beats global-mean BCE? | n/a (see note) | **paused** -- remaining cold models are Arena-only |

New models on the cost/quality map: **`yi-34b-chat` acc 0.674 @ $0.19 / 1k** --
the cheap-and-good model the pool lacked; oracle routes it 520 / 3677 queries
(14 %). `code-llama-34b-instruct` acc 0.21 @ $0.18 -- weak, oracle uses it 44
times.

**Ablation (leave-one-out on the routing matrices, predictor unchanged):**

| removed | oracle ceiling | d ceiling | d ZOIB -3pt saving |
| --- | --- | --- | --- |
| -- (full E1 pool) | -92.6 % | -- | -- |
| yi-34b-chat | -91.4 % | **+1.2 pp** | +0.8 pp |
| code-llama-34b-instruct | -92.4 % | +0.2 pp | +0.0 pp |
| both | -91.1 % (= E0) | +1.5 pp | +0.8 pp |

**Read:** un-holding `yi-34b-chat` is the whole of E1's movement -- +1.2 pp of
ceiling and ~+0.8 pp of achievable router saving, plus a real predictor-quality
gain (NLL 0.388 -> 0.372) from ~2x the correctness observations. `code-llama-34b`
contributes essentially nothing. The oracle ceiling is now -92.6 %, ~4.4 pp short
of the "1/30" (-97 %) claim, and the ZOIB router still leaves ~46 pp of that on
the table at -3 pt because its escalation is imperfect. Bigger cheap-and-competent
additions (E3) are where the ceiling should move materially.

*Note on E0 cold-start:* E1 reassigns the RouterBench cold arm (yi-34b,
code-llama) to warm, so re-running the E0 battery now reports cold-start
"skipped". E0's original 9-warm projected cold-start did **not** beat the
global-mean BCE baseline (0.757 vs 0.722), consistent with Phases 1-2; that
finding is preserved here and in `artifacts/phase2/`.

## E2 -- RouterBench 5-shot (2026-09-01)

`sources.routerbench.shots: ["0shot", "5shot"]`. **Item-identity policy:** a
question's 0-shot and 5-shot runs are *separate items* -- the 5-shot `query_id`
gets a `:5shot` suffix -- but both carry the **underlying question text** (the
0-shot prompt), because the model has no shot input and a 256-token embedding of
a few-shot prompt would be dominated by the shared exemplars. Consequences:
0-/5-shot of one question share a content hash -> the same split group (no
straddle); every 0-shot `query_id` is byte-identical to E0/E1; the predictor
sees 2x observations of each question's difficulty while the routing matrices
carry the true per-shot scores. Same 11 models. Data: 802 626 obs, 72 966
queries. Bernoulli + ZOIB + ZOIB-projected retrained (unchanged configs,
seed 42); second-seed ZOIB val NLL 0.4170 vs 0.4171 -- reproduces.

| metric | E1 | E2 |
| --- | --- | --- |
| test queries | 3 677 | 7 354 (0-shot + 5-shot) |
| ZOIB test NLL / MAE | 0.372 / 0.321 | **0.413 / 0.351** |
| best single model acc (GPT-4) | 0.837 | 0.853 (5-shot GPT-4 = 0.869) |
| **oracle cost-saving ceiling** | -92.6 % | **-93.5 %** |
| ZOIB saving @ within -1 pt | -11.2 % | **-15.6 %** (lam 0.15) |
| ZOIB saving @ within -3 pt | -46.1 % | -46.7 % |
| ZOIB AIQ improvement | +0.222 | +0.180 |
| GPT-4 : cheapest cost ratio | 73x | 62x (5-shot prompts cost more) |

**5-shot accuracy shift, test set, per model** (mean **+9.0 pt**):

| model | 0-shot | 5-shot | d |
| --- | --- | --- | --- |
| code-llama-34b-instruct | 0.210 | 0.506 | **+29.6** |
| llama-2-70b-chat | 0.348 | 0.622 | **+27.5** |
| mistral-7b-instruct | 0.290 | 0.481 | +19.1 |
| wizardlm-13b-v1.2 | 0.451 | 0.553 | +10.2 |
| mixtral-8x7b-instruct | 0.565 | 0.660 | +9.5 |
| yi-34b-chat | 0.674 | 0.743 | +6.9 |
| gpt-3.5-turbo-1106 | 0.664 | 0.730 | +6.6 |
| gpt-4-1106-preview | 0.837 | 0.869 | +3.2 |
| claude-v1 | 0.674 | 0.691 | +1.7 |
| claude-instant-v1 | 0.634 | 0.615 | -1.8 |
| claude-v2 | 0.690 | 0.560 | **-12.9** |

**Ablation -- routing on the 0-shot subset only** (predictor unchanged):
oracle ceiling **-92.6 %** (= E1 exactly -- the 0-shot routing matrices are
identical), so the 5-shot observations contribute **+0.9 pp** of ceiling and
**+0.5 pp** of ZOIB -3 pt saving.

**Read:** few-shot dramatically lifts the weak open models (code-llama +29.6 pt,
llama-2-70b +27.5 pt) and *slightly* raises the ceiling (-92.6 % -> -93.5 %) and
the cheap operating point (router saves -16 % within -1 pt vs -11 % at E1). But
the effect on the ceiling is modest because on this pool GPT-4 is *also* lifted
by few-shot (0.837 -> 0.869), so the hardest queries still need it; and the
predictor **degrades** (NLL 0.372 -> 0.413) -- the model cannot see the shot
count, so the same question with two different outcomes is irreducible noise to
it. A shot feature would recover that, but that is a model change, not a pool
change, so it is out of scope here. The Claude models actually *lose* accuracy
with 5-shot (claude-v2 -12.9 pt), a known few-shot-format sensitivity.

## E3 -- modern cheap-competent models via lm-eval-harness (SUPERSEDED 2026-09-02)

> **Superseded by the IRT-Router benchmark suite** (`configs/irt_router.yaml`,
> `docs/irt_router.md`). Rather than run 5 lm-eval models against RouterBench and
> join on question identity, we adopt the IRT-Router paper's published 20-LLM x
> 12-dataset response data wholesale (modern cheap/competent models already
> included: Qwen2.5, Llama-3.1, GLM-4, DeepSeek, Gemini-1.5, Ministral, ...) with
> a proper in-distribution / out-of-distribution split. The E3 scaffolding
> (`src/router/data/benchmark_id.py`, `scripts/data/run_lm_harness.py`,
> `configs/model_prices.yaml` E3 rows) is left in place, parked -- not deleted.

Original plan (not executed): target the empty quadrant with 5 open-weight models
run through lm-evaluation-harness on Hugging Face Jobs (GPU):

| id | HF repo | price basis (`configs/model_prices.yaml`, snapshot 2026-09-01) |
| --- | --- | --- |
| llama-3.1-8b-instruct | meta-llama/Llama-3.1-8B-Instruct | $0.03 / $0.05 per 1M in/out |
| qwen2.5-7b-instruct | Qwen/Qwen2.5-7B-Instruct | $0.05 / $0.10 |
| qwen2.5-14b-instruct | Qwen/Qwen2.5-14B-Instruct | $0.20 / $0.30 |
| gemma-2-9b-it | google/gemma-2-9b-it | $0.06 / $0.06 |
| mistral-nemo-instruct-2407 | mistralai/Mistral-Nemo-Instruct-2407 | $0.08 / $0.12 |

*(prices are approximate public serverless list prices and must be re-verified
before any published number relies on them.)*

**Cross-source join (`src/router/data/benchmark_id.py`).** RouterBench and
lm-eval ask the *same underlying question* in different prompt formats, so the
E3 models are joined to the existing 11 on **question identity, not prompt
text**: `benchmark:{task}:{index}`. Valid only for RouterBench's standard-
benchmark subset -- **MMLU (57 subjects), HellaSwag, WinoGrande, ARC-Challenge**
≈ 27k of RouterBench's 36k 0-shot questions. RouterBench `sample_id`
`mmlu-anatomy.val.42` and lm-eval `mmlu_anatomy` doc 42 both map to
`benchmark:mmlu_anatomy:42`. GSM8K / MBPP are excluded from the default map:
RouterBench's counts (7450 / 427) don't match the lm-eval test-split sizes
(1319 / 500), so the index alignment is unverified. The custom RouterBench
tasks (Chinese riddles, abstract2title, consensus_summary, mtbench, ...) have no
lm-eval counterpart and stay on their `routerbench:...` id.

**Cost basis (`router.data.pricing`).** `cost = in_tok*price_in + out_tok*price_out`
filled at ingestion (`build_tables`) for every non-RouterBench source from the
dated snapshot. A model with no price stays `cost=NaN` and is dropped from the
*routing* eval (kept in *prediction*), recorded in the battery.

**Status: blocked on GPU budget authorisation.** All no-cost scaffolding is in
place -- `configs/model_prices.yaml`, `configs/model_registry.yaml` (5 high-
confidence aliases), `configs/phase0.yaml -> sources.lm_harness.models`,
`scripts/data/run_lm_harness.py` (emits the per-model `lm_eval` command + an
`hf jobs run` wrapper), `router.data.{pricing,benchmark_id}`, and tests
(`test_pricing.py`, `test_benchmark_id.py`). Still to do once logs exist:
the RouterBench↔lm-eval `query_id` remap in the loader (log-format-specific),
token extraction from the lm-eval sample logs, an "aligned" NIRT-observation
build keyed on `benchmark:*`, retrain, battery + per-model in/out ablation.
