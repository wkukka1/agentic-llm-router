# Candidate-Pool Expansion — phased plan + results ledger

You are continuing the LLM-router research implementation.

## Context

Phases 0, 1, 2 are complete. The Phase 2 winner is the **ZOIB** response head
(`artifacts/phase2/zoib`), on the **same frozen representation** as Phase 1.

The routing analysis (`scripts/route_compare.py`,
`artifacts/phase2/routing_comparison.json`) showed the router's cost savings are
**capped by the candidate pool, not the predictor**:

* pool = 9 "warm" RouterBench models, GPT-4-dominated (GPT-4 acc 0.84; next best
  0.69; the two models priced ≤ 1/30 of GPT-4 have acc 0.29 / 0.45).
* **oracle** (perfect router) cost saving vs always-GPT-4 = **−76 %**, so
  −97 % ("1/30") is impossible on this pool regardless of router quality.
* ZOIB router: −5 % at λ=0, **−57 % at −3.7 accuracy points** (λ=0.5).
* 2 of the 11 RouterBench models are currently **held out** as cold-start —
  including `yi-34b-chat` (acc 0.68 at 1/18 the price of GPT-4), exactly the
  "cheap-and-good" model the pool lacks.

**Objective of this workstream:** expand the candidate pool in isolated phases,
re-run the *identical* evaluation battery each time, and record how the pool's
routing ceiling and the ZOIB router's achievable savings move as models are
added. This is orthogonal to the Phase 3 (correctness/length coupling) track.

---

# 0. Discipline (applies to every phase)

1. **One variable per phase.** Add exactly the models / data that the phase
   specifies. Do not also change the model config, `theta_dim`, encoder,
   regularisation, split methodology, or `configs/phase2.yaml` training block.
2. **Same model, same seed.** Train the Phase 1 Bernoulli baseline and the
   Phase 2 ZOIB head with the *unchanged* `configs/phase1.yaml` /
   `configs/phase2.yaml` and `seed: 42`. The point is to isolate the pool
   effect, not to retune.
3. **Identical eval battery** (§1) re-run verbatim each phase.
4. **Ablation within the phase.** Also evaluate the pool *without* the newly
   added model(s) so the marginal contribution of each addition is attributable.
5. **No leakage on new data.** New models get a warm/cold split assignment by
   the existing deterministic hash. New *sources* get their queries assigned to
   train/val/test by the existing content-hash group split — verify
   `router.data.splits.check_leakage` is clean and that no new-source query
   collides with an existing test query.
6. **Cost comparability.** RouterBench carries a precomputed per-(model, query)
   USD cost. Any model added another way (lm-harness, external source) has **no
   cost** — you must supply a documented price basis (§4) or that model is
   excluded from the routing (not the prediction) eval and this is recorded.
7. **Reproducibility.** Every phase writes a provenance block (git sha, phase-0
   artefact hashes, seed, pool composition, config) and a second-seed sanity
   check on the ZOIB run.
8. **Record before moving on.** Append the phase's row to the ledger (§2) and
   fill the phase's results section (§10) before starting the next phase.

---

# 1. The fixed evaluation battery

Run **all** of this, identically, at the end of every phase, on the **test**
split, and write it under `artifacts/pool_expansion/<phase>/`:

### 1a. Pool description
Table: `model_id | source(s) | n_correctness_obs | test accuracy | mean $/1k |
warm/cold`. Plus: number of warm models, GPT-4:cheapest cost ratio, count of
models priced ≤ 1/30 of the most expensive and their mean accuracy.

### 1b. Prediction quality (ZOIB, and Bernoulli control)
`scripts/evaluate_continuous.py --checkpoint <zoib> --split test` and
`scripts/evaluate_baseline.py --checkpoint <bernoulli> --split test`:
NLL, MAE, RMSE, mean-calibration ECE, 50/90 % interval coverage, boundary
calibration (ZOIB `P(y=0)`/`P(y=1)` vs empirical), θ effective rank, per-family
breakdown, pathology flags.

### 1c. Routing (the headline)
`scripts/route_compare.py --split test` — the policy table
(oracle / always-best-model / cheapest / NIRT / baseline-NIRT / ZOIB E[Y] /
ZOIB LCB) with accuracy, graded quality, $/1k, `cost_savings_vs_ref`,
`d_accuracy_vs_ref`; the ZOIB λ-sweep frontier; and **AIQ** (absolute +
improvement-over-linear).

### 1d. Derived headline numbers (compute + record explicitly)
* **oracle cost-saving ceiling** = `1 - oracle_cost / always_best_model_cost`.
* **router saving at matched accuracy** = the largest `cost_savings_vs_ref` on
  the ZOIB frontier whose accuracy ≥ (best-single-model accuracy − 1.0 pt), and
  the same at −3.0 pt. If no frontier point qualifies, record "none".
* **fraction of test queries the oracle routes off the best model**, and the
  oracle's model mix.
* **ZOIB router escalation rate** (share NOT sent to the best model) at λ=0 and
  at the λ that gives the −3 pt operating point.

### 1e. Cold-start status
If a cold-start arm exists this phase: `model_params="projected"` ZOIB run,
`cold_start` block (accuracy + BCE-diag vs global-mean baseline), and projected
θ effective rank. Record whether projected cold-start now beats the
global-mean baseline on BCE (Phases 1–2: it did not, with 9 warm models).

### 1f. Ablation
Re-run 1c with the newly added model(s) removed from the routing matrices
(prediction model unchanged). Record the delta in oracle ceiling and in the
ZOIB −3 pt saving. This attributes the phase's movement to specific models.

---

# 2. The results ledger

Maintain `artifacts/pool_expansion/ledger.json` (and a rendered
`docs/pool_expansion_results.md` table). **One row per phase**, columns:

```
phase
n_warm_models
sources
gpt4_cheapest_cost_ratio
zoib_test_nll
zoib_test_mae
zoib_mean_cal_ece
oracle_cost_saving_ceiling          <- moves as the pool improves
zoib_saving_at_minus1pt              <- headline
zoib_saving_at_minus3pt              <- headline
zoib_aiq_improvement
oracle_offbest_fraction
coldstart_beats_global_mean         <- bool / n/a
notes
```

The trajectory of `oracle_cost_saving_ceiling` and `zoib_saving_at_minus3pt`
across rows **is the deliverable** — it answers "how close can this pool get to
the 1/30 claim, and which additions moved the needle".

---

# 3. Phases

## Phase E0 — lock the baseline (no new models)

* Freeze the current 9-warm-model pool as the reference row.
* Run the full battery (§1) against the existing
  `artifacts/phase1/baseline` + `artifacts/phase2/zoib` checkpoints — no
  retraining. This produces the row-0 ledger entry and the exact JSON shapes
  every later phase must match.
* Acceptance: battery runs end-to-end; ledger has one row; numbers match the
  Phase 2 report (test NLL 0.388, oracle ceiling −76 %, ZOIB −57 % at −3.7 pt).

## Phase E1 — un-hold the RouterBench models (Tier 0, free)

* The two held-out RouterBench models (`yi-34b-chat`,
  `code-llama-34b-instruct`) already have full data. Bring them warm.
* Keep a cold-start arm: add `split.force_warm` / `split.force_cold` lists to
  `configs/phase0.yaml` and `router.data.splits.make_splits` (forced ids skip
  the cold lottery). Warm both RouterBench models; if a cold arm is still
  wanted, hold out **Arena-only** models or a documented subset instead. If no
  clean cold arm remains, set `cold_start_model_fraction: 0.0` and record that
  the cold-start eval is paused this phase.
* Rebuild: `build_splits.py` → `build_nirt_dataset.py` → (optional)
  `-m router.phase0 --taxonomy --query-bank` → retrain Bernoulli + ZOIB
  (unchanged configs) → full battery + ablation (pool with vs without yi-34b).
* Expected: `yi-34b-chat` (acc 0.68 @ $0.19/1k) enters the "cheap-and-good"
  quadrant; oracle ceiling and the ZOIB −3 pt saving should both improve.
  Attribute the movement to yi-34b vs code-llama-34b via the ablation.

## Phase E2 — RouterBench 5-shot (Tier 1, ~1 h compute)

* `configs/phase0.yaml -> sources.routerbench.shots: ["0shot", "5shot"]`.
* Decide + document the item-identity policy: 0-shot and 5-shot of the same
  prompt are either (a) **separate items** (recommended — different `query_id`
  suffix, both get an independent split assignment by content hash of the full
  prompt) or (b) **pooled** (mean score per prompt). Do NOT let a 0-shot and
  5-shot version of one prompt straddle train/test.
* Same 11 models, ~2× observations. Few-shot lifts the weak models more —
  record the per-model test-accuracy shift and whether the useful routing band
  widened.
* Full pipeline rebuild → retrain → battery. Ablation: 0-shot-only vs +5-shot.

## Phase E3 — modern cheap-competent models via lm-eval-harness (Tier 2)

* Define the model set in `configs/phase0.yaml` (new `lm_harness.models` list).
  Target the empty quadrant: e.g. `Llama-3.1-8B-Instruct`,
  `Qwen2.5-7B-Instruct`, `Qwen2.5-14B-Instruct`, `Gemma-2-9B-it`,
  `Mistral-Nemo-Instruct`, `gpt-4o-mini`, `claude-3-5-haiku`,
  `gemini-1.5-flash`. Pick the exact set and pin versions.
* `scripts/data/run_lm_harness.py --print-only` emits the `lm_eval` command +
  task list matched to the RouterBench eval set. Run it (`--log_samples`),
  local GPU for open models / OpenAI-compatible endpoint for hosted ones.
* Drop `*samples*.jsonl` per-sample logs into `data/raw/lm_harness/`.
* **Cost basis (required):** create `configs/model_prices.yaml`
  (`<model>: {input_per_1m, output_per_1m}`), add a pipeline step that fills
  `cost = input_tokens*price_in + output_tokens*price_out` from the response
  logs' token counts. Document the price snapshot date and source. Models
  without a price are kept in the *prediction* eval but excluded from the
  *routing* eval (recorded).
* `build_response_matrix.py` ingests them (canonical schema + chance-correction
  + model registry merge are automatic). New models get a warm/cold assignment
  by the hash; force-warm the ones you want in the routing pool.
* Retrain → battery → ablation (each new model in / out of the routing matrix).
* This is the phase expected to move the oracle ceiling substantially toward
  −90 %+ if the chosen models are genuinely cheap-and-good.

## Phase E4 — external benchmark source (Tier 3)

* Add one more router benchmark (candidates: the `evaluations/LLMRouterBench`
  submodule's pooled results; RouteLLM data; an HF `router-bench` / MixInstruct
  dataset). Write a loader in `src/router/data/loaders.py` conforming to the
  canonical `(query_id, model_id, query, score, metric_type, source, cost?)`
  schema; register model identities in `configs/model_registry.yaml`
  (confidence-based; only `high`-confidence aliases merge).
* Verify: new-source queries get split-assigned; no new-source query's content
  hash collides with an existing test query (else it leaks); metric types are
  kept distinct (never rescaled onto one axis).
* Retrain → battery. Record which new models/tasks this brought and the pool /
  ceiling movement.

## Phase E5 — profile-only cold-start models (Tier 4)

* Only attempt once `n_warm_models >= ~20` (Phases 1–2 showed projected-mode
  cold-start fails with 9). Re-run the projected ZOIB cold-start eval (§1e); if
  it now beats the global-mean baseline on BCE, the profile encoder is viable.
* Then add 2–3 models with **only a text profile** (extend
  `configs/model_profiles.yaml`, run `scripts/models/build_model_profiles.py` +
  the `irt` profile-embedding build) and **no benchmark data at all**. Score
  them with `theta_m = W_theta @ e_m`; place them in the routing matrix using
  the predicted E[Y] and a documented price.
* Record: do the profile-only models land in a sensible place on the
  cost/quality frontier? Does adding them (predicted-only) still improve the
  routing frontier, or does prediction error swamp the benefit?

---

# 4. Cost normalisation (read before E3+)

* RouterBench `total_cost` is USD per (model, query), already token-weighted.
* lm-harness / external models: `cost = Σ tokens · price`. Use
  `configs/model_prices.yaml`, snapshot the prices with a date, prefer the
  provider the model would realistically be served from (cheapest reputable).
* The routing report's `lam` uses `cost / max_cell_cost` so λ is comparable
  across pools — but the **absolute** `$/1k` and `cost_savings_vs_ref` numbers
  are only comparable across phases if the price basis is stable. If you
  re-price, re-run every prior phase's routing eval or clearly flag the break.
* Report `cost_savings_vs_ref` against **the best single model of the current
  pool** (usually still GPT-4-1106) so the "vs always use the best" framing is
  constant.

---

# 5. Tests

Add / extend:

* `tests/test_splits.py` — `force_warm` / `force_cold` honoured; forced ids
  never appear in the cold lottery; leakage still clean.
* `tests/test_data.py` — 5-shot ingestion (item-identity policy), lm-harness
  sample-log ingestion, `model_prices.yaml` cost fill (a model with a known
  token count gets the expected cost), new-source loader conforms to schema.
* `tests/test_pool_expansion.py` (new) — the battery-runner produces the ledger
  row shape; ablation removes exactly the named models from the routing
  matrices and nothing else; derived headline numbers (oracle ceiling, saving
  at matched accuracy) computed correctly on a toy pool.

`pytest -q` green at every phase.

---

# 6. Acceptance criteria (per phase)

* `pytest -q` passes.
* The full battery (§1) ran and its artefacts exist under
  `artifacts/pool_expansion/<phase>/`.
* The ledger has the new row; `docs/pool_expansion_results.md` re-rendered.
* The ablation (§1f) attributes the phase's ceiling/saving movement to specific
  added models.
* ZOIB second-seed run reproduces validation NLL within run-to-run tolerance.
* Phase 1 Bernoulli control still reproduces on the *same* pool (bit-identical
  if the pool is unchanged; approximately if the pool grew).
* Provenance block written.

---

# 7. Final report (after the last phase you run)

```
POOL EXPANSION COMPLETE (through Phase E<n>)

Ledger (one line per phase):
  E0  warm= 9  ...  oracle_ceiling -76%  zoib@-3pt -57%
  E1  warm=11  ...  oracle_ceiling ____  zoib@-3pt ____
  ...

Trajectory:
  oracle cost-saving ceiling:   -76%  ->  ...
  ZOIB router saving @ -1pt:     none  ->  ...
  ZOIB router saving @ -3pt:     -57%  ->  ...
  ZOIB AIQ improvement:          +0.21 ->  ...

Which additions moved the needle (from the ablations):
  yi-34b-chat:            oracle ceiling ____, zoib@-3pt ____
  <model>:               ...

Prediction quality vs pool size:
  ZOIB NLL / MAE / mean-ECE across phases
  cold-start (projected) beats global-mean baseline?  first true at warm=____

Cost basis:
  RouterBench: precomputed USD.  Custom models: configs/model_prices.yaml (snapshot ____).

Closest approach to the "1/30" (-97%) claim:
  best pool = Phase E<n>, oracle ceiling ____%, achievable router saving at
  (best-model acc -1pt) = ____%.  Gap to -97% attributable to: ____.

Files created / modified:
Known limitations:
Next-phase readiness: YES / NO
```

---

# 8. Do NOT

* Retune the ZOIB / Bernoulli config, `theta_dim`, or encoder to chase a
  routing number. The pool is the variable.
* Merge model identities across sources without registry review.
* Let 5-shot / 0-shot or external-source duplicates straddle the split.
* Report a routing saving without also reporting the accuracy delta and the
  oracle ceiling for that pool.
* Start the Phase 3 (correctness/length coupling) track from here — that is a
  separate workstream on the frozen ZOIB head.
