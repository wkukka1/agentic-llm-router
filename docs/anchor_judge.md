# Anchor-topology judge: query-matched gold/preference overlap set

Spawned by a confound found in the capacity workstream (`docs/nirt_capacity.md`,
Item 2): the `train.preference.weight` sweep concluded "Arena/Judge preference
and RouterBench correctness are different notions of quality" from a
monotonically-harmful-with-no-sweet-spot regret curve. But the correctness
query set (`routerbench:...`, 72,966 ids) and the preference query set
(`chatbot_arena:...`/`gpt4_judge:...`, 160,308 ids) have **zero query_id
overlap** (verified directly), and the sweep's regret is computed **only** on
the correctness-side validation split. So that curve is equally explained by
plain domain shift (160k out-of-distribution training queries hurting
in-distribution eval) as by incompatible utilities — the experiment as run
cannot distinguish these. `docs/nirt_capacity.md`'s Item 2 now carries an
inline correction to this effect.

This pipeline manufactures the missing overlap: judge RouterBench's own model
answers against each other, on RouterBench's own query_ids, so gold
(correctness) and preference (judge) labels exist on **identical** queries.
RouterBench's raw pickles already contain every model's actual generated
answer text (`routerbench/routerbench_0shot.pkl`'s `<model>|model_response`
columns — dropped at ingestion, but present) — no new model generations
needed.

## Pipeline

1. **`scripts/data/select_anchor_model.py`** — picks the anchor: the
   **median**-quality model of the RouterBench-11 pool by mean training
   accuracy (`router.nirt.routing.train_quality`). Anchoring on the strongest
   model floors most judged gains; the median is computed, not guessed.
   Writes `data/processed/anchor_judge/anchor_selection.json`.
2. **`scripts/data/select_anchor_judge_queries.py`** — samples 2,500 queries
   (default 1,750 "discriminative" + 750 "natural"; `src/router/data/stratify.py`)
   from the 0-shot-only correctness matrix. Discriminative = correctness
   varies across the pool (real routing signal); natural = an unweighted
   sample of everything else. Kept as two disjoint, separately-reported
   strata rather than one reweighted blend.
3. **`scripts/data/collect_anchor_judgments.py`** — Phase A judges every
   candidate's answer against the anchor's (`src/router/judge/`), Phase B
   re-judges a ~10% margin-stratified subsample with position swapped (the
   noise floor). Any real provider requires `--confirm-budget` plus explicit
   per-token prices; the script prints projected cost from the already-known
   response texts before placing a paid call.
4. **`scripts/nirt/anchor_judge_analysis.py`** — applies the decision rule
   below and prints a verdict.

## Judge contract

The rubric (`src/router/judge/prompts.py`) explicitly targets helpfulness /
completeness / clarity / formatting, explicitly **not** correctness, and the
judge is never shown ground truth. Output is a graded margin in `[-3, 3]`
(positive favors the candidate), not a win/loss/tie sign — a sign-only
outcome can't produce a quality-*gain* matrix and ties would dominate on
RouterBench's often near-identical short answers.

## Pre-registered decision rule

The primary estimand is **routing**, not pairwise judge accuracy: build the
`[Q, M]` judge gain matrix (anchor column = 0 by construction) and score it
against gold correctness via the existing `routing_evaluation` — regret
directly comparable to the 0.1181 baseline in `docs/nirt_capacity.md` — plus
the query-level decision-disagreement rate (`argmax(gain) != argmax(R)`).

**Two hard validity gates run first, before any fork is evaluated:**

- **Parse-quality gate**: unparseable judge output ≤ 5%.
- **Correctness-classifier void-check**: on the discriminative stratum's
  informative pairs (gold correctness differs between anchor and candidate),
  the judge must prefer the gold-*incorrect* response more than 10% of the
  time. A judge that never does this is acting as a correctness classifier,
  not a preference judge — `VOID`.

**Then an informativeness gate** (added after dry-run verification found a
real design gap — see below): the judge's routing must beat a
structure-matched random-routing baseline (same fixed anchor-column-at-0,
candidate columns drawn from the same -3..3 scale) by more than the swap
noise floor. Raw decision-disagreement/regret against a correctness oracle is
large for *any* uninformative signal, so without this gate a judge that
carries zero information about correctness looks identical to "genuine
conflict" — `INCONCLUSIVE` if it fails on every stratum (not a retraction,
not evidence of two utilities; the judge run itself needs fixing).

**Only once both hard gates and the informativeness gate pass:**

- **Excess regret/decision-disagreement ≈ 0** (indistinguishable from the
  swap noise floor) → routing by preference costs nothing beyond judge noise.
  **`DOMAIN SHIFT`**: retract (not just caveat) `docs/nirt_capacity.md` Item
  2's "different notions of quality" conclusion — the original sweep
  measured domain shift, not incompatible utilities. Close the
  mixture/regime-model research line.
- **Stable, material excess** → **`TWO UTILITIES`**: build the minimal
  per-channel-bias model — shared low-rank `A`, model embeddings `e_m`,
  per-channel bias `β_{m,channel}` (channel **observed** via the existing
  `source` column; no gate, no latent mixture, no Ψ matrix) — and test it
  against the pooled `train.preference.weight` baseline on this
  manufactured set only.

## Verification (real dry runs, 2026-09-04)

Ran the full pipeline against real RouterBench data with the `dummy` judge
provider (`--dry-run --dummy-mode ...`, no network, no cost) before any real
judge budget would be spent:

- `dummy-correctness-proxy` (always prefers the gold-more-correct side) →
  **`VOID: judge appears to act as a correctness classifier`**, as required
  — `prefers_incorrect_rate ≈ 0`.
- `dummy-random` (uniform random margin) → initially, **incorrectly**
  produced `TWO UTILITIES` — a real bug, not a false alarm to wave off: raw
  decision-disagreement against a correctness oracle is inherently large for
  *any* uninformative signal, and the swap-noise floor alone (a few hundred
  pairs) is far too small to explain that baseline. The fix was the
  informativeness gate above; a first version of it still false-"beat
  random" because its random baseline randomized *all* 11 columns while the
  judge matrix always fixes the anchor column at 0 (a structural asymmetry,
  not sampling noise — confirmed by it persisting from n=30 to n=400).
  Matching that structure fixed it: `dummy-random` now correctly reports
  **`INCONCLUSIVE`**, `prefers_incorrect_rate ≈ 0.50`, and its own regret is
  statistically indistinguishable from the structure-matched random baseline.

This means the decision rule has **three** outcomes in practice, not two:
`VOID` (bad judge), `INCONCLUSIVE` (no detectable signal either way), and
the pre-registered `DOMAIN SHIFT` / `TWO UTILITIES` fork. Only the last two
were specified going in; `INCONCLUSIVE` was a necessary addition once the
dry-run verification step (which exists specifically to catch this class of
mistake) surfaced the gap.

## Running it for real

```
python scripts/data/select_anchor_model.py
# copy anchor_selection.json's anchor_model_id into configs/phase0.yaml's
# anchor_judge.anchor_model_id (optional -- collect_anchor_judgments.py falls
# back to reading the file directly)
python scripts/data/select_anchor_judge_queries.py
python scripts/data/collect_anchor_judgments.py --dry-run                      # free, sanity check first
python scripts/data/collect_anchor_judgments.py --dry-run --dummy-mode random   # should be INCONCLUSIVE
python scripts/nirt/anchor_judge_analysis.py
# only after both dry-run modes behave as documented above:
python scripts/data/collect_anchor_judgments.py --confirm-budget \
    --provider openai --model <model> \
    --price-per-1k-input <price> --price-per-1k-output <price>
python scripts/nirt/anchor_judge_analysis.py
```

Config: `configs/phase0.yaml`'s `anchor_judge:` block. Env: `JUDGE_API_KEY`
(and optionally `JUDGE_BASE_URL`), see `.env.example`.
