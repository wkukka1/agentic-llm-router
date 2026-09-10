# Data sources — correctness, pairwise preference, model identity

How Phase 0 ingests the three raw benchmark sources into the canonical
`(query_id, model_id, score, metric_type, source, ...)` schema: multiple-choice
chance correction, pairwise-preference handling (Arena / GPT-4-Judge), and the
canonical model registry that keeps identities straight across sources.

---

## 1. Multiple-choice chance correction

### Constraint (confirmed 2026-08-27)

**No guessing parameter of any kind in Phase 1** — neither a *fitted* item
guessing term `c_q` (3PL) nor a *fixed* lower asymptote `c_q = 1/n_q`. The
multiple-choice guessing floor is handled entirely at the **data level**, here in
Phase 0, by rescaling the score. `n_choices` is still carried on every
observation for reference/diagnostics, but the Phase 1 response model is a plain
2PL/AMIRT with no lower asymptote.

A model that knows nothing still scores ~`1/n` on an `n`-option question; the
rescaling below removes that floor before Phase 1 sees the data.

### Approach: data-level rescaling (no new parameters)

Implemented in
[`src/router/data/chance_correction.py`](../src/router/data/chance_correction.py).
For a multiple-choice observation with `n` answer choices, chance level is
`c = 1/n`. The corrected score rescales `[c, 1]` onto `[0, 1]`:

```
corrected_score = clip( (score − c) / (1 − c),  0, 1 )
```

* Pure-chance performance → `0`; perfect performance → `1`.
* The classic "correction for guessing" / "adjusted accuracy" transform.
* Applied **per observation** using only that item's option count. Adds **no
  latent quantity** for Phase 1 to estimate — unlike 3PL's `c_q`.

`method: normalized` in `configs/phase0.yaml` selects this; `method: none`
disables it. `clip: true/false` controls the final clamp.

The Phase 1 facade (`router.data.phase1`) exposes `score_effective`
(= `corrected_score` where a correction was applied, else the original `score`)
as the default correctness signal, plus `score_raw` and `score_corrected` for
ablations across the binary / ordinal / continuous formulations.

### Columns

The original `score` is **never overwritten**. `apply_chance_correction` adds:

| column            | meaning |
|-------------------|---------|
| `score`           | untouched original |
| `corrected_score` | rescaled value, `NA` where no correction applied |
| `chance_level`    | `1/n`, `NA` where `n` unknown |
| `chance_corrected`| bool flag |

### When we do **not** correct

* Non-multiple-choice observations — untouched.
* `is_multiple_choice=True` but `n_choices` unknown — untouched, and a
  **warning** is emitted (`warn_on_missing_choices`). We never guess `n`.

### Which observations are multiple choice

Driven by `multiple_choice.by_task` / `multiple_choice.by_prefix` in the config
(exact `eval_name` or prefix → number of options). RouterBench tasks currently
mapped: `hellaswag` (4), `arc-challenge`/`arc-easy` (4), `winogrande` (2),
`mmlu*` (4), `mmlu-professional-law` (4), `truthful_qa` (4, MC1). Extend the
config as tasks are added — do not hard-code in source.

### Per-model bias (diagnostic only)

`estimate_model_bias` reports, per model, mean raw MC score, mean chance level,
and `mc_excess = mean_score − mean_chance` (crude skill-above-chance). Surfaced
for inspection and as an optional per-model offset for Phase 1; Phase 0 does not
bake it into any stored score. Example from the current data:

```
gpt-4-1106-preview      mc_excess ≈ 0.57
claude-v2               mc_excess ≈ 0.38
llama-2-70b-chat        mc_excess ≈ 0.01   (near chance)
code-llama-34b-instruct mc_excess ≈ -0.14  (below chance — format/refusal issues)
```

Negative `mc_excess` (e.g. code-llama) signals that the model's MC failures are
not random guessing but systematic (bad answer formatting, refusals). A per-model
bias term in Phase 1 can absorb this; an item-level `c_q` cannot.

---

## 2. Pairwise preference sources (Arena + GPT-4 Judge)

Covers **Chatbot Arena** (`chatbot_arena`, human votes) and **GPT-4 Judge
battles** (`gpt4_judge`, `routellm/gpt4_judge_battles`, LLM-judged). Both use the
identical battle schema and identical normalization; they differ only in `source`
and `metric_type` (`arena_preference` vs `judge_preference`).

### Two separate heads

Human preference and LLM-judge preference are **not assumed to be calibrated the
same way**, so they get distinct `metric_type`s and Phase 1 is expected to give
them separate heads / separate temperature (λ) weights:

```
L = L_correctness  +  λ_arena · L_arena_preference  +  λ_judge · L_judge_preference
```

The GPT-4 Judge data in this repo is entirely one matchup —
`gpt-4-1106-preview` (A) vs `mixtral-8x7b-instruct` (B), 109,101 battles — so it
informs the *relative* placement of those two models' abilities on each query,
nothing more. Arena spans ~60 models with non-random matchmaking.

### The problem

Arena data is **pairwise preference**: for a prompt `q`, models `A` and `B`
produce responses and a human picks a winner (or a tie). RouterBench / lm-harness
data is **absolute**: model `M` on item `q` gets an F1 / EM / pass@1 / accuracy
score that does not depend on any other model. An Arena vote is **not**
equivalent to an F1 or pass@1 score; pretending it is would corrupt the Phase 1
IRT response matrix.

### What Phase 0 does

For each battle `(q, A, B, winner)` we emit **two** observations:

| query_id | model_id | metric_type       | score | metadata.opponent_model_id | metadata.role |
|----------|----------|-------------------|-------|----------------------------|---------------|
| `q`      | `A`      | `arena_preference`| 1.0 / 0.5 / 0.0 | `B` | `a` |
| `q`      | `B`      | `arena_preference`| 0.0 / 0.5 / 1.0 | `A` | `b` |

* `score = 1.0` win, `0.5` tie, `0.0` loss — **for that model against that
  specific opponent on that prompt**.
* `metric_type` keeps these rows separable from every absolute metric. They are
  never pooled onto a common scale with accuracy / F1 / pass@1.
* `metadata.pair_id` uniquely identifies the battle; `metadata.opponent_model_id`
  and `metadata.role` record the comparison context.

### Statistical interpretation (for Phase 1, not implemented here)

`score` is an observation of the pairwise outcome random variable
`Y_{q,A,B} ∈ {0, 0.5, 1}` with `E[Y_{q,A,B}] = P(A ⪰ B | q)` — a **relative**
quantity. A model's Arena score on a query depends on which opponent it faced, so
these rows do **not** slot into `R[q, m]` as an item-level absolute. Phase 1 has
a few defensible options (all out of scope for Phase 0):

1. **Bradley–Terry / Thurstonian layer.**
   `P(A beats B | q) = σ(f(θ_A, r_q) − f(θ_B, r_q))` — IRT parameters feed the
   comparison; no absolute score is fabricated.
2. **Per-query win-rate aggregation.** Collapse `(q, M)` battles into a win rate
   `w_{q,M} = mean(score)` with an effective sample size — absolute-looking but
   conditional on the (non-random) opponent set, so it must carry that caveat.
3. **Drop Arena from the response matrix** and use it only for held-out
   preference-agreement evaluation.

### Ambiguities flagged for human review

* **Opponent confounding.** Arena matchmaking is not random; strong models meet
  strong models. Any aggregation to an absolute per-`(q, M)` score inherits this
  bias. Phase 0 does not aggregate — every battle row keeps its opponent — so the
  decision is deferred to Phase 1.
* **Multiple battles per (q, M).** The same prompt can be voted on many times
  with different opponents. These are legitimately multiple observations of the
  same `(query_id, model_id, {arena,judge}_preference)` cell; the canonical "one
  row per (query_id, model_id, metric_type)" rule therefore has a **documented
  exception for the pairwise metrics** (uniqueness there keys on `pair_id` +
  `role`). The quality check `duplicate_pairwise_battle` enforces the exception.
* **Ties.** Encoded as `0.5` (half-count in a BT layer or a win-rate). No
  information discarded.
* **`winner_tie` vs "both bad".** The 55k release has a single `winner_tie`
  flag; some releases distinguish "tie" from "both bad". We collapse to `0.5`.

The manufactured **query-matched** gold/preference overlap set (judging
RouterBench's own model answers on RouterBench's own query ids) is a separate
workstream — see [anchor_judge.md](anchor_judge.md).

---

## 3. Canonical model registry

File: [`configs/model_registry.yaml`](../configs/model_registry.yaml) ·
Loader: [`src/router/data/model_registry.py`](../src/router/data/model_registry.py)

### Why

Three sources spell models differently and at different granularities. The AMIRT
model must not think `gpt-4-0314`, `gpt-4-0613`, `gpt-4-1106-preview` are one
test-taker — nor should it split a single checkpoint that two sources both used.

### Merge policy

Each alias carries a `confidence`:

| confidence | behavior |
|------------|----------|
| `high`     | merged to `canonical_id` at load time |
| `medium` / `low` | recorded (`alias_confidence()`), **not** merged; the native string becomes its own `canonical_id` |

`high` is reserved for (a) exact provider-label identity, (b) unambiguous
provider-prefix rewrites (`meta/llama-2-70b-chat` → `llama-2-70b-chat`), and a
small number of cross-source merges where only one checkpoint ever existed.

### Cross-source decisions (needing periodic human review)

| models | decision | reason |
|--------|----------|--------|
| RB `mistralai/mixtral-8x7b-chat`, Arena/Judge `mixtral-8x7b-instruct-v0.1` | **merged** → `mixtral-8x7b-instruct` | only one Mixtral-8x7B-Instruct checkpoint (v0.1) ever existed |
| RB `claude-v2` vs Arena `claude-2.0`, `claude-2.1` | **kept separate** | RouterBench label is ambiguous between 2.0 and 2.1 |
| RB `claude-v1` vs Arena `claude-1` | kept separate | unconfirmed same snapshot |
| RB `claude-instant-v1` vs Arena `claude-instant-1` | kept separate | same |
| RB `code-llama-34b-instruct` vs Arena `codellama-34b-instruct` | kept separate (probably mergeable) | not yet reviewed |
| RB `mistral-7b-instruct` vs Arena `mistral-7b-instruct-v0.1` / `-v0.2` | kept separate | RouterBench doesn't say which checkpoint |
| RB `wizardlm-13b-v1.2` vs Arena `wizardlm-13b` | kept separate | Arena version unconfirmed |
| `gpt-4-1106-preview` in RouterBench and in GPT-4-Judge | **merged** (same id) | identical version string |

Every model observed in the data has a `canonical_id` (69 as of 2026-08-27); the
quality check `uncanonicalized_model` fails loudly if a new source introduces an
unmapped string.

### API

```python
from router.data.model_registry import (
    canonical_model_id, model_info, is_canonical, alias_confidence, all_canonical_ids
)
canonical_model_id("mistralai/mixtral-8x7b-chat")   # -> "mixtral-8x7b-instruct"
alias_confidence("claude-2.1")                      # -> "high"  (identity alias)
model_info("gpt-4-1106-preview")                    # ModelInfo(provider="openai", family="gpt-4", version="1106-preview", ...)
```
