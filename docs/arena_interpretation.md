# Pairwise preference sources → response representation

Covers **Chatbot Arena** (`chatbot_arena`, human votes) and **GPT-4 Judge
battles** (`gpt4_judge`, `routellm/gpt4_judge_battles`, LLM-judged). Both use the
identical battle schema and identical normalization; they differ only in
`source` and `metric_type` (`arena_preference` vs `judge_preference`).

## Two separate heads

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

## The problem

Arena data is **pairwise preference**: for a prompt `q`, models `A` and `B`
produce responses and a human picks a winner (or a tie). RouterBench /
lm-harness data is **absolute**: model `M` on item `q` gets an F1 / EM / pass@1 /
accuracy score that does not depend on any other model.

An Arena vote is **not** equivalent to an F1 or pass@1 score. Pretending it is
would corrupt the Phase 1 IRT response matrix.

## What Phase 0 does

For each battle `(q, A, B, winner)` we emit **two** observations:

| query_id | model_id | metric_type       | score | metadata.opponent_model_id | metadata.role |
|----------|----------|-------------------|-------|----------------------------|---------------|
| `q`      | `A`      | `arena_preference`| 1.0 / 0.5 / 0.0 | `B` | `a` |
| `q`      | `B`      | `arena_preference`| 0.0 / 0.5 / 1.0 | `A` | `b` |

* `score = 1.0` win, `0.5` tie, `0.0` loss — **for that model against that
  specific opponent on that prompt**.
* `metric_type = "arena_preference"` keeps these rows separable from every
  absolute metric. They are never pooled onto a common scale with accuracy /
  F1 / pass@1.
* `metadata.pair_id` uniquely identifies the battle; `metadata.opponent_model_id`
  and `metadata.role` record the comparison context.

## Statistical interpretation (for Phase 1, not implemented here)

`score` is an observation of the pairwise outcome random variable

```
Y_{q,A,B} ∈ {0, 0.5, 1},   E[Y_{q,A,B}] = P(A ⪰ B | q)
```

This is a **relative** quantity. A model's Arena score on a query depends on
which opponent it faced, so these rows do **not** slot into `R[q, m]` as an
item-level absolute the way RouterBench scores do. Phase 1 has a few defensible
options (all out of scope for Phase 0):

1. **Bradley–Terry / Thurstonian layer.** Treat `arena_preference` rows with a
   paired-comparison likelihood: `P(A beats B | q) = σ(f(θ_A, r_q) − f(θ_B, r_q))`.
   The IRT ability/relevance parameters feed the comparison; no absolute score
   is fabricated.
2. **Per-query win-rate aggregation.** For `(q, M)` collapse all battles into a
   win rate `w_{q,M} = mean(score)` with an effective sample size. This yields
   an absolute-looking number but it is conditional on the (non-random) set of
   opponents `M` happened to face, so it must carry that caveat.
3. **Drop Arena from the response matrix** and use it only for held-out
   preference-agreement evaluation.

## Ambiguities flagged for human review

* **Opponent confounding.** Arena matchmaking is not random; strong models meet
  strong models. Any aggregation to an absolute per-`(q, M)` score inherits this
  bias. Phase 0 does not aggregate — it keeps every battle row with its
  opponent — so the decision is deferred to Phase 1.
* **Multiple battles per (q, M).** The same prompt can be voted on many times
  with different opponents. These are legitimately multiple observations of the
  same `(query_id, model_id, {arena,judge}_preference)` cell; the canonical
  "one row per (query_id, model_id, metric_type)" rule therefore has a
  **documented exception for the pairwise metrics** (uniqueness there keys on
  `pair_id` + `role`). The quality check `duplicate_pairwise_battle` enforces
  the exception explicitly.
* **Ties.** Encoded as `0.5`. A model that "wins" a Phase 1 BT layer treats
  `0.5` as a half-count; a model that aggregates to win-rate does the same. No
  information is discarded.
* **`winner_tie` vs "both bad".** The 55k release has a single `winner_tie`
  flag; some Arena releases distinguish "tie" from "both bad". We collapse to
  `0.5` and note it here.
