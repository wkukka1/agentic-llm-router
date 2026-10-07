# MOD-001: `target_ceiling` returns a reliability (variance share) but is used as a ceiling for Spearman rho

- **Status:** confirmed (analytic; simulation in `known_bugs/_repros/repro_length_ceiling_scale.py`). Depends on the two arena models being roughly parallel measures of one prompt-driven signal.
- **Severity:** low-med (an analysis error that changes a headline conclusion; no runtime failure)
- **Where:** `target_ceiling` ([length_audit.py:157-168](../../../src/evaluation/prompt_decomposition/length_audit.py#L157-L168)); the claim in [arena_corpus.py:20-21](../../../src/training/prompt_decomposition/data/arena_corpus.py#L20-L21), [length_audit.py:16-22](../../../src/evaluation/prompt_decomposition/length_audit.py#L16-L22), and docs (`prompt_signals.md` lines 66, 202, 254-278; `prompt_decomposition.md:133-134`; `prompt_decomposition_experiments.md:522-528`)
- **Found:** 2026-09-25

## What

`target_ceiling` applies Spearman-Brown to the two models' length agreement
(`r = 0.590`) and returns `2r/(1+r) = 0.742`. That number is a
**reliability**: the share of the target's *variance* that is signal. It is the
right bar for an R-squared, which is what `length_audit._score` uses. But the
docs and docstrings compare it to the shipped head's **Spearman rho** (0.573)
and conclude the head is at "~77% of the ceiling". A correlation is bounded by
the square root of the reliability:

| | docs' "ceiling" | correlation ceiling |
|---|---|---|
| predicting one model's length | 0.590 | sqrt(0.590) = 0.768 |
| predicting the two-model average (the shipped target) | 0.742 | sqrt(0.742) = 0.861 |

Simulating two noisy measures of one signal at the measured agreement, the best
possible prompt-only predictor reaches Spearman 0.753 against one model and
0.851 against the average (matching the square-root values), not 0.59 / 0.742.

## Impact

The shipped head (rho 0.573, 1024-d encoder) is at **~67%** of the achievable
correlation (0.573 / 0.861), not ~77%. On the variance scale the docs' own
R-squared for that head is 0.399, which is ~54% of 0.742; the correct
comparisons are 67% (rho) or 54% (R-squared), and "77%" matches neither because
it divides a correlation by a variance share. `docs/prompt_decomposition_experiments.md:522-528`
records having already corrected a first version of this comparison ("compared
0.573 to 0.590 and claimed 94%") but the fix stayed on the wrong scale, so the
remaining headroom is understated. That matters for the "is the length head
finished?" judgement.

Length-audit's own R-squared verdicts (`gap`, `curve_slope`, permutation) are
unaffected.

## Suggested fix

Either compare R-squared to 0.742, or compare rho to `sqrt(target_ceiling(...))`;
have `target_ceiling` say which scale it returns (or return both), and fix the
five doc passages. Worth re-reading with a caveat: the two models are not
exactly parallel (different variances / shared bias), so treat the square-root
values as an upper bound rather than a hard limit.
