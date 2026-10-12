# MOD-001: a unanimous *graded* query is put in the "discriminative" stratum

- **Status:** reproduced (`known_bugs/_repros/repro_small_data_edges.py`, section a)
- **Severity:** low
- **Where:** `sample_stratified_queries` ([stratify.py:51](../../../src/evaluation/data/stratify.py#L51))
- **Found:** 2026-09-24

## What

The module says the discriminative stratum holds queries where "correctness
varies across the pool", and that all-correct / all-wrong queries "carry zero
routing signal". The mask is `(row_mean > tol) & (row_mean < 1 - tol)`, which
is only equivalent to "varies" for binary 0/1 correctness. `R` here can be
graded (RouterBench `performance` in `{0, .25, .5, .75, 1}`, IRT-Router graded
scores), and a row where every model scores 0.5 has `row_mean = 0.5` and
`row_std = 0`: unanimous, but selected as discriminative. `row_std` is computed
one line earlier and reported in the output, but never used for the mask.

```
R rows: unanimous_half (0.5,0.5,0.5), all_right, all_wrong, mixed
discriminative stratum: ['mixed', 'unanimous_half']   # unanimous_half has row_std 0.0
```

## Impact

The anchor-judge overlap set spends judge calls (real API cost) on queries
where no pair can disagree, and the "discriminative rate" is diluted. Zero
impact on 0/1 matrices.

## Suggested fix

`discriminative_mask = row_std > tol` (with the mean-based test kept only if the
all-0 / all-1 exclusion is wanted separately; it is implied by `std > 0`).
