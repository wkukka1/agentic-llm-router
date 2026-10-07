# MOD-001: IRT-Router rows with 0 tokens and $0 cost (but a real score) are ingested as "free"

- **Status:** fixed 2026-09-25, see [fix.md](fix.md)
- **Severity:** med
- **Where:** `_melt_irt_router` ([loaders.py:543-548](../../../../src/training/data/loaders.py#L543-L548)); rule broken: [schemas.py:10-13](../../../../src/training/data/schemas.py#L10-L13)
- **Found:** 2026-09-24

## What

The schema's first design rule is that missing `input_tokens` / `output_tokens`
/ `cost` are left null, **never** filled with 0: "0 tokens / 0 cost is a
different, wrong claim". `_melt_irt_router` takes the CSV's `input_tokens`,
`output_tokens` and `cost` verbatim (`pd.to_numeric`), so any 0/0/0 record in
the source survives as a genuine $0.00 observation.

## Evidence

`data/processed/irt_router/responses.parquet` (760,620 rows) has **10,565** rows
with `cost == 0` and both token counts 0:

| model | zero/zero rows | of its rows | mean score on those rows |
|---|---|---|---|
| `qwen25_72b_instruct` | 7,953 | 38,031 (20.9%) | 0.800 |
| `llama31_70b_instruct` | 2,581 | 38,031 (6.8%) | 0.542 |
| `llama31_405b_instruct` | 29 | | 0.0 |
| `deepseek_chat`, `deepseek_coder` | 1 each | | 0.0 |

The scores are not failures (0.80 mean for `qwen25_72b_instruct`, about its
overall 0.800), so these are answered queries whose usage accounting is
missing, not calls that never happened. `qwen25_72b_instruct`'s mean cost is
0.000236 including them vs 0.000298 without, a 21% understatement.

(`glm_4_flash` also shows `cost == 0` on all 38,031 rows, but with real token
counts (71 to 2,788 in); that is a free model, not this bug.)

## Impact

Everything downstream reads `cost` as ground truth: `eval_matrices` /
`dense_matrices`, `route()`'s per-model mean cost, `oracle_choice`, and the
cost-savings columns. Zero-cost cells make the model look cheaper than it is
and, in `oracle_choice`, win the "cheapest model attaining the max score"
tie-break, which would inflate the oracle's cost-saving ceiling. (The size of
that effect was not measured.) The quality report does not catch it:
`impossible_*` only flags negatives.
