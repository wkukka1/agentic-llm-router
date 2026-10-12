# MOD-001: `fill_costs` / `cost_for` price rows with unknown token counts at $0, not NaN

- **Status:** reproduced (`known_bugs/_repros/repro_fill_costs_zero.py`), latent on the current build
- **Severity:** low-med (latent)
- **Where:** `fill_costs` ([pricing.py:100-112](../../../src/training/data/pricing.py#L100-L112)) and `cost_for` ([pricing.py:71-73](../../../src/training/data/pricing.py#L71-L73))
- **Found:** 2026-09-24

## What

The docstring says rows whose model has "no price **or no token counts** are
left NaN" ([pricing.py:80-82](../../../src/training/data/pricing.py#L80-L82)), and the
schema forbids fabricating zeros ([schemas.py:10-13](../../../src/training/data/schemas.py#L10-L13)).
The code does the opposite for tokens: `it`/`ot` are built with
`.fillna(0.0)`, so a priced model with both counts missing gets
`0 * rate + 0 * rate = 0.0`, a real-looking $0.00. A row with only one count
known is priced on that side alone (the missing side contributes 0), silently
underestimating.

```
model 'phi-3.5-mini-instruct' (in=0.03/1M out=0.05/1M)
   input_tokens  output_tokens      cost
0           NaN            NaN  0.000000   <- should be NaN
1         100.0            NaN  0.000003   <- output side treated as 0
2           NaN           50.0  0.000003
```

## Impact today

None on the current tables: the 341,156 non-RouterBench rows with no tokens are
Arena / judge rows whose models are unpriced, so they stay NaN. It becomes live
the moment a priced model appears in a source that reports no tokens (e.g. an
lm-eval-harness log for a priced model, which `load_lm_harness` writes with
`input_tokens: None`). Same class as AGENTICROUTER-002 / LLMCLIENT-001 (unknown
cost reported as zero).

## Suggested fix

Leave `cost` NaN unless the needed token count(s) are present: compute
`it * in_rate + ot * out_rate` with NaN propagating (no `fillna(0)`), and make
`cost_for` return `nan` when either count is `None`/NaN.
