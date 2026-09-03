# Multiple-choice chance correction

## Constraint (confirmed 2026-08-27)

**No guessing parameter of any kind in Phase 1** — neither a *fitted* item
guessing term `c_q` (3PL) nor a *fixed* lower asymptote `c_q = 1/n_q`
(fixed-asymptote 3PL). The multiple-choice guessing floor is handled entirely at
the **data level**, here in Phase 0, by rescaling the score. `n_choices` is
still carried on every observation for reference/diagnostics, but the Phase 1
response model is a plain 2PL/AMIRT with no lower asymptote.

A model that knows nothing still scores ~`1/n` on an `n`-option question; the
rescaling below removes that floor before Phase 1 sees the data.

## Approach: data-level rescaling (no new parameters)

Implemented in [`src/router/data/chance_correction.py`](../src/router/data/chance_correction.py).

For a multiple-choice observation with `n` answer choices, chance level is
`c = 1/n`. The corrected score rescales `[c, 1]` onto `[0, 1]`:

```
corrected_score = clip( (score − c) / (1 − c),  0, 1 )
```

* Pure-chance performance → `0`.
* Perfect performance → `1`.
* This is the classic "correction for guessing" / "adjusted accuracy" transform.
* It is applied **per observation** using only that item's option count. It adds
  **no latent quantity** for Phase 1 to estimate — unlike 3PL's `c_q`.

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

## Per-model bias (diagnostic only)

`estimate_model_bias` reports, per model, mean raw MC score, mean chance level,
and `mc_excess = mean_score − mean_chance` (crude skill-above-chance). This is
**surfaced for inspection** and as an optional per-model offset for Phase 1;
Phase 0 does not bake it into any stored score. Example from the current data:

```
gpt-4-1106-preview      mc_excess ≈ 0.57
claude-v2               mc_excess ≈ 0.38
llama-2-70b-chat        mc_excess ≈ 0.01   (near chance)
code-llama-34b-instruct mc_excess ≈ -0.14  (below chance — format/refusal issues)
```

Negative `mc_excess` (e.g. code-llama) is a signal that the model's MC failures
are not random guessing but systematic (bad answer formatting, refusals). A
per-model bias term in Phase 1 can absorb this; an item-level `c_q` cannot.
