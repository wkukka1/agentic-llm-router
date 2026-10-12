# MOD-001: `n_choices <= 1` yields NaN `corrected_score` while `chance_corrected` is True

- **Status:** reproduced (`known_bugs/_repros/repro_small_data_edges.py`, section b), latent
- **Severity:** low (latent)
- **Where:** `apply_chance_correction` ([chance_correction.py:86-108](../../../src/training/data/chance_correction.py#L86-L108)); same divide in `normalized_correction` ([chance_correction.py:64-69](../../../src/training/data/chance_correction.py#L64-L69))
- **Found:** 2026-09-24

## What

`correctable = is_mc & n_choices.notna()` is the only gate. The formula
`(s - c) / (1 - c)` with `c = 1/n` divides by zero at `n = 1` and is undefined
at `n = 0` (`c = inf`). Both give NaN, `np.clip(nan)` stays NaN, and the code
still sets `chance_corrected = True` and `chance_level` (`1.0` / `inf`):

```
corrected_score for n_choices=[1, 0, 4]: [nan, nan, 1.0]
chance_corrected:                        [True, True, True]
```

`TrainingData.correctness` then does `corrected.where(corrected.notna(), raw)`,
so the NaN silently falls back to the *raw* score while `chance_corrected` says
otherwise. `n_choices` comes from `multiple_choice.by_task` / `by_prefix` in
config ([loaders.py:45-55](../../../src/training/data/loaders.py#L45-L55)), so a typo
(`1`, `0`) or a one-option task would trigger it; `quality.check_responses`
does not flag it.

## Suggested fix

Require `n_choices >= 2` in `correctable` and treat smaller values as
`missing` (they are already warned about there), and/or validate
`multiple_choice.by_*` at config load.
