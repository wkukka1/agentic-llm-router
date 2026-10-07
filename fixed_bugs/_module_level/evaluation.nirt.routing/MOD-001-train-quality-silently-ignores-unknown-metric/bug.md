# MOD-001: `train_quality`'s `metric` argument silently falls back on anything but `"accuracy"`

- **Status:** fixed 2026-09-25, see [fix.md](fix.md)
- **Severity:** medium
- **Where:** `train_quality` ([evaluation/nirt/routing.py:114-124](../../../../src/evaluation/nirt/routing.py#L114-L124))
- **Found:** 2026-09-22

## What

```python
def train_quality(data, models: list[str], metric: str = "quality") -> dict[str, float]:
    ...
    if metric == "accuracy":
        y = (y >= 0.5).astype(np.float64)
    return pd.Series(y).groupby(obs["model_id"].to_numpy()).mean().to_dict()
```

Only the literal string `"accuracy"` is special-cased. Any other value —
including a typo of the intended `"quality"` (e.g. `"Quality"`,
`"qualtiy"`), or any string that isn't one of the two documented options —
silently takes the `"quality"` branch (mean of the raw graded target) with
no `ValueError` and no warning.

## Failure scenario

A caller who passes `metric="Accuracy"` (capitalization slip) or any other
near-miss gets back per-model mean *quality*, not accuracy, with nothing
indicating the metric name wasn't recognized — a baseline comparison that
silently compares the wrong quantity.
