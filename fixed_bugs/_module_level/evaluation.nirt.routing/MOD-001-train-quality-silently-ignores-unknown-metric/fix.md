# MOD-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-25, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`train_quality` raises `ValueError` for any metric other than `"quality"`/`"accuracy"`. Test: `test_train_quality_rejects_an_unrecognized_metric`.

## Fix suggested when the bug was filed

Validate `metric in {"quality", "accuracy"}` up front and raise `ValueError`
on anything else, matching how the docstring already documents exactly two
valid values.

## Regression tests

`tests/evaluation/nirt/test_nirt_routing.py:76` `test_train_quality_rejects_an_unrecognized_metric`

```python
def test_train_quality_rejects_an_unrecognized_metric():
    """MOD-001 (evaluation.nirt.routing): a metric that isn't "quality" or
    "accuracy" must raise, not silently fall back to the "quality" branch --
    a capitalization slip like "Accuracy" must not compare the wrong quantity
    with no signal that anything was wrong."""
    from evaluation.nirt.routing import train_quality

    d = FakeTrainingData(make_split_obs(seed=11))
    model_ids = sorted(d.observations["model_id"].unique())

    with pytest.raises(ValueError, match="metric"):
        train_quality(d, model_ids, metric="Accuracy")
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/evaluation/nirt/routing.py`

```diff
--- a/src/evaluation/nirt/routing.py
+++ b/src/evaluation/nirt/routing.py
@@ -61,11 +112,18 @@ def align(matrix: pd.DataFrame, like: pd.DataFrame) -> np.ndarray:
     return matrix.reindex(index=like.index, columns=like.columns).to_numpy(np.float64)
 
 
+_TRAIN_QUALITY_METRICS = frozenset({"quality", "accuracy"})
+
+
 def train_quality(data, models: list[str], metric: str = "quality") -> dict[str, float]:
     """Per-model mean training correctness -- for the 'best fixed model' baseline.
 
     ``metric="quality"`` -> mean graded target; ``"accuracy"`` -> mean(target>=0.5).
     """
+    if metric not in _TRAIN_QUALITY_METRICS:
+        raise ValueError(
+            f"train_quality: metric must be one of {sorted(_TRAIN_QUALITY_METRICS)}, got {metric!r}"
+        )
     obs = data.nirt_observations()
     obs = obs[(obs["split"] == "train") & (obs["model_id"].isin(models))]
     y = obs["target"].to_numpy(np.float64)
```
