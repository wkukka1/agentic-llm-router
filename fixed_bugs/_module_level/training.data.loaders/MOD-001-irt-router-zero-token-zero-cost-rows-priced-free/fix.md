# MOD-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-25, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

Regression tests `tests/training/data/test_pricing.py::test_zero_usage_row_is_priced_at_the_rate_recovered_from_its_own_model` and `tests/training/data/test_irt_router.py::test_zero_usage_rows_are_priced_not_ingested_as_free`

Nulling the rows (the suggestion above) was not used: the zero rows are answered
queries at a hosted-API price (qwen2.5-72b, llama-3.1-70b are sold per token), so
they are priced instead.

`training.data.pricing.impute_missing_usage` (called from `load_irt_router`):

* **Rates:** recovered per model by least squares from that model's own priced
  rows (`cost = in * r_in + out * r_out`), which is the source's own pricing.
  Recovered exactly (median relative error 0): qwen25_72b_instruct $1.08/1M,
  llama31_70b_instruct $0.792/1M, llama31_405b_instruct $3.15/1M,
  deepseek_chat / deepseek_coder $0.14 in / $0.28 out per 1M.
* **Tokens (estimated):** input = median for the same query on other models, else
  the model's dataset mean; output = the model's dataset mean.
* **Fallback:** a model with fewer than 30 priced rows, or a non-positive rate,
  gets NaN tokens and cost (`metadata.usage_missing`), never 0.
* Imputed rows carry `metadata.usage_imputed = True`; real-usage rows are
  untouched, and a genuinely free model (real tokens, $0) is unaffected.

**Effect:** mean cost `qwen25_72b_instruct` 0.000236 -> 0.000299 (+26.7%),
`llama31_70b_instruct` 0.000233 -> 0.000248 (+6.6%); other models unchanged.

**Caveats:** the token counts are estimates (imputed vs real qwen rows: 207 / 72
vs 203 / 73 in / out). 31 of the rows (29 `llama31_405b_instruct`, 1 each
deepseek) have score 0 and may be failed calls that were never billed; they are
priced like the rest. The `quality.check_responses` `zero_cost_with_score`
warning suggested above was not added.

**Data:** `data/processed/irt_router/{responses,nirt_observations}.parquet` were
patched in place (10,565 rows, only `cost` / tokens / `metadata` changed; originals
kept as `*.pre-usage-fix`). Anything trained or evaluated on the old costs
(cost-aware oracle, `route_compare`, cost-savings tables, ZOIB cost/quality frontier
in `docs/irt_router.md`) needs re-running.

## Fix suggested when the bug was filed

In `_melt_irt_router`, set `cost`, `input_tokens` and `output_tokens` to NaN
where `input_tokens == 0 and output_tokens == 0 and cost == 0` (a real call
always has a non-zero prompt). Rows for those models then drop out of dense
routing matrices instead of poisoning them, or can be imputed from the price
table (`fill_costs` only touches `NaN` cost, so it would then apply). Add a
`zero_cost_with_score` warning to `quality.check_responses`.

## Regression tests

`tests/training/data/test_pricing.py:95` `test_zero_usage_row_is_priced_at_the_rate_recovered_from_its_own_model`

```python
def test_zero_usage_row_is_priced_at_the_rate_recovered_from_its_own_model():
    out = impute_missing_usage(_usage_frame())
    row = _zero_row(out)
    assert row["input_tokens"] > 0 and row["output_tokens"] > 0
    expected = (row["input_tokens"] * _IN_RATE + row["output_tokens"] * _OUT_RATE) / 1e6
    assert row["cost"] == pytest.approx(expected, rel=1e-6)
```

`tests/training/data/test_irt_router.py:121` `test_zero_usage_rows_are_priced_not_ingested_as_free`

```python
def test_zero_usage_rows_are_priced_not_ingested_as_free(tmp_path):
    """A 0-token / $0 record with a real score is missing accounting, not a free call."""
    rows = [f'{i},"Q {i}",A,,{100 + i},{20 + (i % 7)},{((100 + i) + 2 * (20 + (i % 7))) / 1e6},1,mmlu,gpt_4o'
            for i in range(60)]
    rows.append('900,"Q zero",A,,0,0,0,1,mmlu,gpt_4o')
    root = tmp_path / "raw" / "irt_router"
    (root / "data").mkdir(parents=True)
    for name in ("train", "test1", "test2"):
        (root / "data" / f"{name}.csv").write_text(_csv(rows if name == "train" else rows[:2]),
                                                   encoding="utf-8")
    cfg = Config({"seed": 42, "paths": {"processed": str(tmp_path / "p"), "splits": str(tmp_path / "s")},
                  "sources": {"irt_router": {"local_dir": str(root), "pool": ["gpt_4o"]}},
                  "multiple_choice": {"by_task": {"mmlu": 4}, "by_prefix": {}},
                  "chance_correction": {"method": "normalized", "clip": True},
                  "split": {"presplit": True, "train_fraction": 0.7, "validation_fraction": 0.3}},
                 root=tmp_path)

    df = load_irt_router(cfg)
    zero = df[df["query"] == "Q zero"].iloc[0]
    assert zero["cost"] > 0 and zero["input_tokens"] > 0
    assert zero["metadata"]["usage_imputed"] is True
    assert not ((df["cost"] == 0) & (df["input_tokens"] == 0)).any()
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/training/data/loaders.py`

```diff
--- a/src/training/data/loaders.py
+++ b/src/training/data/loaders.py
@@ -35,6 +35,7 @@ from router.config import Config, section
 from . import schemas
 from .model_registry import canonical_model_id
 from .normalize import make_query_id, render_prompt
+from .pricing import impute_missing_usage
 
 _SPLIT_TOKENS = {"dev", "test", "train", "val", "valid", "validation"}
 
@@ -474,7 +475,9 @@ def load_irt_router(cfg: Config) -> pd.DataFrame:
     (``id, question, ground_truth, completion, input_tokens, output_tokens,
     cost, performance, task, llm``) from ``sources.irt_router.local_dir/data``.
     ``performance`` is a graded [0, 1] correctness score; ``cost`` is the
-    token-weighted USD cost -- both used verbatim (no price table needed).
+    token-weighted USD cost -- both used verbatim (no price table needed),
+    except rows with 0 tokens and $0 cost, which are missing usage accounting
+    and are priced by :func:`~training.data.pricing.impute_missing_usage`.
 
     Query identity is the exact ``question`` text (matches the repo's own
     ``query_id_map[question]`` join). The origin file (train / test1 / test2) is
@@ -499,6 +502,8 @@ def load_irt_router(cfg: Config) -> pd.DataFrame:
                       if (n := _mc_choices(t, cfg)) is not None}
         frames.append(_melt_irt_router(df, origin, pool, mc_by_task))
     out = pd.concat(frames, ignore_index=True)
+    # 0 tokens + $0 on a scored row is missing usage accounting, not a free call.
+    out = impute_missing_usage(out)
     return schemas.coerce_response_frame(out)
 
 
```

`src/training/data/pricing.py`

```diff
--- a/src/training/data/pricing.py
+++ b/src/training/data/pricing.py
@@ -115,3 +115,68 @@ def fill_costs(df: pd.DataFrame, cfg: Optional[Config] = None, *,
 
 def priced_model_ids(cfg: Optional[Config] = None, *, path: Optional[str] = None) -> set[str]:
     return {k for k in load_prices(cfg, path=path)}
+
+
+def impute_missing_usage(df: pd.DataFrame, *, min_priced_rows: int = 30) -> pd.DataFrame:
+    """Price rows whose usage accounting is missing: 0 input tokens, 0 output
+    tokens and ``cost == 0`` on a row that carries a real score.
+
+    A prompt cannot have zero input tokens, so these are answered calls whose
+    usage was not recorded, not free ones. Ingesting them as $0 understates the
+    model's cost and lets it win cost tie-breaks (KNOWN BUG: training.data.loaders
+    MOD-001). A genuinely free model is unaffected: its rows carry real token
+    counts, so they never match.
+
+    For each affected model the per-token rates are recovered by least squares
+    from that model's own priced rows (``cost = in * r_in + out * r_out``) --
+    the source's own pricing, not an external table. Tokens are then estimated:
+
+    * input: the median input count for the same query on other models, else
+      the model's mean for that dataset;
+    * output: the model's mean for that dataset (else its overall mean).
+
+    A model with fewer than ``min_priced_rows`` priced rows, or a non-positive
+    fitted rate, has no trustworthy price: its affected rows get NaN tokens and
+    NaN cost (the schema's "unknown", never 0) and ``metadata["usage_missing"]``.
+    Every imputed row is marked ``metadata["usage_imputed"] = True``; rows with
+    real usage are not touched.
+    """
+    out = df.copy()
+    # float, not the schema's nullable Int64, so the masks below never hold <NA>
+    tok_in = pd.to_numeric(out["input_tokens"], errors="coerce").astype(float)
+    tok_out = pd.to_numeric(out["output_tokens"], errors="coerce").astype(float)
+    cost = pd.to_numeric(out["cost"], errors="coerce").astype(float)
+    missing = (tok_in == 0) & (tok_out == 0) & (cost == 0)
+    if not missing.any():
+        return out
+
+    priced = (tok_in > 0) & (cost > 0)
+    for model in out.loc[missing, "model_id"].unique():
+        rows = missing & (out["model_id"] == model)
+        own = priced & (out["model_id"] == model)
+        rate = None
+        if own.sum() >= min_priced_rows:
+            coef, *_ = np.linalg.lstsq(
+                np.column_stack([tok_in[own], tok_out[own]]).astype(float),
+                cost[own].to_numpy(float), rcond=None)
+            if (coef > 0).all():
+                rate = coef
+        flag = "usage_imputed"
+        if rate is None:
+            out.loc[rows, ["input_tokens", "output_tokens", "cost"]] = np.nan
+            flag = "usage_missing"
+        else:
+            elsewhere = priced & (out["model_id"] != model)
+            by_query = tok_in[elsewhere].groupby(out.loc[elsewhere, "query_id"]).median()
+            own_in = tok_in[own].groupby(out.loc[own, "dataset"]).mean()
+            own_out = tok_out[own].groupby(out.loc[own, "dataset"]).mean()
+            ds = out.loc[rows, "dataset"]
+            est_in = out.loc[rows, "query_id"].map(by_query).fillna(ds.map(own_in)).fillna(tok_in[own].mean())
+            est_out = ds.map(own_out).fillna(tok_out[own].mean())
+            est_in, est_out = est_in.round(), est_out.round()
+            out.loc[rows, "input_tokens"] = est_in
+            out.loc[rows, "output_tokens"] = est_out
+            out.loc[rows, "cost"] = (est_in * rate[0] + est_out * rate[1]).to_numpy()
+        out.loc[rows, "metadata"] = out.loc[rows, "metadata"].map(
+            lambda m: {**(m or {}), flag: True})
+    return out
```
