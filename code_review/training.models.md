# Code review: training.models

**Files reviewed:** 2 (310 lines): `__init__.py`, `profiles.py` · **Context-only:** `src/router/embeddings/encoder.py`, `src/training/data/families.py`, `configs/phase0.yaml`, `configs/irt_router.yaml`
**Summary:** 4 findings: 3 correctness, 0 design, 1 performance · 4 confirmed, 0 suspected

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| TM-01 | `src/training/models/profiles.py:291-305` | correctness | medium | confirmed |
| TM-02 | `src/training/models/profiles.py:218` | correctness | low | confirmed |
| TM-03 | `src/training/models/profiles.py:170-174` | correctness | low | confirmed |
| TM-04 | `src/training/models/profiles.py:64-66` | performance | low | confirmed |

---

## src/training/models/profiles.py

### TM-01: Changed profiles are never re-rendered or re-embedded without `--restart` `:291-305`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
291:    try:
292:        profiles = read_model_profiles(cfg)
301:    return build_store(
302:        out_dir, profiles["model_id"].tolist(), profiles[text_column].tolist(), encoder,
```

**What breaks:** Two layers of caching hide profile changes:
- **Stale parquet:** `build_profile_embeddings` rebuilds `model_profiles.parquet` only when the file is missing. Editing `configs/model_profiles.yaml`, toggling `profiles.include_empirical`, or rebuilding `responses.parquet` keeps the old profile text.
- **Stale vectors:** even with freshly re-rendered text, `build_store` returns the existing store whenever the model id list matches, because the texts are never fingerprinted (RE-01). Switching `text_column` from `profile_text` to `feature` also returns the old vectors, although `text_column` is recorded in `manifest_extra`.

So `theta_m` initialisation and every projected-mode model are trained on stale profile embeddings, with no warning. Only `restart=True` recovers.
**Evidence:** Traced `:291-305` into `encoder.py:439-443`, and confirmed the stale-return behaviour with the RE-01 probe.
**Direction:** Fingerprint the texts in `build_store` (the RE-01 fix), and re-render profiles when the config or responses are newer than the parquet.

### TM-02: The code default for `include_empirical` is `True`, which bakes test-split outcomes into profiles `:218`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
218:    include_emp = bool(pc.get("include_empirical", True))
```

**What breaks:** `empirical_stats` (`:64-114`) summarises a model's correctness, per-family correctness, cost and win rate over all responses, including validation and test queries and cold-start models' own observations. With the flag on, those numbers are written into `profile_text` and embedded. The projected model and `cold_start_eval`, which claims to place a model "from its profile alone", then see test outcomes. Both shipped configs set the flag to `false` (`configs/phase0.yaml:116`, `configs/irt_router.yaml:114`). The leak therefore occurs only for a config that omits the key, where the code default silently turns it on.
**Evidence:** Read `:64-114` and `:212-248`. `empirical_stats` applies no split filter.
**Direction:** Default to `False`, or compute the stats on train-split rows of warm models only.

### TM-03: The price sentence renders `$None` when only the input price is set `:170-174`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
170:    if structured["input_price_per_1k"] is not None:
171:        facts.append(
172:            f"priced at ${structured['input_price_per_1k']}/${structured['output_price_per_1k']} "
```

**What breaks:** A curated entry that has `input_price_per_1k` but no `output_price_per_1k` produces "priced at $0.01/$None per 1K input/output tokens". That text is embedded into the profile vector.
**Evidence:** Read `:151-175`.
**Direction:** Require both prices, or render them separately.

### TM-04: Per-model stats re-filter the whole responses frame for every model `:64-66`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
66:    g = responses[responses["model_id"] == model_id]
```

**What breaks:** `build_model_profiles` calls `empirical_stats` once per model (`:224-225`), and each call scans all rows (about 800k) with a string comparison. That is `O(M × N)`, where one `groupby("model_id")` would do. The cost matters when profiles are rebuilt for large pools.
**Evidence:** Read `:212-248`.
**Direction:** Iterate `responses.groupby("model_id")` once and pass each group in.

---

## src/training/models/__init__.py

No findings.

---

## Checked and ruled out
- **`read_model_profiles` on a missing file:** `pd.read_parquet` raises `FileNotFoundError`, which `build_profile_embeddings` catches.
- **The pairwise label `"human-preference" if "arena" in src`:** `anchor_judge` and `gpt4_judge` are both correctly labelled "LLM-judge".
- **`mc_skill_above_chance` using the raw rather than the corrected score:** this is intended, since it measures skill above chance.
