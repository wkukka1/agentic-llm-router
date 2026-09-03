# LLM profiles

Files:
[`configs/model_profiles.yaml`](../configs/model_profiles.yaml) (curated),
[`src/router/models/profiles.py`](../src/router/models/profiles.py) (builder),
`data/processed/model_profiles.parquet` (output).

## Purpose (NIRT / IRT-Router pathway)

Each candidate LLM gets a short natural-language **profile**. The profile is
embedded by the `irt` encoder pathway (BERT — see
[embedding_pathways.md](embedding_pathways.md)), giving every model a text
vector `p_m`. Phase 1 uses `p_m` to **initialise the latent ability vector
`theta_m`** through a learned projection — so a **cold-start** model (no
response data) can still be placed in ability space from its description alone.

This mirrors the "LLM description" nodes in GraphRouter / IRT-Router
(`"<model> is a <type> model with <context> context and <capabilities>"`),
embedded and used as model features.

## Profile text = curated + empirical

| part | source | example |
|------|--------|---------|
| **curated** | `configs/model_profiles.yaml` `feature` + structured facts (`params_b`, `context_window`, `release`, price); or a template from the model registry when no entry exists | *"GPT-4 Turbo (November 2023 preview) is OpenAI's proprietary large multimodal model… It has 128000-token context window, released 2023-11, priced at $0.01/$0.03 per 1K…"* |
| **empirical** (optional) | computed from `responses.parquet` | *"Observed behaviour in this dataset across 88 tasks: overall correctness 0.78; by area knowledge 0.84, coding 0.69, math 0.65; multiple-choice skill above chance +0.57; mean cost per query $0.0033; human-preference win-rate 0.69 (n=7387); LLM-judge win-rate 0.20 (n=109101)."* |

Set `profiles.include_empirical: false` in `configs/phase0.yaml` for
**description-only** profiles, exactly as in the paper. The empirical addendum is
a grounded enhancement — it lets `theta_m` initialisation see the model's real
strengths/weaknesses, not just marketing copy — and is easy to ablate.

The empirical addendum groups tasks with `profiles.task_families` (a **coarse,
fixed** benchmark grouping: math / code / knowledge / reasoning / language_zh /
chat). This is **not** the learned ability taxonomy — that stage is deferred
until after the AMIRT baseline.

## `model_profiles.parquet` columns

`model_id, model_name, provider, family, version, has_curated, feature,
profile_text, profile_version, n_observations, n_datasets, structured (json),
empirical (json)`

* `feature` — curated description only (use for description-only embeddings)
* `profile_text` — full text (curated + empirical); the default embedding input
* `structured` — `{params_b, context_window, release, modality, prices, provider, family, version}`
* `empirical` — `{overall_correctness, correctness_by_family, mc_skill_above_chance, mean_cost, pairwise:{source:{win_rate,n}}, …}`

## Build

```bash
python scripts/models/build_model_profiles.py            # or --print
python scripts/embeddings/build_profile_embeddings.py --pathway irt
#   --text-column feature   for description-only vectors
```

## Access from Phase 1

```python
d = load_phase1(load_config())
d.model_profiles()                      # DataFrame
d.profile_text("yi-34b-chat")           # str
d.profile_embeddings("irt")             # EmbeddingStore: model_id -> vector
d.profile_embedding("claude-2.0", "irt")  # np.ndarray (cold-start model)
```

## Known limitations

* Curated entries exist for 20 / 69 models (the RouterBench candidates + high-
  traffic Arena models). The rest get a registry template + empirical addendum —
  fine for warm models, thinner for cold-start Arena-only models.
* `params_b` / prices for proprietary models are `null` (unknown), not guessed.
* No LLM-generated descriptions yet; the curated YAML is hand-written. An opt-in
  cached LLM-polish step could be added later (same pattern as the deferred
  query-labeling stage).
