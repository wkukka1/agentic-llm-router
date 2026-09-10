# Query & model representation

Everything between the canonical Phase 0 tables and the NIRT model's input
tensors: the encoder **pathways**, the LLM **profile** texts, the two-layer
**NIRT dataset**, and the **taxonomy → relevance → FAISS warm-up** chain. All of
it is a pure function of the query/profile text and the train/test split — no
labels, no response matrix — so it is leakage-free.

---

## 1. Embedding pathways

[`src/router/embeddings/encoder.py`](../src/router/embeddings/encoder.py),
configured under `configs/phase0.yaml → embedding.pathways`. A **pathway** is a
named `(backend, model, pooling, normalize)` bundle. Phase 0 ships two:

| pathway | backend | model | pooling | normalize | used for |
|---------|---------|-------|---------|-----------|----------|
| `retrieval` | `sentence_transformer` | `all-MiniLM-L6-v2` (384-d) | — | yes | kNN warm-up, FAISS query bank |
| `irt` | `bert` | `bert-base-uncased` (768-d) | `mean` | no | **NIRT text pathway**: encodes queries *and* LLM profiles into one space |

`embedding.default_pathway` (= `irt`) is what a bare `load_encoder(cfg)` and the
build scripts use unless `--pathway` says otherwise.

### Why two

* **Retrieval** wants unit-norm sentence embeddings where cosine distance is
  meaningful — `all-MiniLM-L6-v2` is small, fast, tuned for exactly that.
* **IRT** follows the NIRT / IRT-Router papers: a raw BERT encoder (mean-pooled,
  un-normalized) feeds learned MLP heads that produce the item parameters
  (`b_q`, `a_q`, `r_q`). Crucially the **same** encoder embeds the LLM profile
  texts, so query items and model test-takers live in one text space and a
  cold-start model can be projected in from its profile.

Nothing here is fine-tuned in Phase 0. The BERT weights are frozen; Phase 1 owns
the projection heads.

### Backends

`bert` — HF `AutoModel` + `AutoTokenizer`. `pooling: cls` takes the `[CLS]`
vector; `pooling: mean` is attention-masked mean of the last hidden state.
Optional final L2 normalize.
`sentence_transformer` — a `SentenceTransformer` model; `normalize_embeddings`
and `max_seq_length` honored.
Determinism: eval mode, `torch.no_grad`, seeded (`seed` from config), no dropout.
Same config + same device ⇒ identical vectors (`config_fingerprint` in the
manifest).

### Stores (streaming + resumable)

`<embedding.cache_dir>/<kind>__<pathway>/` where `kind ∈ {query, model_profile}`:

| file | |
|------|--|
| `vectors.npy` | float32 `n × dim`, written via memmap (`np.lib.format.open_memmap`) |
| `ids.parquet` | `<id_field> → row` (`id_field ∈ {query_id, model_id}`) |
| `manifest.json` | backend, model, pooling, dim, `ids_fingerprint`, `config_fingerprint`, `complete` |
| `_progress.json` | present **only mid-build**: `rows_done` + fingerprints |

`build_store(dir, ids, texts, encoder, ...)` encodes batch-by-batch straight into
the memmap and rewrites `_progress.json` every `flush_every` batches. If the
process is killed (OOM, power loss), `vectors.npy` up to the last checkpoint is
valid; re-running with the **same ids + config** resumes from `rows_done`
(fingerprint-checked; mismatch ⇒ restart). On completion `_progress.json` is
deleted and `manifest.json` gets `complete: true`.

`EmbeddingStore.load(dir, mmap=True)` (default) memory-maps `vectors.npy` —
read-only, not resident. Call `store.close()` before deleting/rebuilding on
Windows. Legacy `embeddings*.{npy,parquet,json}` filenames are still read.

```bash
python scripts/embeddings/build_query_embeddings.py   --pathway irt        # resumes if interrupted
python scripts/embeddings/build_query_embeddings.py   --pathway irt --restart   # ignore progress
python scripts/embeddings/build_query_embeddings.py   --all-pathways --split train
python scripts/embeddings/build_profile_embeddings.py --all-pathways
```

### Adding a pathway

Add a block under `embedding.pathways` in the config; no code change:

```yaml
embedding:
  pathways:
    irt_large:
      backend: bert
      model_name: microsoft/deberta-v3-base
      pooling: mean
      normalize: false
```

---

## 2. LLM profiles

Files: [`configs/model_profiles.yaml`](../configs/model_profiles.yaml) (curated),
[`src/router/models/profiles.py`](../src/router/models/profiles.py) (builder),
`data/processed/model_profiles.parquet` (output).

### Purpose (NIRT / IRT-Router pathway)

Each candidate LLM gets a short natural-language **profile**, embedded by the
`irt` pathway (§1) into a text vector `p_m`. Phase 1 uses `p_m` to **initialise
the latent ability vector `theta_m`** through a learned projection — so a
**cold-start** model (no response data) can still be placed in ability space from
its description alone. This mirrors the "LLM description" nodes in GraphRouter /
IRT-Router.

### Profile text = curated + empirical

| part | source | example |
|------|--------|---------|
| **curated** | `configs/model_profiles.yaml` `feature` + structured facts (`params_b`, `context_window`, `release`, price); or a registry template when no entry exists | *"GPT-4 Turbo (November 2023 preview) is OpenAI's proprietary large multimodal model… 128000-token context window, released 2023-11, priced at $0.01/$0.03 per 1K…"* |
| **empirical** (optional) | computed from `responses.parquet` | *"Observed behaviour across 88 tasks: overall correctness 0.78; by area knowledge 0.84, coding 0.69, math 0.65; multiple-choice skill above chance +0.57; mean cost per query $0.0033; human-preference win-rate 0.69 (n=7387); LLM-judge win-rate 0.20 (n=109101)."* |

Set `profiles.include_empirical: false` in `configs/phase0.yaml` for
**description-only** profiles, exactly as in the paper. The empirical addendum
lets `theta_m` initialisation see the model's real strengths/weaknesses, not just
marketing copy, and is easy to ablate. It groups tasks with
`profiles.task_families` (a **coarse, fixed** benchmark grouping: math / code /
knowledge / reasoning / language_zh / chat) — **not** the learned ability
taxonomy (§4), which is deferred until after the AMIRT baseline.

### `model_profiles.parquet` columns

`model_id, model_name, provider, family, version, has_curated, feature,
profile_text, profile_version, n_observations, n_datasets, structured (json),
empirical (json)`

* `feature` — curated description only (use for description-only embeddings)
* `profile_text` — full text (curated + empirical); the default embedding input
* `structured` — `{params_b, context_window, release, modality, prices, provider, family, version}`
* `empirical` — `{overall_correctness, correctness_by_family, mc_skill_above_chance, mean_cost, pairwise:{source:{win_rate,n}}, …}`

### Build & access

```bash
python scripts/models/build_model_profiles.py            # or --print
python scripts/embeddings/build_profile_embeddings.py --pathway irt
#   --text-column feature   for description-only vectors
```

```python
d = load_phase1(load_config())
d.model_profiles()                        # DataFrame
d.profile_text("yi-34b-chat")             # str
d.profile_embeddings("irt")               # EmbeddingStore: model_id -> vector
d.profile_embedding("claude-2.0", "irt")  # np.ndarray (cold-start model)
```

### Known limitations

* Curated entries exist for 20 / 69 models (RouterBench candidates + high-traffic
  Arena models). The rest get a registry template + empirical addendum — fine for
  warm models, thinner for cold-start Arena-only models.
* `params_b` / prices for proprietary models are `null` (unknown), not guessed.
* No LLM-generated descriptions yet; the curated YAML is hand-written. An opt-in
  cached LLM-polish step could be added later.

---

## 3. NIRT training representation

[`src/router/data/nirt.py`](../src/router/data/nirt.py). Two layers, so 768-d
vectors are **never** copied into every training row.

```
data/processed/nirt_observations.parquet     (lightweight: ids + scalars)
        │
        ├── query_id  ──→  embeddings/query__<pathway>/        (shared memmap)
        └── model_id  ──→  embeddings/model_profile__<pathway>/ (shared memmap)
                                   │
                              NIRTDataset[i] → { query_embedding, model_embedding, target, ... }
```

### Layer 1 — `nirt_observations.parquet`

One row per `(query_id, model_id, metric)` **correctness** example:

| column | notes |
|--------|-------|
| `query_id`, `model_id` | join keys (canonical ids) |
| `target` | supervision signal; `score_kind=effective` (default) = chance-corrected where MC, else raw. `raw` / `corrected` also available |
| `score_raw` | original source score, always |
| `metric_type` | `accuracy` / `mc_accuracy` / `exact_match` / `f1` / `pass@1` / `acc` / `acc_norm` |
| `source` | `routerbench` / `lm_eval_harness` |
| `split` | `train` / `validation` / `test` (content-hash groups, leakage-free) |
| `cost`, `is_multiple_choice`, `n_choices` | reference features |

**Not included:** pairwise preference (Arena / Judge) — opponent-dependent, use
`Phase1Data.pairwise(...)` for a BT head. Cold-start models are excluded
(`models="warm"`); rows whose query is in no split are dropped.

```bash
python scripts/data/build_nirt_dataset.py                       # effective target, all metrics
python scripts/data/build_nirt_dataset.py --score-kind raw --metrics accuracy,mc_accuracy
# then embed just the queries this table needs (≈36.5k, not the full 197k corpus):
python scripts/embeddings/build_query_embeddings.py --pathway irt --from-nirt
python scripts/embeddings/build_profile_embeddings.py --pathway irt
```

### Layer 2 — `NIRTDataset`

A `torch.utils.data.Dataset` (works without torch too). Filters observations to
those with **both** embeddings present, precomputes integer row indices, and on
access returns memmap **views** — no per-row vector storage.

```python
d  = load_phase1(load_config())
ds = d.nirt_dataset(split="train", pathway="irt")     # NIRTDataset

ds[0]
# {"query_embedding": float32[768], "model_embedding": float32[768],
#  "target": float32, "metric": "mc_accuracy", "source": "routerbench", "cost": nan}

ds.query_dim, ds.model_dim         # 768, 768
ds.targets                          # float32[N]
ds.dropped                          # obs with no embedding (skipped)

for batch in ds.dataloader(batch_size=256, shuffle=True):
    q = batch["query_embedding"]    # torch.float32 [B, 768]
    ...
g = ds.gather()                     # {"query_embedding": [N,768], ...}  (materializes — copies)
```

`pathway` picks the encoder space to join against; `query_pathway` overrides
**only** the query store (used for the kNN-imputed representation — see
[nirt_model.md](nirt_model.md#knn-imputed-queries)); `query_features` names a
structured per-query feature vector concatenated to `e_q`. `from_config` raises a
clear error naming the build script if a store is missing.

### Why no vector duplication

* `nirt_observations.parquet` is ~pure scalars (~10 MB for 330k rows).
* `NIRTDataset` holds **references** to the two `EmbeddingStore.matrix` `np.memmap`
  arrays, so the 600 MB query matrix is paged from disk, not resident, and shared
  across every row.
* `__getitem__` slices those matrices — a 768-float view per side, per item.
* `gather()` is the one place that copies; it's opt-in and documented.

---

## 4. Ability taxonomy + FAISS query bank

The two deferred Phase 0 stages, built to support the baseline's relevance
conditioning (`r_q`) and warm-up blend.

### Artefacts

| artefact | path | shape | built by |
|---|---|---|---|
| cluster assignment | `data/taxonomy/clusters.parquet` | `query_id, cluster_id, cluster_prob, is_noise` | `scripts/taxonomy/cluster_queries.py` |
| cluster centroids | `data/taxonomy/centroids.npy` | `(C, dim)` unit vectors | ″ |
| clustering meta | `data/taxonomy/clustering.json` | params, sizes, noise fraction | ″ |
| taxonomy labels | `data/taxonomy/taxonomy.json` | per-cluster family / top terms / datasets / label | `scripts/taxonomy/generate_taxonomy.py` |
| relevance vectors `r_q` | `data/processed/embeddings/query_relevance__<pathway>/` | EmbeddingStore, `(n, C)` rows sum to 1 | `scripts/taxonomy/build_relevance.py` |
| FAISS query bank | `indexes/query_bank/` | `index.faiss`, `ids.parquet`, `manifest.json` (train queries only) | `scripts/retrieval/build_query_bank.py` |
| warm-up representations | `data/processed/embeddings/query_warmup__<pathway>/` | EmbeddingStore, `(n, model_dim)` | `scripts/retrieval/build_warmup.py` |

Or in one shot: `python -m router.phase0 --taxonomy --query-bank`.

### 4.1 Clustering (`router.taxonomy.clustering`)

`retrieval`-pathway query embeddings (MiniLM-384, the 36.5k queries that carry a
NIRT correctness observation) → **UMAP** (`n_components=5`, `n_neighbors=15`,
cosine, `random_state=seed`) → **HDBSCAN** (`min_cluster_size=150`,
`min_samples=10`, EOM). Result on the current data: **39 clusters, ~17 % noise**.
The parameters were tuned once (`configs/phase0.yaml → clustering`) — the MiniLM
space over these queries has one dominant blob, so smaller `min_cluster_size`
fragments into 200+ micro-clusters and larger collapses to 3. Determinism: UMAP
with a fixed `random_state` (single-threaded) + HDBSCAN ⇒ same embeddings +
config always give the same `cluster_id`. Centroids are the unit-normalised mean
of each cluster's members in the **original** embedding space (not UMAP space),
so `r_q` for any unseen query needs only its raw embedding and `centroids.npy`.

### 4.2 Relevance vectors `r_q` (`router.taxonomy.relevance`)

```
r_q = softmax( cos(e_q, centroid_c) / tau )      c = 1..C
```

A distribution over the `C` ability clusters: "which ability dimensions is this
query about?". `tau = 0.1` (config). Mean row entropy 2.15 nats (max `ln 39 =
3.66`), mean max-weight 0.47 — informative but not one-hot. Noise-point queries
still get an `r_q` (just flatter). Queries missing from the store fall back to
uniform `1/C` at join time. `r_q` conditions the Phase 1 discrimination head:
`a_q = softplus(f_a(e_q)) * sigmoid(W_r r_q)`, a per-latent-dimension relevance
gate (`~1` at init).

### 4.3 Taxonomy labels (`router.taxonomy.taxonomy`)

Per cluster, no LLM: dominant coarse benchmark **family**
(`profiles.task_families`), top **TF-IDF terms**, top source **datasets**, and a
short `label`. Enough to interpret the NIRT latent dimensions; an opt-in LLM
refinement (`labeling` config) is future work.

### 4.4 FAISS query bank (`router.retrieval.query_bank`)

`IndexFlatIP` over the **train-split** query embeddings only (29,172 vectors,
retrieval pathway, L2-normalised). Train-only is deliberate: putting
validation/test queries in the bank would leak them into the warm-up. Built once,
**never rebuilt or extended during model training**.
`QueryBank.search(vecs, k, exclude_ids=...)` returns `(neighbour_ids, cosine)`;
`exclude_ids` drops a query's own id.
`QueryBank.neighbor_weighted_mean(..., weighting="similarity"|"uniform")` is the
averaging primitive reused by the kNN-imputed representation.

### 4.5 Warm-up representations (`router.retrieval.warmup`)

For every NIRT query: the mean **irt-pathway** (BERT-768) embedding of its `k=5`
nearest *train* queries (neighbours found with the retrieval-pathway bank, self
excluded). Pre-computed and stored so the dataset joins it by `query_id` like
`r_q`; the model's `WarmupBlender` then does
`e_q' = (1-alpha) e_q + alpha * neighbour_mean`. Off by default
(`ablation.use_warmup`), on via `--warmup`.
