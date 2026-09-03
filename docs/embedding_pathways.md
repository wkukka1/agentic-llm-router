# Embedding pathways

[`src/router/embeddings/encoder.py`](../src/router/embeddings/encoder.py),
configured under `configs/phase0.yaml → embedding.pathways`.

A **pathway** is a named `(backend, model, pooling, normalize)` bundle. Phase 0
ships two:

| pathway | backend | model | pooling | normalize | used for |
|---------|---------|-------|---------|-----------|----------|
| `retrieval` | `sentence_transformer` | `all-MiniLM-L6-v2` (384-d) | — | yes | kNN warm-up, FAISS query bank (deferred) |
| `irt` | `bert` | `bert-base-uncased` (768-d) | `mean` | no | **NIRT text pathway**: encodes queries *and* LLM profiles into one space |

`embedding.default_pathway` (= `irt`) is what a bare `load_encoder(cfg)` and the
build scripts use unless `--pathway` says otherwise.

## Why two

* **Retrieval** wants unit-norm sentence embeddings where cosine distance is
  meaningful — `all-MiniLM-L6-v2` is small, fast, and tuned for exactly that.
* **IRT** follows the NIRT / IRT-Router papers: a raw BERT encoder (mean-pooled,
  un-normalized) feeds learned MLP heads that produce the item parameters
  (difficulty `b_q`, discrimination `a_q`, relevance `r_q`) in Phase 1. Crucially
  the **same** encoder embeds the LLM profile texts, so query items and model
  test-takers live in one text space and a cold-start model can be projected in
  from its profile.

Nothing here is fine-tuned in Phase 0. The BERT weights are frozen; Phase 1 owns
the projection heads.

## Backends

`bert` — HF `AutoModel` + `AutoTokenizer`. `pooling: cls` takes the `[CLS]`
vector; `pooling: mean` is attention-masked mean of the last hidden state.
Optional final L2 normalize.

`sentence_transformer` — a `SentenceTransformer` model; `normalize_embeddings`
and `max_seq_length` honored.

Determinism: eval mode, `torch.no_grad`, seeded (`seed` from config), no dropout.
Same config + same device ⇒ identical vectors (`config_fingerprint` in the
manifest).

## Stores (streaming + resumable)

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

```python
d = load_phase1(load_config())
d.query_embeddings("irt")       # EmbeddingStore or None
d.profile_embeddings("irt")
d.query_embeddings("retrieval")  # for the (deferred) FAISS bank
```

## Adding a pathway

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
