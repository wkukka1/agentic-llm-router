# NIRT training representation

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

## Layer 1 — `nirt_observations.parquet`

One row per `(query_id, model_id, metric)` **correctness** example:

| column | notes |
|--------|-------|
| `query_id`, `model_id` | join keys (canonical ids) |
| `target` | supervision signal; `score_kind=effective` (default) = chance-corrected where MC, else raw. `raw` / `corrected` also available |
| `score_raw` | original source score, always |
| `metric_type` | `accuracy` / `mc_accuracy` / `exact_match` / `f1` / `pass@1` / `acc` / `acc_norm` |
| `source` | `routerbench` / `lm_eval_harness` |
| `split` | `train` / `validation` / `test` (from `data/splits`, content-hash groups, leakage-free) |
| `cost`, `is_multiple_choice`, `n_choices` | reference features |

**Not included:** pairwise preference (Arena / Judge). That signal is
opponent-dependent — use `Phase1Data.pairwise(...)` for a Bradley-Terry head.
Cold-start models are excluded (`models="warm"`); rows whose query is in no
split are dropped.

```bash
python scripts/data/build_nirt_dataset.py                       # effective target, all metrics
python scripts/data/build_nirt_dataset.py --score-kind raw --metrics accuracy,mc_accuracy
```

Then embed just the queries this table needs with the BERT pathway (≈36.5k, not
the full 197k corpus — much faster on CPU):

```bash
python scripts/embeddings/build_query_embeddings.py --pathway irt --from-nirt
python scripts/embeddings/build_profile_embeddings.py --pathway irt
```

## Layer 2 — `NIRTDataset`

A `torch.utils.data.Dataset` (works without torch too). Filters observations to
those with **both** embeddings present, precomputes integer row indices, and on
access returns memmap **views** — no per-row vector storage.

```python
from router.config import load_config
from router.data.phase1 import load_phase1

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
    m = batch["model_embedding"]    # torch.float32 [B, 768]
    y = batch["target"]             # torch.float32 [B]

g = ds.gather()                     # {"query_embedding": [N,768], ...}  (materializes — copies)
```

`pathway` picks which encoder space to join against (`irt` = BERT, the NIRT
text pathway; `retrieval` = MiniLM). `from_config` raises a clear error naming
the build script if a store is missing.

## Why no vector duplication

* `nirt_observations.parquet` is ~pure scalars (~10 MB for 330k rows).
* `NIRTDataset` holds **references** to the two `EmbeddingStore.matrix` arrays,
  which are `np.memmap` (`EmbeddingStore.load(..., mmap=True)`), so the 600 MB
  query matrix is paged from disk, not resident, and shared across every row.
* `__getitem__` slices those matrices — a 768-float view per side, per item.
* `gather()` is the one place that copies; it's opt-in and documented.
