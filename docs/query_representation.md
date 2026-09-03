# Query representation: ability taxonomy + FAISS query bank

These are the two deferred Phase 0 stages, built to support the Phase 1 NIRT
baseline's relevance conditioning (`r_q`) and warm-up blend. Everything here is a
pure function of the **query text** (via the frozen embeddings) and the
train/test split -- no labels, no response matrix -- so it is leakage-free.

## Artefacts

| artefact | path | shape | built by |
|---|---|---|---|
| cluster assignment | `data/taxonomy/clusters.parquet` | `query_id, cluster_id, cluster_prob, is_noise` | `scripts/taxonomy/cluster_queries.py` |
| cluster centroids | `data/taxonomy/centroids.npy` | `(C, dim)` unit vectors, row `c` = cluster `c` | ″ |
| clustering meta | `data/taxonomy/clustering.json` | params, sizes, noise fraction | ″ |
| taxonomy labels | `data/taxonomy/taxonomy.json` | per-cluster family / top terms / datasets / label | `scripts/taxonomy/generate_taxonomy.py` |
| relevance vectors `r_q` | `data/processed/embeddings/query_relevance__<pathway>/` | EmbeddingStore, `(n, C)` rows sum to 1 | `scripts/taxonomy/build_relevance.py` |
| FAISS query bank | `indexes/query_bank/` | `index.faiss`, `ids.parquet`, `manifest.json` (train queries only) | `scripts/retrieval/build_query_bank.py` |
| warm-up representations | `data/processed/embeddings/query_warmup__<pathway>/` | EmbeddingStore, `(n, model_dim)` | `scripts/retrieval/build_warmup.py` |

Or in one shot: `python -m router.phase0 --taxonomy --query-bank`.

## 1. Clustering (`router.taxonomy.clustering`)

`retrieval`-pathway query embeddings (MiniLM-384, the 36.5 k queries that carry a
NIRT correctness observation) -> **UMAP** (`n_components=5`, `n_neighbors=15`,
cosine, `random_state=seed`) -> **HDBSCAN** (`min_cluster_size=150`,
`min_samples=10`, EOM).

Result on the current data: **39 clusters, ~17 % noise**. The parameters were
tuned once (see `configs/phase0.yaml -> clustering`); the MiniLM space over these
queries has one dominant blob, so smaller `min_cluster_size` fragments into
200+ micro-clusters and larger collapses to 3. 39 is a reasonable "~20-40 ability
dimensions" target.

Determinism: UMAP with a fixed `random_state` (single-threaded) + HDBSCAN
(deterministic given its input) => the same embeddings + config always give the
same `cluster_id`.

Centroids are the **unit-normalised mean of each cluster's members in the
original embedding space** (not UMAP space), so `r_q` for any query -- including
an unseen one -- needs only its raw embedding and `centroids.npy`.

## 2. Relevance vectors `r_q` (`router.taxonomy.relevance`)

```
r_q = softmax( cos(e_q, centroid_c) / tau )      c = 1..C
```

A distribution over the `C` ability clusters: "which ability dimensions is this
query about?". `tau = 0.1` (config). On the current data mean row entropy is
2.15 nats (max `ln 39 = 3.66`), mean max-weight 0.47 -- informative but not
one-hot. Noise-point queries still get an `r_q` (they are just not near any
centroid, so the vector is flatter). Queries missing from the store fall back to
uniform `1/C` at join time.

`r_q` conditions the Phase 1 discrimination head: `a_q = softplus(f_a(e_q)) *
sigmoid(W_r r_q)`, a per-latent-dimension relevance gate (`~1` at init).

## 3. Taxonomy labels (`router.taxonomy.taxonomy`)

Per cluster, no LLM: dominant coarse benchmark **family**
(`profiles.task_families`), top **TF-IDF terms**, top source **datasets**, and a
short `label`. This "candidate taxonomy" is enough to interpret the NIRT latent
dimensions; an opt-in LLM refinement (`labeling` config,
`scripts/taxonomy/label_queries.py`) is future work.

## 4. FAISS query bank (`router.retrieval.query_bank`)

`IndexFlatIP` over the **train-split** query embeddings only (29 172 vectors,
retrieval pathway, L2-normalised). Train-only is deliberate: putting
validation/test queries in the bank would leak them into the Phase 1 warm-up.
The index is a Phase-0 artefact -- built once, **never rebuilt or extended during
model training**.

`QueryBank.search(vecs, k, exclude_ids=...)` returns `(neighbour_ids, cosine)`;
`exclude_ids` drops a query's own id so a train query is warmed up from its
*other* neighbours.

## 5. Warm-up representations (`router.retrieval.warmup`)

For every NIRT query: the mean **irt-pathway** (BERT-768, the space the NIRT
model works in) embedding of its `k=5` nearest *train* queries (neighbours found
with the retrieval-pathway bank, self excluded). Pre-computed and stored so the
Phase 1 dataset joins it by `query_id` like `r_q`; the model's `WarmupBlender`
then does `e_q' = (1-alpha) e_q + alpha * neighbour_mean`. Off by default
(`ablation.use_warmup`), on via `--warmup`.
