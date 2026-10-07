# MOD-001: the taxonomy clusters / centroids (and so `r_q`) are fit on validation, test and held-out OOD queries

- **Status:** fixed 2026-09-25, see [fix.md](fix.md)
- **Severity:** med
- **Where:** `_nirt_query_ids` ([clustering.py:125-134](../../../../src/training/taxonomy/clustering.py#L125-L134)) feeding `build_clusters` ([clustering.py:144-163](../../../../src/training/taxonomy/clustering.py#L144-L163)); called with no `query_ids` from [phase0.py:124](../../../../src/training/phase0.py#L124) and [scripts/taxonomy/cluster_queries.py:34](../../../../scripts/taxonomy/cluster_queries.py#L34)
- **Found:** 2026-09-24

## What

`build_clusters` defaults `query_ids` to `_nirt_query_ids(cfg)`, i.e. **every**
query in `nirt_observations`, whatever its split. `build_nirt_observations` only
drops queries that are in *no* split, so train, validation, test and the OOD
families all go into UMAP+HDBSCAN. The centroids (`centroids.npy`) are the
means of those clusters, and `relevance_from_embeddings` computes
`r_q = softmax(cos(e_q, centroid) / tau)` against them, so every query's `r_q`
(which conditions the discrimination head) is a function of centroids that saw
the test queries.

This contradicts the repo's own stated intent:

* [response_matrix.py:127](../../../../src/training/data/response_matrix.py#L127): `cluster_id` is "filled by clustering stage (**train only**)".
* [relevance.py:6-8](../../../../src/training/taxonomy/relevance.py#L6-L8): `r_q` is "a pure function of the frozen embedding + centroids (**leakage-free**)".
* The rest of the OOD pipeline is careful to be leakage-safe: the FAISS bank is train-only, and the OOD pass rebuilds it with the held-out families excluded (`build_query_bank(exclude_query_ids=...)`, `knn_impute` docstring).

## Evidence

Against the built data (`data/taxonomy/clusters.parquet`, `data/splits/*.json`):

| split | queries in split | queries clustered |
|---|---|---|
| train | 216,939 | 88,820 |
| validation | 27,072 | 11,033 |
| test | 27,235 | 11,144 |

`clustering.json` records `query_source: nirt_observations`. Of the 110,997
clustered queries, **19,850 (18%) belong to the held-out OOD families**
(`math`, `code`; `evaluation.nirt.ood.DEFAULT_HOLDOUT`), so the cluster
geometry is also fit on the families the OOD evaluation claims to have never
seen.

## Impact

Unsupervised (no labels), so this is transductive leakage of the *test
manifold*, not of targets. It is mild for the in-distribution test split, but
it is a real hole in the OOD experiment: a "held-out family" still has its own
cluster(s) and centroid(s), so `r_q` for an OOD query can point at a cluster
that exists only because that family was clustered. That flatters exactly the
number the leakage-safe FAISS bank was built to keep honest.

## Held-out-family variant (added 2026-09-25)

Train-only is not enough for a held-out-family run: `math` and `code` are held out
of the ordinary splits, so 15,784 of their 19,850 queries also sit in the train
split. As with the FAISS bank (`build_query_bank(exclude_query_ids=...)`), the
taxonomy has a separate leakage-safe variant:

* `build_clusters(exclude_query_ids=..., out_dir=...)` and
  `build_relevance(taxonomy_dir=..., out_dir=...)`; `load_clusters` /
  `load_centroids` / `load_meta` take an optional `directory`.
* `scripts/taxonomy/cluster_queries.py --exclude-ood-families` writes
  `<taxonomy>__ood/`; `scripts/taxonomy/build_relevance.py --ood` reads it and
  writes `query_relevance__<pathway>__ood/`. The main taxonomy and r_q store are
  never touched. `holdout_query_ids` (`evaluation.nirt.ood`) is shared with
  `build_query_bank.py`.
* Dry run on the real data (nothing written): 19,850 queries excluded -> 73,036
  clustered, **123 clusters**, noise 28.6%.
* Tests: `test_build_clusters_can_exclude_query_ids`,
  `test_clusters_and_relevance_can_be_written_to_a_separate_directory`,
  `test_ood_variant_locations_sit_beside_the_main_ones`,
  `test_holdout_query_ids_are_the_queries_of_the_held_out_families`.

## Correction: no OOD run consumes r_q today

This report originally called the leak "a real hole in the OOD experiment". Checked
2026-09-25: the main NIRT model and trainer (`router/nirt/model.py`,
`training/nirt/train.py`, `data/nirt.py`) never read `r_q`, so `train_nirt.py --ood`
is unaffected; only the **baseline** NIRT (`training/nirt/baseline/`) consumes it,
and it has no OOD mode. The IRT-Router suite's OOD split is a separate origin
(`test2`), already excluded by train-only clustering. So the leak matters for
(a) validation / test numbers of `r_q`-using models, and (b) any future OOD run of
a relevance-using model. Because no `--ood` script loads a taxonomy, nothing was
wired to select the `__ood` variant; that has to be added together with the first
consumer.
