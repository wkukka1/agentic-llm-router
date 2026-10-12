# MOD-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-25, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

; **the on-disk taxonomy has not been rebuilt**, so `data/taxonomy/` still holds the leaky clusters. Regression tests `tests/training/taxonomy/test_taxonomy.py::test_build_clusters_fits_on_train_queries_only` and `::test_relevance_is_still_built_for_every_query_not_just_the_clustered_ones`. (Originally reproduced by `known_bugs/_repros/repro_cluster_leak.py`.)

* `build_clusters(..., split="train")` fits UMAP + HDBSCAN on the queries of one
  split, default **train** (`_nirt_query_ids(cfg, split=...)` does the filtering,
  the same way `build_query_bank` already does). `split=None` restores the old
  all-splits behaviour. `clustering.json` records
  `query_source: "nirt_observations:train"`.
* `build_relevance` now defaults to **every** NIRT query, not the clustered ids.
  Without this the train-only clusters would have left validation / test / ood
  queries without an `r_q`. `r_q` is a pure function of embedding + centroids,
  so it is computed for all queries and is leakage-free once the centroids are
  train-only.

**Dry run on the real data (nothing written):** train-only clustering gives
**118 clusters** (was 154), noise 26.0% (was 26.9%), from 88,820 train queries
(was 110,997), in about 2 minutes.

## Still open

* **Rebuild not done.** `data/taxonomy/*` and the `query_relevance__*` store still
  come from the all-splits fit. Rebuilding changes the cluster count (154 -> 118),
  so every checkpoint trained against the old `r_q` (`artifacts/phase1`,
  `phase2`, `irt_router`) becomes stale and must be retrained; `load_relevance`'s
  centroid-fingerprint guard will refuse a mismatched store.
* **The `__ood` variant is not built** and nothing loads it yet (see above).

## Fix suggested when the bug was filed

Give `build_clusters` a `split="train"` default (intersect `_nirt_query_ids`
with `splits["train"]`) and let the OOD pass exclude the held-out families the
same way `build_query_bank` does. Rebuild `r_q` afterwards; NIRT checkpoints
trained on the old `r_q` are then stale (`load_relevance` already refuses a
centroid-fingerprint mismatch, so a rebuild is detected rather than silent).

## Regression tests

`tests/training/taxonomy/test_taxonomy.py:124` `test_build_clusters_fits_on_train_queries_only`

```python
def test_build_clusters_fits_on_train_queries_only(tmp_path, monkeypatch):
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    from training.taxonomy.clustering import build_clusters, load_centroids, load_clusters, load_meta

    cfg, ids, split, axes = _split_world(tmp_path, monkeypatch)
    build_clusters(cfg)

    clustered = set(load_clusters(cfg)["query_id"])
    assert clustered == {q for q, s in zip(ids, split) if s == "train"}
    # no centroid points at the blob that exists only in test / ood
    centroids = load_centroids(cfg)
    assert (centroids @ (axes[3] / 3.0)).max() < 0.5
    assert (centroids @ (axes[4] / 3.0)).max() < 0.5
    assert load_meta(cfg)["query_source"] == "nirt_observations:train"
```

`tests/training/taxonomy/test_taxonomy.py:141` `test_relevance_is_still_built_for_every_query_not_just_the_clustered_ones`

```python
def test_relevance_is_still_built_for_every_query_not_just_the_clustered_ones(tmp_path, monkeypatch):
    """r_q is a pure function of embedding + centroids, so validation / test / ood
    queries still need one even though only train queries shaped the centroids."""
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    from training.taxonomy.clustering import build_clusters
    from training.taxonomy.relevance import build_relevance

    cfg, ids, split, _ = _split_world(tmp_path, monkeypatch)
    build_clusters(cfg)
    rel = build_relevance(cfg)
    assert set(rel.ids) == set(ids)
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/training/taxonomy/clustering.py`

```diff
--- a/src/training/taxonomy/clustering.py
+++ b/src/training/taxonomy/clustering.py
@@ -122,18 +122,29 @@ def cluster_embeddings(embeddings, config: ClusterConfig, *, ids=None) -> Cluste
 # --------------------------------------------------------------------------- #
 # build / persist                                                             #
 # --------------------------------------------------------------------------- #
-def _nirt_query_ids(cfg: Config) -> list:
+def _nirt_query_ids(cfg: Config, *, split: Optional[str] = None) -> list:
     """Query ids carrying a NIRT correctness observation -- the observation
     table is built on demand (in memory) if it doesn't exist yet, rather than
     silently widening to every query in ``queries.parquet`` (including
     pairwise-only prompts with no correctness label) and making the taxonomy
-    depend on build order (XD-05 / XA-11)."""
+    depend on build order (XD-05 / XA-11).
+
+    ``split`` limits the ids to one split (``"train"``); ``None`` returns every
+    split -- what ``r_q`` needs, since it is computed for all queries."""
     from ..data.nirt import observations
 
     obs = observations(cfg, build=True)
+    if split is not None:
+        obs = obs[obs["split"] == split]
     return sorted(obs["query_id"].unique())
 
 
+def ood_taxonomy_dir(cfg: Config) -> Path:
+    """Where the held-out-family taxonomy goes: beside the main one, never over it."""
+    d = cfg.path("taxonomy")
+    return d.with_name(d.name + "__ood")
+
+
 def centroids_fingerprint(centroids: np.ndarray) -> str:
     import hashlib
 
@@ -142,7 +153,17 @@ def centroids_fingerprint(centroids: np.ndarray) -> str:
 
 
 def build_clusters(cfg: Config, *, pathway=None, query_ids=None, config: Optional[ClusterConfig] = None,
-                   save: bool = True) -> ClusterResult:
+                   save: bool = True, split: Optional[str] = "train",
+                   exclude_query_ids=None, out_dir: Optional[Path] = None) -> ClusterResult:
+    """UMAP + HDBSCAN over the query embeddings of ``split`` (default: train).
+
+    Fitting on train only keeps the centroids -- and so every query's ``r_q`` --
+    from depending on validation, test or held-out OOD queries. ``split=None``
+    fits on every NIRT query (the old behaviour; leaks the test manifold).
+
+    ``exclude_query_ids`` are dropped before clustering -- pass the held-out OOD
+    families' ids for a leakage-safe variant (as ``build_query_bank`` does), and
+    ``out_dir`` to write it beside, not over, the main taxonomy."""
     from router.embeddings import EmbeddingStore, default_store_dir
 
     config = config or ClusterConfig.from_config(cfg, pathway=pathway)
@@ -151,23 +172,27 @@ def build_clusters(cfg: Config, *, pathway=None, query_ids=None, config: Optiona
         raise FileNotFoundError(
             f"query embeddings for pathway '{config.pathway}' not built ({store_dir})")
     store = EmbeddingStore.load(store_dir)
-    source = "explicit" if query_ids else "nirt_observations"
-    ids = [q for q in (query_ids or _nirt_query_ids(cfg)) if q in store]
+    source = ("explicit" if query_ids
+              else f"nirt_observations:{split}" if split else "nirt_observations")
+    drop = set(map(str, exclude_query_ids or ()))
+    ids = [q for q in (query_ids or _nirt_query_ids(cfg, split=split))
+           if q in store and str(q) not in drop]
     if not ids:
         raise ValueError("no query ids present in the embedding store")
 
     result = cluster_embeddings(store.gather(ids), config, ids=ids)
     if save:
         write_clusters(result, cfg, store_fingerprint=store.manifest.get("ids_fingerprint"),
-                       query_source=source)
+                       query_source=source, out_dir=out_dir, excluded_count=len(drop))
     return result
 
 
 def write_clusters(result: ClusterResult, cfg: Config, *, store_fingerprint=None,
-                   query_source: Optional[str] = None) -> Path:
+                   query_source: Optional[str] = None, out_dir: Optional[Path] = None,
+                   excluded_count: int = 0) -> Path:
     import pandas as pd
 
-    out = cfg.path("taxonomy")
+    out = Path(out_dir) if out_dir is not None else cfg.path("taxonomy")
     out.mkdir(parents=True, exist_ok=True)
     ids = result.ids if result.ids is not None else list(range(len(result.labels)))
     pd.DataFrame({"query_id": ids, "cluster_id": result.labels.astype(int),
@@ -177,7 +202,7 @@ def write_clusters(result: ClusterResult, cfg: Config, *, store_fingerprint=None
     (out / META_FILE).write_text(json.dumps({
         "method": "umap_hdbscan", "pathway": result.config.pathway,
         "config": result.config.to_dict(), "query_embeddings_fingerprint": store_fingerprint,
-        "query_source": query_source,
+        "query_source": query_source, "excluded_count": int(excluded_count),
         "centroids_fingerprint": centroids_fingerprint(result.centroids),
         "taxonomy_version": int(cfg.get("taxonomy.version", 1)),
         **result.summary(), **result.extra,
@@ -185,23 +210,23 @@ def write_clusters(result: ClusterResult, cfg: Config, *, store_fingerprint=None
     return out / META_FILE
 
 
-def _load(cfg: Config, name: str, loader):
-    p = cfg.path("taxonomy") / name
+def _load(cfg: Config, name: str, loader, directory: Optional[Path] = None):
+    p = (Path(directory) if directory is not None else cfg.path("taxonomy")) / name
     if not p.exists():
         raise FileNotFoundError(f"{p} not built; run scripts/taxonomy/cluster_queries.py")
     return loader(p)
 
 
-def load_clusters(cfg: Config):
+def load_clusters(cfg: Config, directory: Optional[Path] = None):
     import pandas as pd
 
-    return _load(cfg, CLUSTERS_FILE, pd.read_parquet)
+    return _load(cfg, CLUSTERS_FILE, pd.read_parquet, directory)
 
 
-def load_centroids(cfg: Config) -> np.ndarray:
-    return _load(cfg, CENTROIDS_FILE, np.load)
+def load_centroids(cfg: Config, directory: Optional[Path] = None) -> np.ndarray:
+    return _load(cfg, CENTROIDS_FILE, np.load, directory)
 
 
-def load_meta(cfg: Config) -> dict:
-    p = cfg.path("taxonomy") / META_FILE
+def load_meta(cfg: Config, directory: Optional[Path] = None) -> dict:
+    p = (Path(directory) if directory is not None else cfg.path("taxonomy")) / META_FILE
     return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
```

`src/training/taxonomy/relevance.py`

```diff
--- a/src/training/taxonomy/relevance.py
+++ b/src/training/taxonomy/relevance.py
@@ -17,7 +17,7 @@ from typing import Optional, Sequence
 import numpy as np
 
 from router.config import Config
-from .clustering import _unit, centroids_fingerprint, load_centroids, load_clusters, load_meta
+from .clustering import _nirt_query_ids, _unit, centroids_fingerprint, load_centroids, load_meta
 
 DEFAULT_TAU = 0.1
 
@@ -41,16 +41,31 @@ def relevance_store_dir(cfg: Config, pathway: str) -> Path:
     return default_store_dir(cfg, "query_relevance", pathway)
 
 
+def ood_relevance_dir(cfg: Config, pathway: str) -> Path:
+    """Where the held-out-family ``r_q`` store goes: beside the main one."""
+    d = relevance_store_dir(cfg, pathway)
+    return d.with_name(d.name + "__ood")
+
+
 def build_relevance(cfg: Config, *, pathway: Optional[str] = None, tau: float = DEFAULT_TAU,
-                    query_ids: Optional[Sequence[str]] = None, save: bool = True):
-    """Build + save ``r_q`` for every clustered NIRT query."""
+                    query_ids: Optional[Sequence[str]] = None, save: bool = True,
+                    taxonomy_dir: Optional[Path] = None, out_dir: Optional[Path] = None):
+    """Build + save ``r_q`` for every NIRT query, whatever its split.
+
+    The centroids are fit on train queries only, but ``r_q`` is a pure function of
+    a query's embedding and the centroids, so validation / test / ood queries get
+    one too (defaulting to the clustered ids would leave them without).
+
+    ``taxonomy_dir`` reads the centroids from a variant taxonomy (e.g. one fit with
+    held-out families excluded) and ``out_dir`` writes the store beside, not over,
+    the main one."""
     from router.embeddings import EmbeddingStore, default_store_dir
 
     pw = _pathway(cfg, pathway)
-    centroids = load_centroids(cfg)
+    centroids = load_centroids(cfg, taxonomy_dir)
     store = EmbeddingStore.load(default_store_dir(cfg, "query", pw))
     ids = [q for q in (list(query_ids) if query_ids is not None
-                       else load_clusters(cfg)["query_id"].tolist()) if q in store]
+                       else _nirt_query_ids(cfg)) if q in store]
     R = relevance_from_embeddings(store.gather(ids), centroids, tau=tau)
     manifest = {"kind": "query_relevance", "pathway": pw, "dim": int(centroids.shape[0]),
                 "id_field": "query_id", "count": len(ids), "tau": float(tau), "normalized": True,
@@ -59,7 +74,7 @@ def build_relevance(cfg: Config, *, pathway: Optional[str] = None, tau: float =
                 "taxonomy_version": int(cfg.get("taxonomy.version", 1))}
     rel = EmbeddingStore(ids, R, manifest, id_field="query_id")
     if save:
-        rel.save(relevance_store_dir(cfg, pw))
+        rel.save(Path(out_dir) if out_dir is not None else relevance_store_dir(cfg, pw))
     return rel
 
 
```
