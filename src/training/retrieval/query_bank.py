"""FAISS index over training-query embeddings + kNN lookup.

``build_query_bank`` writes the index; ``QueryBank.load`` reads it back for the
Phase 1 warm-up blend. Ids + vectors only -- no targets, no response matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from router._normalize import unit_rows as _unit
from router.config import Config

INDEX_FILE, IDS_FILE, MANIFEST_FILE = "index.faiss", "ids.parquet", "manifest.json"


def query_bank_dir(cfg: Config) -> Path:
    return cfg.path("indexes") / "query_bank"


def _bank_query_source(cfg: Config) -> str:
    """Which query set :func:`_bank_query_ids` draws from (recorded in the manifest)."""
    try:
        p = cfg.path("processed") / "nirt_observations.parquet"
    except AttributeError:  # config stub without paths (ids supplied another way)
        return "unknown"
    return "nirt_observations" if p.exists() else "all_split_queries"


def _bank_query_ids(cfg: Config, split: str) -> list:
    """Training queries that carry a NIRT correctness observation."""
    import warnings

    import pandas as pd

    from ..data.splits import load_splits

    splits = load_splits(cfg)
    if splits.get(split) is None:
        raise FileNotFoundError(f"split '{split}' not built; run scripts/data/build_splits.py")
    ids = set(splits[split]["query_ids"])
    p = cfg.path("processed") / "nirt_observations.parquet"
    if p.exists():
        obs = pd.read_parquet(p, columns=["query_id", "split"])
        ids &= set(obs.loc[obs["split"] == split, "query_id"])
    else:
        warnings.warn(
            f"{p} not built: the query bank indexes EVERY '{split}' query, including "
            f"pairwise-only prompts with no correctness label. Build the NIRT "
            f"observation table first (scripts/data/build_nirt_dataset.py)."
        )
    return sorted(ids)


def content_groups(cfg: Config) -> Optional[dict]:
    """``query_id -> content_hash`` from ``queries.parquet`` (``None`` if absent).

    Text-identical queries (RouterBench 0-/5-shot twins, cross-source
    duplicates) share a content hash; kNN self-exclusion must drop the whole
    group, not just the query's own id."""
    import pandas as pd

    p = cfg.path("processed") / "queries.parquet"
    if not p.exists():
        return None
    q = pd.read_parquet(p, columns=["query_id", "content_hash"])
    return dict(zip(q["query_id"].astype(str), q["content_hash"].astype(str)))


def build_query_bank(cfg: Config, *, split: str = "train", pathway: Optional[str] = None,
                     index_type: Optional[str] = None, save: bool = True,
                     exclude_query_ids: Optional[Sequence[str]] = None,
                     holdout_families: Optional[Sequence[str]] = None,
                     out_dir: Optional[str | Path] = None) -> "QueryBank":
    """Build the FAISS bank over ``split`` query embeddings.

    ``exclude_query_ids`` are dropped before indexing -- pass the union of the
    held-out OOD families to get a leakage-safe bank for imputing OOD queries.
    ``out_dir`` overrides the default ``indexes/query_bank`` location.
    """
    import faiss

    from router.embeddings import EmbeddingStore, default_store_dir

    pathway = pathway or cfg.get("retrieval.pathway", "retrieval")
    index_type = index_type or cfg.get("retrieval.index_type", "flat_ip")
    store_dir = default_store_dir(cfg, "query", pathway)
    if not EmbeddingStore.exists(store_dir):
        raise FileNotFoundError(f"query embeddings for pathway '{pathway}' not built ({store_dir})")
    store = EmbeddingStore.load(store_dir)

    drop = set(map(str, exclude_query_ids or ()))
    ids = [q for q in _bank_query_ids(cfg, split) if q in store and str(q) not in drop]
    if not ids:
        raise ValueError(f"no '{split}' query ids present in embedding store '{pathway}'")
    X = np.ascontiguousarray(store.gather(ids), dtype=np.float32)

    if index_type == "flat_ip":
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(_unit(X).astype(np.float32))
        normalized = True
    elif index_type == "flat_l2":
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
        normalized = False
    else:
        raise ValueError(f"unknown index_type {index_type!r}; use flat_ip | flat_l2")

    bank = QueryBank(index, ids, {
        "split": split, "pathway": pathway, "index_type": index_type, "normalized": normalized,
        "dim": int(X.shape[1]), "count": len(ids), "default_k": int(cfg.get("retrieval.k", 5)),
        "query_embeddings_fingerprint": store.manifest.get("ids_fingerprint"),
        "seed": int(cfg.get("seed", 42)),
        "excluded_count": len(drop),
        "query_set": _bank_query_source(cfg),
        "holdout_families": list(holdout_families) if holdout_families else None,
    })
    if save:
        bank.save(Path(out_dir) if out_dir is not None else query_bank_dir(cfg))
    return bank


@dataclass
class KNNResult:
    ids: np.ndarray        # (n, k) str
    scores: np.ndarray     # (n, k) cosine sim (flat_ip) or L2 distance (flat_l2)


class QueryBank:
    def __init__(self, index, ids: Sequence[str], manifest: dict):
        self.index = index
        self.ids = list(ids)
        self.manifest = dict(manifest)
        self.normalized = bool(manifest.get("normalized", True))
        self.default_k = int(manifest.get("default_k", 5))

    def __len__(self) -> int:
        return len(self.ids)

    def save(self, directory: str | Path) -> Path:
        import faiss
        import pandas as pd

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / INDEX_FILE))
        pd.DataFrame({"query_id": self.ids, "row": range(len(self.ids))}).to_parquet(
            d / IDS_FILE, index=False)
        (d / MANIFEST_FILE).write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        return d

    @classmethod
    def load(cls, cfg_or_dir) -> "QueryBank":
        import faiss
        import pandas as pd

        d = Path(cfg_or_dir if isinstance(cfg_or_dir, (str, Path)) else query_bank_dir(cfg_or_dir))
        if not (d / MANIFEST_FILE).exists():
            raise FileNotFoundError(
                f"query bank not built ({d}); run scripts/retrieval/build_query_bank.py")
        return cls(faiss.read_index(str(d / INDEX_FILE)),
                   pd.read_parquet(d / IDS_FILE).sort_values("row")["query_id"].tolist(),
                   json.loads((d / MANIFEST_FILE).read_text(encoding="utf-8")))

    @classmethod
    def exists(cls, cfg) -> bool:
        return (query_bank_dir(cfg) / MANIFEST_FILE).exists()

    def _prep(self, vecs) -> np.ndarray:
        q = np.ascontiguousarray(np.atleast_2d(np.asarray(vecs, dtype=np.float32)))
        return _unit(q).astype(np.float32) if self.normalized else q

    def _max_group_size(self, groups: dict) -> int:
        from collections import Counter

        cache = getattr(self, "_group_size_cache", None)
        if cache is None or cache[0] is not groups:
            counts = Counter(groups.get(i, i) for i in self.ids)
            cache = (groups, max(counts.values(), default=1))
            self._group_size_cache = cache
        return cache[1]

    def search(self, vecs, k: Optional[int] = None, *,
               exclude_ids: Optional[Sequence[str]] = None,
               exclude_groups: Optional[dict] = None) -> KNNResult:
        """kNN for a batch of query vectors. ``exclude_ids`` drops self-matches
        (pass each query's own id). ``exclude_groups`` (``id -> group key``, e.g.
        :func:`content_groups`) extends that to every bank query sharing the
        excluded id's group, so a text-identical twin is not its neighbour.
        Rows left short of ``k`` are padded with ``None`` ids at score 0."""
        k = int(k or self.default_k)
        q = self._prep(vecs)
        extra = 0
        if exclude_ids is not None:
            extra = self._max_group_size(exclude_groups) if exclude_groups else 1
        over = min(k + extra, len(self.ids))
        scores, idx = self.index.search(q, over)
        ids = np.asarray(self.ids, dtype=object)[idx]
        if exclude_ids is None:
            return KNNResult(ids[:, :k], scores[:, :k].astype(np.float32))
        drop = np.asarray([str(x) for x in exclude_ids], dtype=object)[:, None]
        mask = ids == drop
        if exclude_groups:
            grp = np.vectorize(lambda i: exclude_groups.get(i, i), otypes=[object])
            mask |= grp(ids) == grp(drop)
        order = np.argsort(mask, axis=1, kind="stable")[:, :k]   # kept first, original order
        keep_ids = np.take_along_axis(ids, order, axis=1)
        keep_sc = np.take_along_axis(scores, order, axis=1).astype(np.float32)
        dropped = np.take_along_axis(mask, order, axis=1)
        keep_ids[dropped], keep_sc[dropped] = None, 0.0
        if keep_ids.shape[1] < k:                                 # bank smaller than k
            pad = k - keep_ids.shape[1]
            keep_ids = np.concatenate([keep_ids, np.full((len(q), pad), None, object)], axis=1)
            keep_sc = np.concatenate([keep_sc, np.zeros((len(q), pad), np.float32)], axis=1)
        return KNNResult(keep_ids, keep_sc)

    def neighbor_mean(self, query_vecs, store, k: Optional[int] = None, *,
                      exclude_ids: Optional[Sequence[str]] = None,
                      exclude_groups: Optional[dict] = None) -> np.ndarray:
        """Mean embedding of each query's k nearest training queries -> ``(n, dim)``."""
        return self.neighbor_weighted_mean(query_vecs, store, k=k, exclude_ids=exclude_ids,
                                           exclude_groups=exclude_groups, weighting="uniform")[0]

    def neighbor_weighted_mean(self, query_vecs, store, k: Optional[int] = None, *,
                               exclude_ids: Optional[Sequence[str]] = None,
                               exclude_groups: Optional[dict] = None,
                               weighting: str = "similarity") -> tuple[np.ndarray, np.ndarray]:
        """Weighted mean embedding of each query's k nearest training queries.

        ``weighting="similarity"`` weights neighbour ``i`` by ``max(cos_i, 0)``
        (the flat_ip score); a row whose weights all vanish falls back to a plain
        mean. ``weighting="uniform"`` reproduces :meth:`neighbor_mean`.

        Returns ``(vectors (n, dim), found (n,) int)`` -- ``found`` is the number
        of usable neighbours per row, so the caller can substitute a fallback
        where it is 0.
        """
        if weighting not in ("similarity", "uniform"):
            raise ValueError(f"weighting must be 'similarity' | 'uniform', got {weighting!r}")
        res = self.search(query_vecs, k=k, exclude_ids=exclude_ids, exclude_groups=exclude_groups)
        n, kk = res.ids.shape
        out = np.zeros((n, store.dim), dtype=np.float32)
        if n == 0 or len(store) == 0:
            return out, np.zeros(n, dtype=np.int64)
        # whole (n, k) id matrix -> store rows at once (missing / padded -> -1)
        index = store._index
        rows = np.fromiter((index.get(i, -1) if i else -1 for i in res.ids.ravel()),
                           dtype=np.int64, count=n * kk).reshape(n, kk)
        valid = rows >= 0
        found = valid.sum(axis=1).astype(np.int64)
        uniform = valid.astype(np.float64)
        if weighting == "similarity":
            w = np.clip(res.scores.astype(np.float64), 0.0, None) * valid
            w = np.where((w.sum(axis=1, keepdims=True) > 1e-12), w, uniform)
        else:
            w = uniform
        wsum = w.sum(axis=1, keepdims=True)
        w = np.divide(w, wsum, out=np.zeros_like(w), where=wsum > 0)
        vecs = np.asarray(store.matrix)[np.where(valid, rows, 0)].astype(np.float64)  # (n, k, dim)
        out[:] = np.einsum("nk,nkd->nd", w, vecs).astype(np.float32)
        return out, found
