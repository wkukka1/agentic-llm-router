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

from .._normalize import unit_rows as _unit
from ..config import Config

INDEX_FILE, IDS_FILE, MANIFEST_FILE = "index.faiss", "ids.parquet", "manifest.json"


def query_bank_dir(cfg: Config) -> Path:
    return cfg.path("indexes") / "query_bank"


def _bank_query_ids(cfg: Config, split: str) -> list:
    """Training queries that carry a NIRT correctness observation."""
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
    return sorted(ids)


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

    from ..embeddings import EmbeddingStore, default_store_dir

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

    def search(self, vecs, k: Optional[int] = None, *,
               exclude_ids: Optional[Sequence[str]] = None) -> KNNResult:
        """kNN for a batch of query vectors. ``exclude_ids`` drops self-matches
        (pass each query's own id)."""
        k = int(k or self.default_k)
        q = self._prep(vecs)
        over = min(k + (1 if exclude_ids is not None else 0), len(self.ids))
        scores, idx = self.index.search(q, over)
        ids = np.asarray(self.ids, dtype=object)[idx]
        if exclude_ids is None:
            return KNNResult(ids[:, :k], scores[:, :k].astype(np.float32))
        keep_ids = np.empty((len(q), k), dtype=object)
        keep_sc = np.zeros((len(q), k), dtype=np.float32)
        for r, drop in enumerate(exclude_ids):
            row = [(i, s) for i, s in zip(ids[r], scores[r]) if i != drop][:k]
            keep_ids[r, :len(row)] = [i for i, _ in row]
            keep_sc[r, :len(row)] = [s for _, s in row]
        return KNNResult(keep_ids, keep_sc)

    def neighbor_mean(self, query_vecs, store, k: Optional[int] = None, *,
                      exclude_ids: Optional[Sequence[str]] = None) -> np.ndarray:
        """Mean embedding of each query's k nearest training queries -> ``(n, dim)``."""
        res = self.search(query_vecs, k=k, exclude_ids=exclude_ids)
        out = np.zeros((res.ids.shape[0], store.dim), dtype=np.float32)
        for r in range(res.ids.shape[0]):
            nbr = [i for i in res.ids[r] if i in store]
            if nbr:
                out[r] = store.gather(nbr).mean(axis=0)
        return out

    def neighbor_weighted_mean(self, query_vecs, store, k: Optional[int] = None, *,
                               exclude_ids: Optional[Sequence[str]] = None,
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
        res = self.search(query_vecs, k=k, exclude_ids=exclude_ids)
        n = res.ids.shape[0]
        out = np.zeros((n, store.dim), dtype=np.float32)
        found = np.zeros(n, dtype=np.int64)
        for r in range(n):
            pairs = [(i, s) for i, s in zip(res.ids[r], res.scores[r]) if i and i in store]
            if not pairs:
                continue
            nbr = [i for i, _ in pairs]
            found[r] = len(nbr)
            vecs = store.gather(nbr).astype(np.float64)
            if weighting == "similarity":
                w = np.clip(np.array([s for _, s in pairs], dtype=np.float64), 0.0, None)
                if w.sum() <= 1e-12:
                    w = np.ones(len(nbr), dtype=np.float64)
            else:
                w = np.ones(len(nbr), dtype=np.float64)
            out[r] = (vecs * (w / w.sum())[:, None]).sum(axis=0).astype(np.float32)
        return out, found
