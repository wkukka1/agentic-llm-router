"""Pre-computed Phase 1 warm-up representations.

Per NIRT query: the mean ``model_pathway`` (irt) embedding of its ``k`` nearest
*training* queries (found via the retrieval-pathway FAISS bank, self-excluded),
saved as an :class:`~router.embeddings.EmbeddingStore` the dataset joins by
``query_id`` like ``r_q``. Built once; never rebuilt during training.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from router.config import Config
from router.embeddings import EmbeddingStore, default_store_dir
from .query_bank import QueryBank, content_groups


def warmup_store_dir(cfg: Config, pathway: str):
    return default_store_dir(cfg, "query_warmup", pathway)


def build_warmup_representations(
    cfg: Config,
    *,
    model_pathway: Optional[str] = None,      # embedding space the blend happens in
    k: Optional[int] = None,
    query_ids: Optional[Sequence[str]] = None,
    save: bool = True,
) -> EmbeddingStore:
    model_pathway = model_pathway or cfg.get("data.pathway", "irt")
    k = int(k or cfg.get("retrieval.k", 5))

    bank = QueryBank.load(cfg)
    retr_pw = bank.manifest.get("pathway", "retrieval")
    retr_store = EmbeddingStore.load(default_store_dir(cfg, "query", retr_pw))
    model_store = EmbeddingStore.load(default_store_dir(cfg, "query", model_pathway))

    if query_ids is None:
        from training.data.nirt import observations

        query_ids = sorted(observations(cfg, build=False)["query_id"].unique().tolist())
    ids = [q for q in query_ids if q in retr_store and q in model_store]

    retr_vecs = retr_store.gather(ids)
    groups = content_groups(cfg)      # exclude text-identical twins, not just the id
    out = np.zeros((len(ids), model_store.dim), dtype=np.float32)
    B = 4096
    for i in range(0, len(ids), B):
        chunk = ids[i : i + B]
        out[i : i + len(chunk)] = bank.neighbor_weighted_mean(
            retr_vecs[i : i + B], model_store, k=k, exclude_ids=chunk,
            exclude_groups=groups, weighting="uniform")[0]

    manifest = {
        "kind": "query_warmup",
        "pathway": model_pathway,
        "retrieval_pathway": retr_pw,
        "dim": int(model_store.dim),
        "id_field": "query_id",
        "count": len(ids),
        "k": k,
        "complete": True,
        "bank_split": bank.manifest.get("split", "train"),
        "self_exclusion": "content_hash" if groups else "query_id",
        "source": "mean irt-embedding of k nearest TRAIN queries (self-excluded)",
    }
    store = EmbeddingStore(ids, out, manifest, id_field="query_id")
    if save:
        store.save(warmup_store_dir(cfg, model_pathway))
    return store


def load_warmup(cfg: Config, pathway: Optional[str] = None) -> Optional[EmbeddingStore]:
    pathway = pathway or cfg.get("data.pathway", "irt")
    d = warmup_store_dir(cfg, pathway)
    return EmbeddingStore.load(d) if EmbeddingStore.exists(d) else None
