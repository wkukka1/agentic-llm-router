"""FAISS query bank (deferred Phase 0 stage, built for Phase 1).

A nearest-neighbour index over the **training** query embeddings only. Phase 1's
warm-up representation looks an unseen query up here and blends its embedding
with the mean of its k nearest training-query embeddings.

The index is a Phase-0 artefact: built once by
``scripts/retrieval/build_query_bank.py`` and never rebuilt or extended during
model training (that would leak validation/test queries into the bank).

Layout (under ``indexes/query_bank/``): ``index.faiss``, ``ids.parquet``,
``manifest.json``.
"""

from .knn_impute import (
    build_knn_imputed_store,
    imputed_pathway_name,
    imputed_store_dir,
    load_or_build_imputed_store,
)
from .query_bank import QueryBank, build_query_bank, query_bank_dir
from .warmup import build_warmup_representations, load_warmup, warmup_store_dir

__all__ = [
    "QueryBank",
    "build_query_bank",
    "query_bank_dir",
    "build_warmup_representations",
    "load_warmup",
    "warmup_store_dir",
    "build_knn_imputed_store",
    "load_or_build_imputed_store",
    "imputed_pathway_name",
    "imputed_store_dir",
]
