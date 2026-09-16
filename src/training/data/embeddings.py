"""Build the query :class:`~router.embeddings.EmbeddingStore` for a pathway.

``router.embeddings`` owns the *encoder* -- deterministic text-to-vector, no
ground truth, shared with the live serving path
(:func:`router.routing.routers._encode_texts`). Everything here *decides what
to embed* (which queries, restricted by split / NIRT membership / a smoke
limit) from :class:`~training.data.facade.TrainingData`, which a live router
never needs -- so it lives on the training side of the boundary, same
reasoning as :func:`training.data.nirt.build_nirt_observations`.

Builds one pathway per call, same shape as
:func:`training.retrieval.knn_impute.build_knn_imputed_store` -- looping over
multiple pathways is the caller's job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from router.embeddings import (
    EmbeddingStore,
    available_pathways,
    build_store,
    default_store_dir,
    load_encoder,
)

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from .facade import TrainingData


def build_query_embedding_store(
    data: "TrainingData",
    pathway: Optional[str] = None,
    *,
    split: Optional[str] = None,
    from_nirt: bool = False,
    limit: Optional[int] = None,
    restart: bool = False,
    flush_every: int = 25,
) -> EmbeddingStore:
    """Encode ``data.queries`` (optionally filtered) into a query store.

    ``split`` restricts to one of ``data.splits``' query-id sets; ``from_nirt``
    restricts to queries present in ``data.nirt_observations()`` (raises if
    that hasn't been built yet); ``limit`` takes the first N (by ``query_id``,
    after filtering) for a smoke run. Output directory follows
    :func:`router.embeddings.default_store_dir`, suffixed ``__<split>`` or
    ``__smoke`` exactly as the CLI script did.
    """
    cfg = data.cfg
    queries = data.queries

    if split:
        entry = data.splits.get(split)
        if entry is None:
            raise KeyError(f"split {split!r} not built (see scripts/data/build_splits.py)")
        ids = set(entry["query_ids"])
        queries = queries[queries["query_id"].isin(ids)]
    if from_nirt:
        keep = set(data.nirt_observations()["query_id"])
        queries = queries[queries["query_id"].isin(keep)]
    queries = queries.sort_values("query_id")
    if limit:
        queries = queries.head(limit)
    queries = queries.reset_index(drop=True)

    pw = pathway or cfg.get("embedding.default_pathway") or available_pathways(cfg)[0]
    encoder = load_encoder(cfg, pathway=pw)
    # a smoke run never shares a directory with a full store (it would overwrite it)
    suffix = (f"__{split}" if split else "") + ("__smoke" if limit else "")
    out_dir = default_store_dir(cfg, "query", pw + suffix)
    return build_store(
        out_dir, queries["query_id"].tolist(), queries["query"].tolist(),
        encoder, id_field="query_id", resume=not restart, force=restart,
        flush_every=flush_every,
        manifest_extra={"query_set": "nirt" if from_nirt else (split or "all")},
    )
