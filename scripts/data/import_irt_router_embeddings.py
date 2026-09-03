"""Import the IRT-Router repo's precomputed BERT embeddings + relevance vectors.

The paper ships (Git LFS, fetched by ``download_irt_router.py``):

* ``utils/bert_embeddings/query_embeddings.pkl``  -- ``[{index, embedding}]``,
  bert-base-uncased mean-pooled, L2-normalized, 768-d, keyed by the query index
  in ``utils/map/query.csv`` (columns ``index, task, split, id, question``).
* ``utils/bert_embeddings/llm_embeddings.pkl``    -- same, keyed by the model
  index in ``utils/map/llm.csv`` (``index, name, profile``).
* ``utils/relevance/relevance_vectors_cluster_{train,test}_bert.pkl`` --
  ``[{index, relevance_vector}]``, 25-d, keyed by query index.

This script rekeys them onto our ``query_id`` / ``model_id`` and writes native
:class:`~router.embeddings.EmbeddingStore`s under
``embedding.cache_dir`` (``query__irt``, ``model_profile__irt``) and the
relevance store (``query_relevance__irt``), so the rest of the pipeline runs
against ``pathway="irt"`` unchanged.

    python scripts/data/import_irt_router_embeddings.py --config configs/irt_router.yaml
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from router.cli import base_parser, get_config, info
from router.data.model_registry import canonical_model_id
from router.data.normalize import make_query_id, render_prompt
from router.data.schemas import Source
from router.embeddings import EmbeddingStore, default_store_dir
from router.taxonomy.relevance import relevance_store_dir


def _load_pkl_records(path: Path, value_key: str) -> dict[int, np.ndarray]:
    """``[{index, <value_key>}, ...]`` (or a dict) -> ``{index: vector}``."""
    with open(path, "rb") as fh:
        raw = pickle.load(fh)
    out: dict[int, np.ndarray] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[int(k)] = np.asarray(v, dtype=np.float32)
        return out
    for rec in raw:
        out[int(rec["index"])] = np.asarray(rec[value_key], dtype=np.float32)
    return out


def _save_store(directory: Path, ids: list[str], matrix: np.ndarray,
                id_field: str, extra: dict) -> None:
    manifest = {
        "id_field": id_field,
        "dim": int(matrix.shape[1]),
        "count": len(ids),
        "normalized": True,
        "complete": True,
        "source": "imported verbatim from github.com/Mercidaiha/IRT-Router",
        **extra,
    }
    EmbeddingStore(ids, np.ascontiguousarray(matrix, dtype=np.float32),
                   manifest, id_field=id_field).save(directory)
    info(f"wrote {directory}  ({len(ids)} x {matrix.shape[1]})")


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = get_config(args)

    src = cfg.get("sources.irt_router")
    root = cfg.resolve(src.get("local_dir", "data/raw/irt_router"))
    pool = {str(m) for m in (src.get("pool") or [])}

    query_map = pd.read_csv(root / "utils/map/query.csv")   # index, task, split, id, question
    llm_map = pd.read_csv(root / "utils/map/llm.csv")       # index, name, profile

    # -- queries -------------------------------------------------------------
    qid_of_index: dict[int, str] = {
        int(r.index): make_query_id(Source.IRT_ROUTER, str(r.task), render_prompt(r.question))
        for r in query_map.itertuples()
    }
    q_vecs = _load_pkl_records(root / "utils/bert_embeddings/query_embeddings.pkl", "embedding")

    seen: dict[str, np.ndarray] = {}
    for idx, vec in q_vecs.items():
        qid = qid_of_index.get(idx)
        if qid is not None and qid not in seen:
            seen[qid] = vec
    q_ids = sorted(seen)
    _save_store(default_store_dir(cfg, "query", "irt"), q_ids,
                np.stack([seen[q] for q in q_ids]), "query_id",
                {"pathway": "irt", "backend": "bert", "model_name": "bert-base-uncased",
                 "pooling": "mean"})

    # -- model profiles ----------------------------------------------------
    name_of_index = {int(r.index): str(r.name) for r in llm_map.itertuples()}
    m_vecs = _load_pkl_records(root / "utils/bert_embeddings/llm_embeddings.pkl", "embedding")
    model_rows: dict[str, np.ndarray] = {}
    for idx, vec in m_vecs.items():
        native = name_of_index.get(idx)
        if native is None or (pool and native not in pool):
            continue
        model_rows[canonical_model_id(native)] = vec
    m_ids = sorted(model_rows)
    _save_store(default_store_dir(cfg, "model_profile", "irt"), m_ids,
                np.stack([model_rows[m] for m in m_ids]), "model_id",
                {"pathway": "irt", "backend": "bert", "model_name": "bert-base-uncased",
                 "pooling": "mean"})

    # -- relevance vectors (r_q) ----------------------------------------
    rel_rows: dict[str, np.ndarray] = {}
    for tag in ("train", "test"):
        p = root / f"utils/relevance/relevance_vectors_cluster_{tag}_bert.pkl"
        if not p.exists():
            continue
        for idx, vec in _load_pkl_records(p, "relevance_vector").items():
            qid = qid_of_index.get(idx)
            if qid is not None:
                rel_rows.setdefault(qid, vec)
    if rel_rows:
        r_ids = sorted(rel_rows)
        R = np.stack([rel_rows[q] for q in r_ids])
        _save_store(relevance_store_dir(cfg, "irt"), r_ids, R, "query_id",
                    {"kind": "query_relevance", "pathway": "irt",
                     "source": "softmax(cos(e_q, centroid)/tau), imported from IRT-Router"})
    else:
        info("no relevance pkls found -- skipping (build_arrays falls back to uniform r_q)")

    info("import complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
