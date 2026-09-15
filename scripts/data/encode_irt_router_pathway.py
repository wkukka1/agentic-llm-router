"""Encode the IRT-Router queries + LLM profiles with one of our own encoder
pathways (P4a of the capacity workstream).

The default IRT-Router pipeline *imports* the paper's shipped bert-base-uncased
vectors (``import_irt_router_embeddings.py``). This script instead **encodes**
``utils/map/query.csv`` (question text) and ``utils/map/llm.csv`` (profile text)
with a pathway from ``configs/irt_router.yaml -> embedding.pathways`` (e.g.
``mpnet``), keyed onto the same ``query_id`` / ``model_id`` so
``data.pathway: mpnet`` just works.

    python scripts/data/encode_irt_router_pathway.py --config configs/irt_router.yaml --pathway mpnet
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from training.cli import base_parser, get_config, info
from training.data.model_registry import canonical_model_id
from training.data.normalize import make_query_id, render_prompt
from training.data.schemas import Source
from router.embeddings import EmbeddingStore, default_store_dir, load_encoder


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--pathway", required=True, help="pathway name from embedding.pathways")
    args = ap.parse_args()
    cfg = get_config(args)

    src = cfg.get("sources.irt_router")
    root = cfg.resolve(src.get("local_dir", "data/raw/irt_router"))
    pool = {str(m) for m in (src.get("pool") or [])}
    enc = load_encoder(cfg, pathway=args.pathway)

    query_map = pd.read_csv(root / "utils/map/query.csv")   # index, task, split, id, question
    llm_map = pd.read_csv(root / "utils/map/llm.csv")       # index, name, profile

    # -- queries: one row per distinct query_id ----------------------------
    q_text: dict[str, str] = {}
    for r in query_map.itertuples():
        qid = make_query_id(Source.IRT_ROUTER, str(r.task), render_prompt(r.question))
        q_text.setdefault(qid, str(r.question))
    q_ids = sorted(q_text)
    info(f"[{args.pathway}] encoding {len(q_ids)} IRT-Router queries with "
         f"{enc.cfg.backend}:{enc.cfg.model_name}")
    q_mat = enc.encode([q_text[q] for q in q_ids], show_progress=True)
    EmbeddingStore(q_ids, q_mat, {
        "id_field": "query_id", "dim": int(q_mat.shape[1]), "count": len(q_ids),
        "normalized": enc.cfg.normalize, "complete": True, "pathway": args.pathway,
        "backend": enc.cfg.backend, "model_name": enc.cfg.model_name,
        "source": "encoded from IRT-Router utils/map/query.csv",
    }, id_field="query_id").save(default_store_dir(cfg, "query", args.pathway))
    info(f"[{args.pathway}] wrote query store ({len(q_ids)} x {q_mat.shape[1]})")

    # -- model profiles --------------------------------------------------
    prof: dict[str, str] = {}
    for r in llm_map.itertuples():
        native = str(r.name)
        if pool and native not in pool:
            continue
        prof[canonical_model_id(native)] = str(r.profile)
    m_ids = sorted(prof)
    m_mat = enc.encode([prof[m] for m in m_ids], show_progress=True)
    EmbeddingStore(m_ids, m_mat, {
        "id_field": "model_id", "dim": int(m_mat.shape[1]), "count": len(m_ids),
        "normalized": enc.cfg.normalize, "complete": True, "pathway": args.pathway,
        "backend": enc.cfg.backend, "model_name": enc.cfg.model_name,
        "source": "encoded from IRT-Router utils/map/llm.csv",
    }, id_field="model_id").save(default_store_dir(cfg, "model_profile", args.pathway))
    info(f"[{args.pathway}] wrote profile store ({len(m_ids)} x {m_mat.shape[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
