"""Run the deterministic (non-LLM) Phase 0 pipeline end to end.

    python -m router.phase0            # or: router-phase0

Stages implemented in this build:
  1. load + normalize all available sources (RouterBench, Arena, GPT-4 Judge)
  2. chance correction + quality checks
  3. write data/processed/{responses,queries,models}.parquet
  4. build + verify data/splits/*.json
  5. build LLM profiles -> data/processed/model_profiles.parquet
  6. build the NIRT observation table -> data/processed/nirt_observations.parquet
  7. (optional, --embeddings) encode queries + profiles per pathway (bert
     pathways scoped to NIRT queries; streaming + resumable)

Optional extra stages (needed by the Phase 1 relevance + warm-up path):
  --taxonomy    UMAP+HDBSCAN cluster queries -> centroids, r_q relevance vectors,
                readable taxonomy labels (data/taxonomy/, query_relevance__*)
  --query-bank  FAISS index over train queries + pre-computed warm-up reps
Expensive LLM cluster labeling is never run here.

Phase 1 consumes everything through `training.data.facade.load_training_data`.
"""

from __future__ import annotations

import sys

from .cli import base_parser, get_config, info
from .data.quality import check_responses
from .data.response_matrix import build_tables, write_tables
from .data.splits import check_leakage, make_splits, write_splits

_OPTIONAL_LLM = [
    "scripts/taxonomy/label_queries.py  (opt-in LLM cluster labelling, not run here)",
]


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--sources", default=None)
    parser.add_argument("--embeddings", action="store_true",
                        help="also encode all queries (slow on CPU)")
    parser.add_argument("--taxonomy", action="store_true",
                        help="cluster queries -> centroids + r_q relevance vectors + taxonomy")
    parser.add_argument("--query-bank", action="store_true",
                        help="build the FAISS train-query bank + warm-up representations")
    args = parser.parse_args(argv)
    cfg = get_config(args)
    sources = args.sources.split(",") if args.sources else None

    info("stage 1-3: load / normalize / correct / quality / write tables")
    tables = build_tables(cfg, sources)
    report = check_responses(tables["responses"])
    print(report.render())
    report.raise_for_errors()
    write_tables(tables, cfg)

    info("stage 4: splits")
    splits = make_splits(tables["responses"], cfg)
    problems = check_leakage(splits)
    if problems:
        raise SystemExit(f"split leakage: {problems}")
    write_splits(splits, cfg)

    info("stage 5: LLM profiles")
    from .models.profiles import build_model_profiles, write_model_profiles

    profiles = build_model_profiles(cfg, tables["responses"])
    write_model_profiles(profiles, cfg)
    info(f"  {len(profiles)} profiles ({int(profiles['has_curated'].sum())} curated)")

    info("stage 6: NIRT observation table")
    from .data.nirt import build_nirt_observations, write_nirt_observations

    nirt_obs = build_nirt_observations(cfg)
    write_nirt_observations(nirt_obs, cfg)
    info(f"  {len(nirt_obs):,} observations, splits "
         f"{nirt_obs['split'].value_counts().to_dict()}")

    if args.embeddings:
        info("stage 7: embeddings (streaming, resumable)")
        from router.embeddings import available_pathways, build_store, default_store_dir, load_encoder

        q = tables["queries"].sort_values("query_id").reset_index(drop=True)
        p = profiles.sort_values("model_id").reset_index(drop=True)
        nirt_qids = set(nirt_obs["query_id"])
        for pw in available_pathways(cfg):
            enc = load_encoder(cfg, pathway=pw)
            # bert pathways only need the NIRT queries (huge CPU saving)
            qsub = q[q["query_id"].isin(nirt_qids)] if enc.cfg.backend == "bert" else q
            qs = build_store(default_store_dir(cfg, "query", pw),
                             qsub["query_id"].tolist(), qsub["query"].tolist(),
                             enc, id_field="query_id",
                             manifest_extra={"query_set":
                                             "nirt" if enc.cfg.backend == "bert" else "all"})
            ps = build_store(default_store_dir(cfg, "model_profile", pw),
                             p["model_id"].tolist(), p["profile_text"].tolist(),
                             enc, id_field="model_id",
                             manifest_extra={"text_column": "profile_text"})
            info(f"  [{pw}] query {len(qs)}×{qs.dim}, profile {len(ps)}×{ps.dim}")

    if args.taxonomy:
        info("stage 8: query ability taxonomy (UMAP+HDBSCAN -> r_q)")
        from .taxonomy.clustering import build_clusters
        from .taxonomy.relevance import build_relevance
        from .taxonomy.taxonomy import build_taxonomy

        cl = build_clusters(cfg)
        info(f"  {cl.n_clusters} clusters, noise {cl.noise_fraction:.1%}")
        rel = build_relevance(cfg)
        info(f"  r_q: {len(rel)} queries x {rel.dim}")
        tax = build_taxonomy(cfg)
        info(f"  taxonomy.json: {tax['n_clusters']} labelled clusters")

    if args.query_bank:
        info("stage 9: FAISS query bank + warm-up representations")
        from .retrieval.query_bank import build_query_bank
        from .retrieval.warmup import build_warmup_representations

        bank = build_query_bank(cfg)
        info(f"  bank: {len(bank)} train queries")
        wu = build_warmup_representations(cfg)
        info(f"  warm-up: {len(wu)} queries x {wu.dim}")

    info("deterministic Phase 0 complete.")
    info("opt-in LLM stage (not run here): " + "; ".join(_OPTIONAL_LLM))
    return 0


if __name__ == "__main__":
    sys.exit(main())
