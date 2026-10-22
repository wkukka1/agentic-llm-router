"""Build the Elo-anchored model manifold and save it as a profile-embedding store.

Features per pool model (``configs/model_manifold.yaml``): pinned Arena
category Elos (``training.data.public_benchmarks.fetch_arena_category_elo``)
plus normalized Open LLM Leaderboard scores. ``training.models.manifold``
fits a K-dim factor model whose axis 0 is standardized Arena Elo; the
coordinates ``z_m`` are written to ``model_profile__<pathway>`` so NIRT's
``projected`` mode reads them like any profile embedding.

    python scripts/embeddings/build_manifold_embeddings.py --config configs/irt_router.yaml --k 3
    python scripts/embeddings/build_manifold_embeddings.py --config configs/irt_router.yaml --k 3 \
        --drop-overlapping            # leakage ablation -> pathway manifold_k3_noovl
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import yaml

from router.embeddings import EmbeddingStore, default_store_dir
from training.cli import base_parser, get_config, info
from training.data.public_benchmarks import fetch_arena_category_elo, fetch_hf_leaderboard_table
from training.models.manifold import build_feature_matrix, fit_manifold


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--manifold", default="configs/model_manifold.yaml")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ridge", type=float, default=1.0,
                    help="ridge on z and W in std units (1.0 ~ a unit-variance prior on z; "
                         "smaller over-extrapolates Elo for Arena-unrated models)")
    ap.add_argument("--drop-overlapping", action="store_true",
                    help="drop benchmark columns that overlap a suite dataset")
    ap.add_argument("--pathway", default=None,
                    help="store name (default manifold_k<K>[_noovl])")
    args = ap.parse_args()
    cfg = get_config(args)

    mcfg = yaml.safe_load(cfg.resolve(args.manifold).read_text(encoding="utf-8"))
    arena = mcfg["arena"]
    elo_path = fetch_arena_category_elo(cfg, snapshot_date=arena["snapshot_date"],
                                        filename=arena["filename"], revision=arena["revision"])
    olb_rows = json.loads(fetch_hf_leaderboard_table(cfg).read_text(encoding="utf-8"))
    ids, X, columns = build_feature_matrix(mcfg, pd.read_csv(elo_path), olb_rows,
                                           drop_overlapping=args.drop_overlapping)
    man = fit_manifold(X, args.k, columns=columns, ridge=args.ridge)

    pathway = args.pathway or f"manifold_k{args.k}" + ("_noovl" if args.drop_overlapping else "")
    observed = {m: int((~np.isnan(X[i])).sum()) for i, m in enumerate(ids)}
    manifest = {
        "id_field": "model_id", "dim": int(args.k), "count": len(ids),
        "normalized": False, "complete": True, "pathway": pathway,
        "source": "training.models.manifold (Elo-anchored factor model)",
        "arena_snapshot": arena["snapshot_date"], "columns": man["columns"],
        "loadings": np.round(man["W"], 6).tolist(),
        "feature_mean": man["mean"].tolist(), "feature_std": man["std"].tolist(),
        "explained": man["explained"], "rmse": man["rmse"], "ridge": man["ridge"],
        "observed_features": observed, "drop_overlapping": bool(args.drop_overlapping),
    }
    out = default_store_dir(cfg, "model_profile", pathway)
    EmbeddingStore(ids, np.ascontiguousarray(man["Z"], dtype=np.float32), manifest,
                   id_field="model_id").save(out)

    info(f"wrote {out}  ({len(ids)} x {args.k}); explained {man['explained']:.3f}, "
         f"rmse {man['rmse']:.3f} (std units)")
    tbl = pd.DataFrame(man["Z"], index=ids, columns=[f"z{j}" for j in range(args.k)])
    tbl["n_obs"] = pd.Series(observed)
    print(tbl.round(3).sort_values("z0", ascending=False).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
