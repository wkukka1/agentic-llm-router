"""Go/no-go for the Elo-manifold workstream: how much of each pool model's mean
quality (train split) does its public Arena Elo explain?

Per task family (``profiles.task_families`` of the phase-0 config) and overall,
across the Arena-rated pool models: R^2 and Spearman of mean ``score`` on
(a) overall Arena Elo and (b) the family's matching Arena category Elo.

    python scripts/nirt/elo_diagnostic.py --config configs/irt_router.yaml
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from training.cli import base_parser, get_config
from training.data.public_benchmarks import fetch_arena_category_elo

# task family -> the Arena category that matches it best
FAMILY_CATEGORY = {"math": "math", "code": "coding", "knowledge": "full",
                   "reasoning": "hard_6", "language_zh": "chinese"}


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--manifold", default="configs/model_manifold.yaml")
    ap.add_argument("--out", default="investigations/elo_manifold/diagnostic.csv")
    args = ap.parse_args()
    cfg = get_config(args)

    mcfg = yaml.safe_load(cfg.resolve(args.manifold).read_text(encoding="utf-8"))
    arena = mcfg["arena"]
    elo = pd.read_csv(fetch_arena_category_elo(cfg, snapshot_date=arena["snapshot_date"],
                                               filename=arena["filename"], revision=arena["revision"]))
    elo = elo.pivot_table(index="arena_key", columns="category", values="rating")
    key = {}
    for mid, entry in (mcfg.get("models") or {}).items():
        entry = entry or {}
        target = mcfg["models"].get(entry["alias_of"]) if entry.get("alias_of") else entry
        if (target or {}).get("arena_key") in elo.index:
            key[mid] = target["arena_key"]
    models = sorted(key)

    train = set(json.loads((cfg.path("splits") / "train.json").read_text())["query_ids"])
    r = pd.read_parquet(cfg.path("processed") / "responses.parquet",
                        columns=["query_id", "model_id", "score", "dataset"])
    r = r[r["query_id"].isin(train)]

    fams = [("overall", None)] + list(cfg.to_dict().get("profiles", {}).get("task_families", {}).items())
    rows = []
    for fam, datasets in fams:
        sub = r if datasets is None else r[r["dataset"].isin(datasets)]
        if sub.empty:
            continue
        perf = sub.groupby("model_id")["score"].mean().reindex(models)
        for cat in dict.fromkeys(["full", FAMILY_CATEGORY.get(fam, "full")]):
            x = np.array([elo.at[key[m], cat] for m in models])
            y = perf.to_numpy()
            rows.append({"family": fam, "elo_category": cat, "n_models": len(models),
                         "n_queries": sub["query_id"].nunique(),
                         "r2": float(np.corrcoef(x, y)[0, 1] ** 2),
                         "spearman": float(spearmanr(x, y).correlation)})
    df = pd.DataFrame(rows)
    out = cfg.resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
