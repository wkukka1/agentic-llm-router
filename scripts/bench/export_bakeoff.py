"""Export the IRT-Router benchmark as one neutral bundle every router trains on.

Router bake-off step 1 (docs/router_benchmark_experiment.md). External routers
(llmrouter-lib, RouteLLM) run in their own venv and never import this repo, so
they read this bundle instead. Our own routers read the same splits through
``load_training_data``, so every router sees identical rows.

    python scripts/bench/export_bakeoff.py --config configs/irt_router.yaml

Writes ``data/processed/router_bakeoff/<name>/``:

* ``queries.parquet``      query_id, split, dataset, emb_id (row into embeddings.npy)
* ``embeddings.npy``       (n_queries, 768) BERT query embeddings (query__irt store)
* ``observations.parquet`` query_id, model_id, target (graded, in [0, 1]), cost (USD)
* ``models.parquet``       model_id, train_mean_cost, profile embedding (model_profile__irt)
* ``manifest.json``        provenance and counts

``--holdout-dataset`` (leave-one-dataset-out, step 3) drops that dataset from
train/validation and makes its test rows the only evaluation split.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from router.config import load_config
from training.data.facade import load_training_data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/irt_router.yaml")
    ap.add_argument("--pathway", default="irt", help="query/profile embedding store")
    ap.add_argument("--name", default="primary")
    ap.add_argument("--holdout-dataset", default=None)
    ap.add_argument("--out-root", default="data/processed/router_bakeoff")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = load_training_data(cfg)
    obs = d.nirt_observations()[["query_id", "model_id", "target", "cost", "split"]].copy()
    obs["query_id"] = obs["query_id"].astype(str)
    obs["model_id"] = obs["model_id"].astype(str)
    obs["dataset"] = obs["query_id"].str.split(":").str[1]

    if args.holdout_dataset:
        h = args.holdout_dataset
        if h not in set(obs["dataset"]):
            raise SystemExit(f"unknown dataset {h!r}")
        obs = obs[(obs["dataset"] != h) | (obs["split"] == "test")]
        obs = obs[(obs["dataset"] == h) | obs["split"].isin(["train", "validation"])]
        obs.loc[obs["dataset"] == h, "split"] = "heldout"

    store = d.query_embeddings(args.pathway)
    qmeta = obs.drop_duplicates("query_id")[["query_id", "split", "dataset"]].reset_index(drop=True)
    missing = [q for q in qmeta["query_id"] if q not in store]
    if missing:
        raise SystemExit(f"{len(missing)} queries have no embedding, e.g. {missing[:3]}")
    emb = np.stack([np.asarray(store.get(q), dtype=np.float32) for q in qmeta["query_id"]])
    qmeta["emb_id"] = np.arange(len(qmeta))

    train = obs[obs["split"] == "train"]
    costs = train.groupby("model_id")["cost"].mean()
    prof = d.profile_embeddings(args.pathway)
    models = pd.DataFrame({"model_id": sorted(costs.index)})
    models["train_mean_cost"] = models["model_id"].map(costs)
    models["profile"] = [np.asarray(prof.get(m), dtype=np.float32).tolist() for m in models["model_id"]]

    out = cfg.resolve(args.out_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    qmeta.to_parquet(out / "queries.parquet", index=False)
    np.save(out / "embeddings.npy", emb)
    obs[["query_id", "model_id", "target", "cost"]].to_parquet(out / "observations.parquet", index=False)
    models.to_parquet(out / "models.parquet", index=False)
    manifest = {
        "config": args.config, "pathway": args.pathway, "holdout_dataset": args.holdout_dataset,
        "queries_per_split": qmeta["split"].value_counts().to_dict(),
        "n_models": len(models), "embedding_dim": int(emb.shape[1]),
        "graded_share": float(((obs["target"] > 0) & (obs["target"] < 1)).mean()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
