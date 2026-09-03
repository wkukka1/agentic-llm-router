"""Train / validation / test prediction-metric gap for NIRT runs (overfitting check).

    python scripts/nirt/train_test_gap.py
    python scripts/nirt/train_test_gap.py nirt-knn10w-2d-projected nirt-2d-projected

Loads each checkpoint, scores it on all three splits, and prints BCE / Brier /
AUC per split plus the test-minus-train gap. Every run is restricted to the
0-shot observation set (RouterBench ':5shot' items) so `n` matches across runs
whose query store only covers the 0-shot queries (the kNN-imputed stores).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from router.config import load_config
from router.data.phase1 import load_phase1
from router.nirt.evaluate import predict_dataset
from router.nirt.metrics import marginal_baselines, prediction_metrics
from router.nirt.train import load_run

DEFAULT_RUNS = [
    "nirt-2d-projected", "irt-25d-projected",
    "nirt-knn1w-2d-projected", "nirt-knn10w-2d-projected", "nirt-knn25w-2d-projected",
]
SPLITS = ["train", "validation", "test"]


def main() -> int:
    runs = sys.argv[1:] or DEFAULT_RUNS
    d = load_phase1(load_config())
    rows = []
    for run in runs:
        model, cfg, midx = load_run(run)
        dcfg = cfg.get("data", {}) or {}
        pw, qp = dcfg.get("pathway", "irt"), (dcfg.get("query_pathway") or None)
        train_ds = d.nirt_dataset(split="train", pathway=pw, query_pathway=qp)
        for sp in SPLITS:
            ds = d.nirt_dataset(split=sp, pathway=pw, query_pathway=qp)
            y, p = predict_dataset(model, ds, midx)
            keep = ~pd.Series(ds.query_ids).astype(str).str.endswith(":5shot").to_numpy()
            y, p, mids = y[keep], p[keep], np.asarray(ds.model_ids)[keep]
            m = prediction_metrics(y, p)
            base = marginal_baselines(train_ds.targets, train_ds.model_ids, y, mids)
            rows.append({
                "run": run, "split": sp, "n": m["n"], "bce": m["bce"],
                "brier": m["brier"], "auc": m["auc"], "acc@0.5": m["acc@0.5"],
                "spearman": m["spearman_r"],
                "d_bce_vs_model_mean": m["bce"] - base["model_mean"]["bce"],
            })

    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.float_format", lambda x: f"{x:.4f}"):
        print(df.to_string(index=False))
        piv = df.pivot(index="run", columns="split", values="bce")
        piv["gap_test_train"] = piv["test"] - piv["train"]
        piv["gap_val_train"] = piv["validation"] - piv["train"]
        print("\n=== BCE: test/val minus train (larger gap = more overfitting) ===")
        print(piv[["train", "validation", "test", "gap_test_train", "gap_val_train"]].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
