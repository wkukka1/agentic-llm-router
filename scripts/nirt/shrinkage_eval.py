"""Evaluate shrinkage-to-`model_mean` calibration (capacity workstream item 4).

Blends a NIRT run's raw prediction toward the per-model training rate,
proportional to how novel each eval query's embedding is relative to training
(`router.nirt.shrinkage`). Motivation: the P1 diagnostic found every learned
predictor is *worse than `model_mean`* on IRT-Router's OOD split. Reports
prediction + routing metrics for the raw prediction, `model_mean` alone, and a
`midpoint` sweep of the blend, on any split(s).

    python scripts/nirt/shrinkage_eval.py --config configs/irt_router.yaml \
        --nirt-run irtrouter-nirt-2d-projected --splits test,ood

Writes artifacts/<...>/shrinkage_eval.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from training.cli import float_table, raw_parser, write_json, zeroshot_only
from router.config import load_config
from router.nirt.checkpoint import load_run
from router.nirt.predict import predict_matrix
from training.data.facade import load_training_data
from training.nirt.metrics import prediction_metrics
from evaluation.nirt.routing import align, eval_matrices, train_quality
from evaluation.nirt.evaluate import ranking_metrics
from evaluation.routing.oracle import routing_evaluation
from router.nirt.shrinkage import novelty_weight, shrink_predictions


def _metrics_row(tag: str, pred: np.ndarray, true: np.ndarray, cost: np.ndarray,
                 model_ids: list[str]) -> dict:
    ok = np.isfinite(pred).all(axis=1)
    t, p, c = true[ok], pred[ok], cost[ok]
    pm = prediction_metrics(t.ravel(), p.ravel())
    rm = ranking_metrics(p, t)
    re = routing_evaluation(p, t, c, model_ids, lam=0.0)
    return {"variant": tag, "n": int(ok.sum()), "bce": pm["bce"], "auc": pm["auc"],
           "regret": rm["regret"], "optimal_rate": rm["optimal_rate"],
           "oracle_hit_any_best": re["oracle_hit_rate_any_best"]}


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--nirt-run", required=True)
    ap.add_argument("--splits", default="ood", help="comma list; each scored against the same train pathway")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--midpoints", default="0.15,0.25,0.35,0.45,0.55")
    ap.add_argument("--temp", type=float, default=0.08)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = load_training_data(cfg)
    model, rcfg, midx = load_run(args.nirt_run)
    dcfg = rcfg.get("data", {}) or {}
    pathway = dcfg.get("pathway", "irt")
    query_pathway = dcfg.get("query_pathway") or None

    train_q = d.query_embeddings(query_pathway or pathway)
    train_obs = d.nirt_observations()
    train_qids = sorted(set(train_obs.loc[train_obs["split"] == "train", "query_id"]) & set(train_q._index))
    train_emb = train_q.gather(train_qids)
    model_ids_full = sorted(midx, key=midx.get)
    mean_rate_map = train_quality(d, model_ids_full, metric="quality")

    out_all: dict = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        true_df, cost_df = eval_matrices(d, split=split)
        true_df, cost_df = zeroshot_only(true_df, cost_df)
        model_ids = list(true_df.columns)
        true, cost = true_df.to_numpy(np.float64), cost_df.to_numpy(np.float64)
        mean_rate = np.array([mean_rate_map.get(m, float(np.nanmean(true))) for m in model_ids])

        pred = align(predict_matrix(model, midx, d, split, pathway=pathway,
                                    query_pathway=query_pathway), true_df)
        eval_q = d.query_embeddings(query_pathway or pathway)
        eval_emb = eval_q.gather([q for q in true_df.index])

        rows = [_metrics_row("raw", pred, true, cost, model_ids),
               _metrics_row("model_mean", np.tile(mean_rate, (len(true_df), 1)), true, cost, model_ids)]
        for mp in [float(x) for x in args.midpoints.split(",")]:
            w = novelty_weight(eval_emb, train_emb, k=args.k, midpoint=mp, temp=args.temp)
            blended = shrink_predictions(pred, w, mean_rate)
            row = _metrics_row(f"shrink(mid={mp:g})", blended, true, cost, model_ids)
            row["mean_weight"] = float(w.mean())
            rows.append(row)

        df = pd.DataFrame(rows)
        with float_table():
            print(f"\n=== shrinkage eval -- {args.nirt_run} -- {split} split "
                 f"({len(true_df)} queries) ===")
            print(df.to_string(index=False))
        out_all[split] = df.to_dict(orient="records")

    is_irt = args.config and "irt_router" in str(args.config)
    out = Path(args.out) if args.out else Path(
        "artifacts/irt_router/shrinkage_eval.json" if is_irt else "artifacts/phase2/shrinkage_eval.json")
    write_json(out, {"nirt_run": args.nirt_run, "k": args.k, "temp": args.temp, "splits": out_all}, root=cfg.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
