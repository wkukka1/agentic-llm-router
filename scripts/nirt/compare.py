"""Compare a trained NIRT / IRT-Router run against baselines, in-distribution or OOD.

    python scripts/nirt/compare.py --run irt-2d-projected --split test
    python scripts/nirt/compare.py --run irt-2d-ood --ood            # held-out benchmark families

Four tables:
  1. prediction quality  -- NIRT vs classical-IRT (transductive ceiling + theta=0 lower
     bound) vs KNN-router / MLP-router vs marginal baselines
  2. ranking quality     -- per-query optimal-rate / regret / Spearman
  3. routing + cost      -- achieved accuracy, USD/query, cost savings, Reward(alpha)
  4. AIQ                  -- area under the cost/quality curve, per learned router

`--ood` holds `evaluation.ood_holdout_families` out of ALL training (the run must have been
trained with `train_nirt.py --ood`) and evaluates only on those families. Writes
<runs_dir>/<run>/comparison_<label>.json.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import yaml

from training.cli import float_table, raw_parser, resolve, write_json
from router.config import load_config
from router.nirt.baselines_infer import knn_router_matrix, mlp_router_matrix
from router.nirt.checkpoint import load_run
from router.nirt.predict import predict_matrix_from_dataset
from training.data.facade import load_training_data
from training.nirt.metrics import marginal_baselines, prediction_metrics
from training.trainers.mlp_router import fit_mlp_router
from evaluation.baselines.classical_irt import fit_classical_irt
from evaluation.nirt.evaluate import ranking_metrics
from evaluation.nirt.ood import ood_families, split_observations
from evaluation.nirt.routing import (
    add_reward_columns,
    aiq,
    align,
    dense_matrices,
    eval_matrices,
    pareto,
    routing_report,
    train_quality,
)


def _round(o, n=4):
    if isinstance(o, dict):
        return {k: _round(v, n) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, n) for v in o]
    return round(o, n) if isinstance(o, float) else o


def _keep(m: dict) -> dict:
    return {k: m[k] for k in ("bce", "mse", "auc", "acc@0.5", "spearman_r") if k in m}


def _pred_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    """``prediction_metrics`` over the finite entries only -- a kNN-imputed /
    KNN-router matrix may not cover every eval query (e.g. RouterBench ':5shot')."""
    t, p = np.asarray(true, float).ravel(), np.asarray(pred, float).ravel()
    ok = np.isfinite(t) & np.isfinite(p)
    return prediction_metrics(t[ok], p[ok])


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--ood", action="store_true", help="evaluate on held-out benchmark families")
    ap.add_argument("--config", default="configs/nirt.yaml")
    ap.add_argument("--phase0-config", default=None)
    ap.add_argument("--irt-dim", type=int, default=None)
    ap.add_argument("--reference", default=None)
    args = ap.parse_args()

    nirt_cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    runs_dir = resolve(nirt_cfg.get("runs_dir", "data/processed/nirt_runs"))

    p0 = load_config(args.phase0_config) if args.phase0_config else load_config()
    d = load_training_data(p0)
    model, run_cfg, model_index = load_run(args.run, runs_dir=runs_dir)
    pathway = run_cfg.get("data", {}).get("pathway", "irt")
    query_pathway = run_cfg.get("data", {}).get("query_pathway") or None
    seed = int(nirt_cfg.get("seed", 42))

    irt_cfg = nirt_cfg.get("classical_irt", {}) or {}
    routing_cfg = nirt_cfg.get("routing", {}) or {}
    bl_cfg = nirt_cfg.get("baselines", {}) or {}
    irt_dim = args.irt_dim or irt_cfg.get("dim") or int(run_cfg.get("model", {}).get("dim", 1))
    reference = args.reference or routing_cfg.get("reference_model", "gpt-4-1106-preview")
    lams = tuple(routing_cfg.get("lambda_sweep", (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)))
    alphas = tuple(routing_cfg.get("reward_alphas", (0.5, 0.7, 0.8, 0.9)))

    # -- eval context: matrices + which queries + baseline training obs ----
    from router.data.nirt import NIRTDataset

    q_store, p_store = d.query_embeddings(pathway), d.profile_embeddings(pathway)
    # the NIRT run may consume a kNN-imputed query store (data.query_pathway);
    # the classical-IRT / KNN / MLP baselines always use the raw `pathway`.
    nirt_q_store = q_store
    if query_pathway:
        nirt_q_store = d.query_embeddings(query_pathway)
        if nirt_q_store is None:
            raise FileNotFoundError(
                f"run '{args.run}' was trained with data.query_pathway='{query_pathway}' "
                f"but that query store is missing; rebuild it with "
                f"scripts/nirt/knn_impute_sweep.py or training.retrieval.knn_impute"
            )
    if args.ood:
        fams = ood_families(nirt_cfg)
        label = f"ood[{'+'.join(fams)}]"
        bl_train_obs, ood_obs = split_observations(d, fams)
        eval_ds = NIRTDataset(ood_obs, nirt_q_store, p_store)
        true_df, cost_df = dense_matrices(ood_obs)
        cls_query_ids = list(true_df.index)
    else:
        label = args.split
        bl_train_obs = None                      # baselines use the standard train split
        eval_ds = d.nirt_dataset(split=args.split, pathway=pathway, query_pathway=query_pathway)
        true_df, cost_df = eval_matrices(d, split=args.split)
        cls_query_ids = None

    model_ids = list(true_df.columns)
    true = true_df.to_numpy(np.float64)
    cost = cost_df.to_numpy(np.float64)
    pool_costs = cost.mean(0)
    pool_qual = true_df.mean(0).to_numpy()
    n_q = len(true)

    # -- predictors ---------------------------------------------------------
    nirt_pred = align(predict_matrix_from_dataset(model, model_index, eval_ds), true_df)

    irt_kw = dict(dim=irt_dim, epochs=int(irt_cfg.get("epochs", 500)),
                  lr=float(irt_cfg.get("lr", 0.05)),
                  weight_decay=float(irt_cfg.get("weight_decay", 1e-3)), seed=seed)
    irt_cell = fit_classical_irt(d, protocol="cell", query_ids=cls_query_ids,
                                 split=args.split,
                                 holdout_frac=float(irt_cfg.get("holdout_frac", 0.15)), **irt_kw)
    irt_pred = align(pd.DataFrame(irt_cell.pred_matrix, index=irt_cell.query_ids,
                                  columns=irt_cell.model_ids), true_df)

    knn = bl_cfg.get("knn_router", {}) or {}
    mlp = bl_cfg.get("mlp_router", {}) or {}
    knn_pred = align(knn_router_matrix(d, list(true_df.index), k=int(knn.get("k", 5)),
                                       pathway=knn.get("pathway", "retrieval"),
                                       train_obs=bl_train_obs), true_df)
    mlp_model, mlp_ids = fit_mlp_router(d, hidden=int(mlp.get("hidden", 128)),
                                        epochs=int(mlp.get("epochs", 40)),
                                        lr=float(mlp.get("lr", 1e-3)),
                                        pathway=mlp.get("pathway", "irt"),
                                        seed=seed, train_obs=bl_train_obs)
    mlp_pred = align(mlp_router_matrix(mlp_model, mlp_ids, d, list(true_df.index),
                                       pathway=mlp.get("pathway", "irt")), true_df)

    learned = {
        "NIRT": nirt_pred,
        f"classical IRT {irt_dim}D (ceiling)": irt_pred,
        f"KNN-router (k={knn.get('k', 5)})": knn_pred,
        "MLP-router": mlp_pred,
    }

    # ============ 1. prediction ============
    train_ds = d.nirt_dataset(split="train", pathway=pathway)
    base = marginal_baselines(train_ds.targets, train_ds.model_ids,
                              true.ravel(), np.repeat([model_ids], n_q, axis=0).ravel())
    pred_rows = [{"method": "NIRT", **_keep(_pred_metrics(true, nirt_pred))},
                 {"method": f"classical IRT {irt_dim}D (ceiling)", **_keep(irt_cell.metrics)},
                 {"method": f"KNN-router (k={knn.get('k', 5)})",
                  **_keep(_pred_metrics(true, knn_pred))},
                 {"method": "MLP-router", **_keep(_pred_metrics(true, mlp_pred))},
                 {"method": "baseline: per-model mean", **base["model_mean"]},
                 {"method": "baseline: global mean", **base["global_mean"]}]
    pred_table = pd.DataFrame(pred_rows)[["method", "bce", "mse", "auc", "acc@0.5", "spearman_r"]]

    # ============ 2. ranking ============
    rank_table = pd.DataFrame(
        [{"method": k, **ranking_metrics(p, true)} for k, p in learned.items()]
    )

    # ============ 3. routing + cost + reward ============
    tq = train_quality(d, model_ids)
    routing = routing_report(learned, true, cost, model_ids=model_ids,
                             train_quality=tq, reference=reference)
    routing = add_reward_columns(routing, pool_costs, alphas=alphas)

    # ============ 4. AIQ ============
    paretos = {k: pareto(p, true, cost, lams=lams) for k, p in learned.items()}
    aiq_table = pd.DataFrame([
        {"router": k, **aiq(pv, pool_mean_costs=pool_costs, pool_quality=pool_qual)}
        for k, pv in paretos.items()
    ])

    # -- print ------------------------------------------------------
    print(f"\n################ COMPARISON  run={args.run}  eval={label}  "
          f"({n_q} queries, {len(model_ids)} models) ################")
    with float_table(200, **{"display.max_columns": 30}):
        print("\n== 1. PREDICTION QUALITY ==")
        print(pred_table.to_string(index=False))
        print("\n== 2. RANKING (per query) ==")
        print(rank_table.to_string(index=False))
        print("\n== 3. ROUTING + COST + REWARD ==")
        print(routing.to_string(index=False))
        print("\n== 4. AIQ (area under cost/quality curve) ==")
        print(aiq_table.to_string(index=False))
        print("\n-- NIRT Pareto --")
        print(paretos["NIRT"][["lam", "quality", "accuracy", "cost_per_1k_queries"]]
              .to_string(index=False))

    tag = label.replace("[", "_").replace("]", "").replace("+", "-")
    write_json(runs_dir / args.run / f"comparison_{tag}.json", _round({
        "run": args.run, "eval": label, "n_queries": int(n_q), "model_ids": model_ids,
        "prediction": pred_table.to_dict(orient="records"),
        "ranking": rank_table.to_dict(orient="records"),
        "routing": routing.to_dict(orient="records"),
        "aiq": aiq_table.to_dict(orient="records"),
        "pareto": {k: v.to_dict(orient="records") for k, v in paretos.items()},
        "classical_irt_ceiling_metrics": irt_cell.metrics,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
