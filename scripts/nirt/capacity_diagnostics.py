"""P1 capacity / ceiling decomposition for the NIRT ``query_latent`` model.

The 2026-09-03 head search saturated latent width ``K`` and MLP width with zero
overfitting (``docs/nirt_model.md``): the model *underfits*. This script
partitions the gap between NIRT and "perfect" into

  1. **irreducible** -- how low BCE can go on this response matrix at all
     (transductive per-query ceiling: free ``theta_q``, no text, matrix
     factorisation; ``fit_classical_irt(protocol="cell")``). Reports held-out-cell
     AND train-cell BCE (the pure rank floor).
  2. **bilinear-bottleneck** -- an IRT-free MLP ``e_q -> R^M`` with dropout
     (``fit_mlp_router``) at increasing width. MLP >> NIRT  =>  the strictly
     bilinear logit is the wall (-> P2c). MLP plateaus with NIRT  =>  the frozen
     ``e_q`` is the wall (-> P4).
  3. **representation** -- NIRT with ``data.query_pathway in {null, knn10w}``.
  4. **per-family** -- NIRT test BCE + routing regret grouped by benchmark family
     (``evaluation.nirt.ood.family_of_query``): which families are underfit.

Every block also reports the **routing regret** of the predictor (the workstream's
success metric), so a prediction-BCE win that never reaches the arg-max is visible.

    python scripts/nirt/capacity_diagnostics.py
    python scripts/nirt/capacity_diagnostics.py --config configs/irt_router.yaml --splits test,ood
    python scripts/nirt/capacity_diagnostics.py --nirt-run nirt-2d-projected --quick

Writes artifacts/phase2/capacity_diagnostics.json  (or artifacts/irt_router/... for
the IRT-Router suite).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from router.config import load_config
from router.nirt.baselines_infer import mlp_router_matrix
from router.nirt.checkpoint import load_run
from router.nirt.predict import predict_matrix, predict_matrix_from_dataset
from training.data.facade import load_training_data
from training.nirt.metrics import marginal_baselines, per_group_metrics, prediction_metrics
from training.trainers.mlp_router import fit_mlp_router
from evaluation.baselines.classical_irt import fit_classical_irt
from evaluation.nirt.evaluate import ranking_metrics
from evaluation.nirt.ood import family_of_query, ood_datasets, ood_matrices, split_observations
from evaluation.nirt.routing import align, eval_matrices
from evaluation.routing.oracle import routing_evaluation

_OOD_FAMILIES = ("math", "code")


def _flat(true_df: pd.DataFrame, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true = true_df.to_numpy(np.float64)
    ok = np.isfinite(pred).all(axis=1)
    return true[ok], pred[ok], ok


def _pred_block(true_df: pd.DataFrame, cost_df: pd.DataFrame, pred: np.ndarray,
                model_ids: list[str]) -> dict:
    t, p, ok = _flat(true_df, pred)
    c = cost_df.to_numpy(np.float64)[ok]
    pm = prediction_metrics(t.ravel(), p.ravel())
    rm = ranking_metrics(p, t)
    re = routing_evaluation(p, t, c, model_ids, lam=0.0)
    return {
        "n_queries": int(ok.sum()),
        "bce": pm["bce"], "brier": pm["brier"], "auc": pm["auc"],
        "acc@0.5": pm["acc@0.5"], "spearman_r": pm["spearman_r"], "ece": pm["ece"],
        "regret": rm["regret"], "optimal_rate": rm["optimal_rate"],
        "oracle_hit_any_best": re["oracle_hit_rate_any_best"],
    }


def _nirt_matrix(run: str, d, split: str, ood_ds=None) -> pd.DataFrame:
    model, cfg, midx = load_run(run)
    dcfg = cfg.get("data", {}) or {}
    pw = dcfg.get("pathway", "irt")
    qp = dcfg.get("query_pathway") or None
    qf = dcfg.get("query_features") or None
    if ood_ds is not None:
        if qp is not None or qf is not None:
            from router.data.nirt import NIRTDataset

            q = d.query_embeddings(qp) if qp is not None else ood_ds.query_store
            feat = d.query_features(qf) if qf is not None else None
            ood_ds = NIRTDataset(ood_ds.obs, q, ood_ds.profile_store, feature_store=feat)
        return predict_matrix_from_dataset(model, midx, ood_ds)
    return predict_matrix(model, midx, d, split, pathway=pw, query_pathway=qp, query_features=qf)


def run_split(d, split: str, seed: int, args) -> dict:
    # IRT-Router ships a first-class `ood` split (test2.csv); RouterBench builds
    # its OOD as held-out benchmark families. Use whichever this config has.
    has_ood_split = d.splits.get("ood") is not None
    is_family_ood = split == "ood" and not has_ood_split
    # a "held-out family" OOD arm cannot fit a transductive ceiling on those
    # families (they are not in any train/val/test split) and reuses train obs.
    is_ood = is_family_ood

    if is_family_ood:
        true_df, cost_df = ood_matrices(d, _OOD_FAMILIES)
    else:
        true_df, cost_df = eval_matrices(d, split=split)
    keep = [not str(q).endswith(":5shot") for q in true_df.index]
    true_df, cost_df = true_df.loc[keep], cost_df.loc[keep]
    model_ids = list(true_df.columns)
    out: dict = {"split": split, "n_queries": int(len(true_df)), "models": model_ids}

    # -- marginal baselines ---------------------------------------------------
    tr = d.nirt_dataset(split="train", pathway="irt")
    base = marginal_baselines(tr.targets, tr.model_ids, true_df.to_numpy(np.float64).ravel(),
                              np.repeat([model_ids], len(true_df), axis=0).ravel())
    out["marginal_baselines"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_model"}
                                 for k, v in base.items()}

    # -- 1. transductive per-query ceiling ----------------------------------
    if not is_ood:
        ceil_rows = []
        dims = [8] if args.quick else [8, 16, 32, 64]
        wds = [1e-3] if args.quick else [1e-4, 1e-3, 1e-2]
        for dim in dims:
            for wd in wds:
                r = fit_classical_irt(d, split=split, protocol="cell", dim=dim,
                                      holdout_frac=0.15, epochs=200 if args.quick else 500,
                                      lr=0.05, weight_decay=wd, seed=seed)
                ceil_rows.append({"dim": dim, "weight_decay": wd,
                                  "heldout_bce": r.metrics["bce"], "heldout_auc": r.metrics["auc"],
                                  "train_bce": r.train_metrics["bce"]})
        out["transductive_ceiling"] = ceil_rows
        out["ceiling_best_heldout_bce"] = min(r["heldout_bce"] for r in ceil_rows)
        out["ceiling_min_train_bce"] = min(r["train_bce"] for r in ceil_rows)

    # -- 2. IRT-free MLP-router bound --------------------------------------
    mlp_rows = []
    grid = [(128, 0.1)] if args.quick else [(128, 0.0), (128, 0.1), (512, 0.1), (2048, 0.1), (2048, 0.3)]
    tr_obs = None
    ood_ds = None
    if is_ood:
        tr_obs, _ = split_observations(d, _OOD_FAMILIES)
        _, _, ood_ds = ood_datasets(d, _OOD_FAMILIES, pathway="irt")
    for hidden, dropout in grid:
        model, mids = fit_mlp_router(d, hidden=hidden, dropout=dropout,
                                     epochs=20 if args.quick else 40, pathway="irt",
                                     seed=seed, train_obs=tr_obs)
        M = align(mlp_router_matrix(model, mids, d, list(true_df.index), pathway="irt"), true_df)
        mlp_rows.append({"hidden": hidden, "dropout": dropout,
                         **_pred_block(true_df, cost_df, M, model_ids)})
    out["mlp_router"] = mlp_rows

    # -- 3. NIRT reference + representation ablation ----------------------
    nirt_rows = []
    for run in ([args.nirt_run] if args.quick else args.nirt_runs):
        try:
            M = align(_nirt_matrix(run, d, split, ood_ds=ood_ds if is_ood else None), true_df)
        except FileNotFoundError:
            print(f"[cx-diag] nirt run '{run}' not found; skipping")
            continue
        nirt_rows.append({"run": run, **_pred_block(true_df, cost_df, M, model_ids)})
    out["nirt"] = nirt_rows

    # -- 4. per-family breakdown for the primary NIRT run -----------------
    if nirt_rows:
        try:
            M = align(_nirt_matrix(args.nirt_run, d, split,
                                   ood_ds=ood_ds if is_ood else None), true_df)
            t, p, ok = _flat(true_df, M)
            fam_of = family_of_query(d)
            qids = np.asarray(true_df.index)[ok]
            fam = np.array([fam_of.get(str(q), "other") for q in qids])
            fam_rep = np.repeat(fam, t.shape[1])
            pg = per_group_metrics(t.ravel(), p.ravel(), fam_rep, min_n=50)
            out["per_family"] = {k: {kk: v[kk] for kk in ("n", "bce", "auc", "acc@0.5", "pos_rate")}
                                 for k, v in pg.items()}
        except FileNotFoundError:
            pass

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="phase-0 config (default: RouterBench)")
    ap.add_argument("--splits", default="test", help="comma list: test[,ood]")
    ap.add_argument("--nirt-run", default="nirt-2d-projected", help="primary NIRT run for blocks 3/4")
    ap.add_argument("--nirt-runs", default=None,
                    help="comma list for the representation ablation (default: raw + knn10w)")
    ap.add_argument("--quick", action="store_true", help="tiny grids / fewer epochs for a smoke run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.nirt_runs = ([s.strip() for s in args.nirt_runs.split(",")] if args.nirt_runs
                      else [args.nirt_run, "nirt-knn10w-2d-projected"])

    cfg = load_config(args.config) if args.config else load_config()
    d = load_training_data(cfg)
    seed = int(yaml.safe_load(Path(cfg.root / "configs/nirt.yaml").read_text("utf-8")).get("seed", 42))
    results = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        print(f"\n########## capacity diagnostics -- {split} ##########")
        results[split] = run_split(d, split, seed, args)
        r = results[split]
        print(f"  marginal model_mean BCE : {r['marginal_baselines']['model_mean']['bce']:.4f}")
        if "ceiling_best_heldout_bce" in r:
            print(f"  transductive ceiling BCE: {r['ceiling_best_heldout_bce']:.4f}  "
                  f"(min train-cell {r['ceiling_min_train_bce']:.4f})")
        for row in r["mlp_router"]:
            print(f"  MLP-router h{row['hidden']} do{row['dropout']}: bce {row['bce']:.4f}  regret {row['regret']:.4f}")
        for row in r["nirt"]:
            print(f"  NIRT {row['run']:32s}: bce {row['bce']:.4f}  regret {row['regret']:.4f}  "
                  f"hit* {row['oracle_hit_any_best']:.3f}")
        for fam, m in r.get("per_family", {}).items():
            print(f"    [{fam:8s}] n={m['n']:>7d}  bce {m['bce']:.4f}  auc {m['auc']:.4f}  pos {m['pos_rate']:.3f}")

    is_irt = args.config and "irt_router" in str(args.config)
    out = Path(args.out) if args.out else Path(
        "artifacts/irt_router/capacity_diagnostics.json" if is_irt
        else "artifacts/phase2/capacity_diagnostics.json")
    out = out if out.is_absolute() else cfg.root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": args.config, "splits": results}, indent=2, default=float),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
