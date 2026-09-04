"""A/B the two IRT orientations on the same data / seed / budget.

    python scripts/nirt/ab_orientation.py --dim 2 --model-params projected --epochs 40

  query_latent (ours)     : theta_q = f(e_q) ;  y = sigmoid(a_m . theta_q - b_m)
  model_latent (IRT-Router): theta_m from e_m ;  y = sigmoid(a_q . theta_m - b_q)

Trains both (reusing saved runs when --reuse), then compares prediction / ranking
/ routing on <split> against the shared classical-IRT transductive ceiling.
Writes <runs_dir>/_ab_orientation_<split>.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import yaml

from router.cli import float_table, raw_parser, resolve, write_json
from router.config import load_config
from router.data.phase1 import load_phase1
from router.nirt.baselines import fit_classical_irt
from router.nirt.evaluate import predict_matrix, ranking_metrics
from router.nirt.metrics import prediction_metrics
from router.nirt.routing import align, eval_matrices, pareto, routing_report, train_quality
from router.nirt.train import fit, load_run

_ORI = {"query_latent": "nirt", "model_latent": "irt"}


def _load_cfg(path) -> dict:
    c = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    c.setdefault("model", {})
    c.setdefault("train", {})
    return c


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--model-params", choices=["projected", "free"], default="projected")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--reuse", action="store_true", help="reuse saved runs if present")
    ap.add_argument("--config", default="configs/nirt.yaml")
    ap.add_argument("--phase0-config", default=None)
    args = ap.parse_args()

    base_cfg = _load_cfg(args.config)
    runs_dir = resolve(base_cfg.get("runs_dir", "data/processed/nirt_runs"))

    p0 = load_config(args.phase0_config) if args.phase0_config else load_config()
    d = load_phase1(p0)
    pathway = base_cfg.get("data", {}).get("pathway", "irt")
    reference = base_cfg.get("routing", {}).get("reference_model", "gpt-4-1106-preview")

    # -- shared matrices + classical-IRT ceiling --------------------------
    true_df, cost_df = eval_matrices(d, split=args.split)
    model_ids = list(true_df.columns)
    true, cost = true_df.to_numpy(np.float64), cost_df.to_numpy(np.float64)
    tq = train_quality(d, model_ids)
    irt_dim = base_cfg.get("classical_irt", {}).get("dim") or args.dim
    ceil = fit_classical_irt(d, split=args.split, protocol="cell", dim=irt_dim,
                             holdout_frac=base_cfg.get("classical_irt", {}).get("holdout_frac", 0.15),
                             weight_decay=base_cfg.get("classical_irt", {}).get("weight_decay", 1e-3),
                             seed=int(base_cfg.get("seed", 42)))
    ceil_pred = align(pd.DataFrame(ceil.pred_matrix, index=ceil.query_ids, columns=ceil.model_ids), true_df)

    rows = [_row(f"classical IRT {irt_dim}D (ceiling)", ceil_pred, true, cost, model_ids, tq,
                 reference, pred_metrics=ceil.metrics)]
    preds = {}

    for orientation in ("query_latent", "model_latent"):
        name = f"{_ORI[orientation]}-{args.dim}d-{args.model_params}"
        run_dir = runs_dir / name
        if args.reuse and (run_dir / "model.pt").exists():
            model, run_cfg, midx = load_run(name, runs_dir=runs_dir)
            print(f"[ab] reusing {name}")
        else:
            cfg = json.loads(json.dumps(base_cfg))
            cfg["model"].update(orientation=orientation, dim=args.dim,
                                model_params=args.model_params)
            if args.epochs:
                cfg["train"]["epochs"] = args.epochs
            print(f"[ab] training {name} ...")
            res = fit(cfg, phase0_cfg=p0, name=name, verbose=False)
            model, midx = res.model, res.model_index

        pred = align(predict_matrix(model, midx, d, args.split, pathway), true_df)
        preds[orientation] = pred
        rows.append(_row(f"{orientation}", pred, true, cost, model_ids, tq, reference))

    table = pd.DataFrame(rows)
    print(f"\n================ ORIENTATION A/B  (split={args.split}, dim={args.dim}, "
          f"{args.model_params}) ================")
    with float_table(200, **{"display.max_columns": 30}):
        print(table.to_string(index=False))

    # verdict
    ql, ml = table[table.method == "query_latent"].iloc[0], table[table.method == "model_latent"].iloc[0]
    better = "model_latent" if ml["bce"] < ql["bce"] else "query_latent"
    print(f"\n>>> lower test BCE: {better}  "
          f"(query_latent {ql['bce']:.4f} vs model_latent {ml['bce']:.4f}); "
          f"ranking optimal_rate {ql['optimal_rate']:.3f} vs {ml['optimal_rate']:.3f}; "
          f"both vs ceiling {rows[0]['bce']:.4f}.")

    write_json(runs_dir / f"_ab_orientation_{args.split}.json", {
        "split": args.split, "dim": args.dim, "model_params": args.model_params,
        "model_ids": model_ids, "table": table.to_dict(orient="records"),
        "pareto": {o: pareto(p, true, cost).to_dict(orient="records") for o, p in preds.items()},
        "classical_irt_ceiling": ceil.metrics,
    })
    return 0


def _row(method, pred, true, cost, model_ids, tq, reference, pred_metrics=None) -> dict:
    flat = pred_metrics or prediction_metrics(true.ravel(), pred.ravel())
    rank = ranking_metrics(pred, true)
    rep = routing_report({"m": pred}, true, cost, model_ids=model_ids,
                         train_quality=tq, reference=reference)
    r0 = rep[rep.policy.str.startswith("route: m")].iloc[0]
    par = pareto(pred, true, cost)
    ref_acc = float((true[:, model_ids.index(reference)] >= 0.5).mean()) if reference in model_ids else np.nan
    ok = par[par["accuracy"] >= ref_acc - 0.03]
    cheap = ok["cost"].min() if not ok.empty else float("nan")
    return {
        "method": method,
        "bce": flat["bce"], "auc": flat["auc"], "spearman": flat["spearman_r"],
        "optimal_rate": rank["optimal_rate"], "regret": rank["regret"],
        "route_q@l0": float(r0["quality"]), "route_acc@l0": float(r0["accuracy"]),
        "cost@-3pt_per1k": float(cheap * 1000),
    }


if __name__ == "__main__":
    sys.exit(main())
