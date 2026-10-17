"""Evaluate a ZOIB checkpoint's expected-cost probe on real data.

1. Cost prediction: B0 / B1 / B2 / probe against REALIZED per-cell cost.
2. Routing on one source's dense pool, scored on realized ``Y - lam * C_true``:

   A       predicted quality - lam * C(m)        (per-model train-mean cost)
   B       predicted quality - lam * C_hat(q, m) (the cost probe)
   B*      predicted quality - lam * C_true(q, m) (ceiling for cost prediction alone)
   Oracle  true quality      - lam * C_true(q, m)

   over the fixed 0..200 lambda grid AND one derived from the cost spread
   (``cost_baselines.lambda_grid``).

    python scripts/nirt/baseline/cost_probe_eval.py --ckpt artifacts/phase2/zoib_cost \\
        --split test --source routerbench

Writes <ckpt>/cost_probe_eval_<split>[_<source>].json
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from evaluation.nirt.cost_baselines import (
    cost_baselines,
    cost_prediction_metrics,
    lambda_grid,
)
from evaluation.nirt.routing import align, eval_matrices
from evaluation.nirt.routing_benchmark import baseline_cell_predictions
from evaluation.routing.oracle import routing_evaluation
from router.nirt.frames import pivot_qm
from training.cli import add_source_arg, get_phase0_config, raw_parser, resolve, write_json
from training.data.facade import load_training_data
from training.data.pricing import load_prices
from training.data.tokens import cell_token_table
from training.nirt.baseline.length import length_labels

FIXED_LAMS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)


def _cost_table(train: pd.DataFrame, cells: pd.DataFrame, prices: dict) -> dict:
    preds = {**cost_baselines(train, cells, prices), "probe": cells["cost_pred"].to_numpy()}
    q, true = cells["query_id"].to_numpy(), pd.to_numeric(cells["cost"], errors="coerce").to_numpy()
    out = {"all": {k: cost_prediction_metrics(v, true, q) for k, v in preds.items()}}
    b2_ok = np.isfinite(preds["B2"])
    if b2_ok.any():   # B2 only exists for priced models with token counts: compare like for like
        out["b2_covered"] = {k: cost_prediction_metrics(np.where(b2_ok, v, np.nan), true, q)
                             for k, v in preds.items()}
    return out


def _length_report(long: pd.DataFrame, mcfg: dict) -> dict:
    """Length-classifier diagnostics on the eval split plus the train price fit."""
    ok = np.isfinite(long["output_tokens"].to_numpy(np.float64))
    sub = long[ok]
    vals = np.asarray(mcfg["length_buckets"]["values"])
    lab = length_labels(sub["output_tokens"])

    def err(e, t):
        return float(np.abs(np.log1p(e) - np.log1p(t)).mean())

    by_model = {}
    for m, g in sub.groupby("model_id"):
        pr = np.exp(np.asarray(mcfg["length_priors"]["per_model"].get(
            str(m), mcfg["length_priors"]["default"])))
        price = mcfg.get("token_prices", {}).get(str(m), {})
        by_model[str(m)] = {
            "median_out": float(g["output_tokens"].median()),
            "mean_out": float(g["output_tokens"].mean()),
            "mean_expected": float(g["expected_length"].mean()),
            "prior_mean": float(pr @ vals / pr.sum()),
            "price_r2": price.get("r2"), "p_out_per_1m": (price.get("p_out", np.nan) * 1e6),
        }
    return {"n_cells_with_length": int(ok.sum()),
            "mean_abs_log1p_err": err(sub["expected_length"], sub["output_tokens"]),
            "bucket_hist": np.bincount(lab[lab >= 0], minlength=len(vals)).tolist(),
            "by_model": by_model}


def _row(name, lam, r) -> dict:
    ca = r["cost_aware"]
    return {"router": name, "lam": lam, "quality": r["mean_selected_quality"],
            "cost_per_1k": r["mean_selected_cost_per_1k"],
            "utility": ca["mean_selected_utility"], "utility_regret": ca["mean_utility_regret"],
            "oracle_utility": ca["mean_cost_aware_oracle_utility"],
            "oracle_quality": ca["cost_aware_oracle_mean_quality"],
            "oracle_cost_per_1k": ca["cost_aware_oracle_mean_cost_per_1k"],
            "mix": r["selected_model_mix"]}


def _routing(long: pd.DataFrame, d, split: str, source, c_model: pd.Series) -> dict:
    true_df, cost_df = eval_matrices(d, split, models=sorted(long["model_id"].unique()),
                                     source=source)
    models = list(true_df.columns)
    pq = align(pivot_qm(long, "quality_pred"), true_df)
    pc = align(pivot_qm(long, "cost_pred"), true_df)
    true, cost = true_df.to_numpy(np.float64), cost_df.to_numpy(np.float64)
    keep = np.isfinite(pq).all(1) & np.isfinite(pc).all(1)
    pq, pc, true, cost = pq[keep], pc[keep], true[keep], cost[keep]
    C = c_model.reindex(models).to_numpy(np.float64)
    if not np.isfinite(C).all():
        raise SystemExit(f"no train-mean cost for {list(np.array(models)[~np.isfinite(C)])}")

    derived = lambda_grid(pq, pc)
    rows = []
    for grid, lams in (("fixed", FIXED_LAMS), ("derived", derived)):
        for lam in lams:
            lam = float(lam)
            for name, mc in (("A", C), ("B", pc), ("B*", cost)):
                r = routing_evaluation(pq, true, cost, models, lam=lam, model_costs=mc)
                rows.append({"grid": grid, **_row(name, lam, r)})
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["grid", "lam"], columns="router", values="utility")
    orc = df.groupby(["grid", "lam"])["oracle_utility"].first()
    gap = (orc - piv["A"]).replace(0, np.nan)
    piv["gap_closed_B"] = (piv["B"] - piv["A"]) / gap
    piv["gap_closed_Bstar"] = (piv["B*"] - piv["A"]) / gap
    piv["oracle"] = orc
    return {"n_queries": int(keep.sum()), "n_dropped_missing_pred": int((~keep).sum()),
            "models": models, "derived_lambdas": derived.tolist(),
            "rows": df.to_dict("records"),
            "summary": piv.reset_index().to_dict("records")}


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--phase0-config", default=None)
    add_source_arg(ap)
    args = ap.parse_args()

    cfg = get_phase0_config(args)
    d = load_training_data(cfg)
    ckpt = resolve(args.ckpt, cfg.root)
    long, mcfg = baseline_cell_predictions(ckpt, cfg, d, args.split)

    obs = d.nirt_observations()
    train = obs[obs["split"] == "train"][["query_id", "model_id", "cost", "source"]]
    tok = cell_token_table(d, cfg)
    train = train.merge(tok, on=["query_id", "model_id"], how="left")
    cells = long.merge(tok, on=["query_id", "model_id"], how="left").merge(
        obs[obs["split"] == args.split][["query_id", "model_id", "source"]],
        on=["query_id", "model_id"], how="left")
    # B2 prices: the checkpoint's train-fitted token prices when it has them (they
    # cover RouterBench), else the list prices in configs/model_prices.yaml
    prices = mcfg.get("token_prices") or load_prices(cfg)

    cost_report = {"pooled": _cost_table(train, cells, prices)}
    for src, sub in cells.groupby("source"):
        cost_report[str(src)] = _cost_table(train, sub.reset_index(drop=True), prices)

    tr_cost = pd.to_numeric(train["cost"], errors="coerce")
    c_model = train.assign(cost=tr_cost)[np.isfinite(tr_cost) & (tr_cost >= 0)] \
        .groupby("model_id")["cost"].mean()
    routing = _routing(long, d, args.split, args.source, c_model)

    out = {"ckpt": str(ckpt), "split": args.split, "source": args.source,
           "cost_prediction": cost_report, "routing": routing}
    if "expected_length" in long:
        out["length"] = _length_report(long, mcfg)
    suffix = f"_{args.source}" if args.source else ""
    write_json(ckpt / f"cost_probe_eval_{args.split}{suffix}.json", out)

    print("\n== cost prediction (pooled, all cells) ==")
    print(pd.DataFrame(cost_report["pooled"]["all"]).T.round(4).to_string())
    if "length" in out:
        print("\n== output length ==")
        print(pd.DataFrame(out["length"]["by_model"]).T.round(3).to_string())
        print({k: v for k, v in out["length"].items() if k != "by_model"})
    print(f"\n== routing ({routing['n_queries']} queries, {len(routing['models'])} models) ==")
    print(pd.DataFrame(routing["summary"]).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
