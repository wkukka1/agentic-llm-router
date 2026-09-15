"""Where does the NIRT router route to a worse model than the oracle?

Attribution layer on top of scripts/nirt/route_eval.py. Consumes the artifacts
that route_eval already wrote for a run (pass --per-query-model there first):

    route_eval_<split>_per_query.csv
    route_eval_<split>_per_query_model.parquet
    oracle_labels_<split>.parquet

and joins queries.parquet / nirt_observations.parquet for the task / metric
grouping. Breaks the routing regret down by size, by whether a good model was
even available, by dataset / source / metric, by which model was wrongly picked
vs wrongly skipped, and by where NIRT ranked the model it should have taken.

    python scripts/nirt/misroute_analysis.py --run nirt-2d-projected --split test

Writes <runs_dir>/<run>/misroute_<split>.json
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import yaml

from training.cli import float_table, raw_parser, resolve, write_json
from router.config import load_config

_REGRET_EDGES = [-1, 1e-9, 0.02, 0.1, 0.25, 0.5, 0.9, 1.01]
_REGRET_NAMES = ["0", "(0,.02]", "(.02,.1]", "(.1,.25]", "(.25,.5]", "(.5,.9]", "(.9,1]"]


def _pct(x: float) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:5.1f}%"


def _load(run_dir, split, processed):
    pq = pd.read_csv(run_dir / f"route_eval_{split}_per_query.csv")
    pqm_path = run_dir / f"route_eval_{split}_per_query_model.parquet"
    if not pqm_path.exists():
        sys.exit(f"missing {pqm_path.name} -- rerun route_eval.py with --per-query-model")
    pqm = pd.read_parquet(pqm_path)
    q = pd.read_parquet(processed / "queries.parquet")[["query_id", "dataset", "source"]]
    obs = pd.read_parquet(processed / "nirt_observations.parquet")
    meta = (obs.groupby("query_id")
            .agg(metric_type=("metric_type", "first"),
                 is_mc=("is_multiple_choice", "first"))
            .reset_index())
    pq = pq.merge(q, on="query_id", how="left").merge(meta, on="query_id", how="left")
    return pq, pqm


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--nirt-config", default="configs/nirt.yaml")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="regret > tol counts as a mis-route")
    ap.add_argument("--recoverable-at", type=float, default=0.8,
                    help="oracle_score >= this => a good model was available")
    args = ap.parse_args()

    ncfg = yaml.safe_load(resolve(args.nirt_config).read_text(encoding="utf-8"))
    run_dir = resolve(ncfg.get("runs_dir", "data/processed/nirt_runs")) / args.run
    processed = load_config().path("processed")

    pq, pqm = _load(run_dir, args.split, processed)
    n = len(pq)
    reg = pq["oracle_regret"].to_numpy()
    mis = pq["oracle_regret"] > args.tol
    total = float(reg.sum())
    J: dict = {"run": args.run, "split": args.split, "tol": args.tol, "n_queries": n}

    print(f"\n{'=' * 68}\n  NIRT MIS-ROUTING  --  run={args.run}  split={args.split}\n{'=' * 68}")
    print(f"queries                      {n:,}")
    print(f"any-best hits (regret<=tol)  {(~mis).sum():,}   ({_pct((~mis).mean())})")
    print(f"MIS-ROUTES (regret>{args.tol})       {int(mis.sum()):,}   ({_pct(mis.mean())})")
    print(f"mean regret (all / misrouted) {reg.mean():.4f} / {reg[mis.values].mean():.4f}")
    print(f"total regret mass            {total:,.1f} quality-points")
    J["headline"] = {
        "any_best_rate": float((~mis).mean()), "misroute_rate": float(mis.mean()),
        "mean_regret": float(reg.mean()), "total_regret_mass": total,
    }

    # 1. size of the misroutes ------------------------------------------------
    b = pd.cut(pq.loc[mis, "oracle_regret"], _REGRET_EDGES, labels=_REGRET_NAMES)
    t = pd.DataFrame({"queries": b.value_counts().reindex(_REGRET_NAMES, fill_value=0)})
    mass = pq.loc[mis].groupby(b, observed=False)["oracle_regret"].sum().reindex(_REGRET_NAMES)
    t["% misroutes"] = (t["queries"] / max(int(mis.sum()), 1)).map(_pct)
    t["regret mass"] = mass.round(1)
    t["% all regret"] = (mass / total).map(_pct)
    print(f"\n--- size of the {int(mis.sum()):,} mis-routes ---")
    print(t.to_string())
    cat = pq["oracle_regret"] > 0.5
    print(f"catastrophic (regret>0.5): {int(cat.sum()):,} ({_pct(cat.mean())}) "
          f"= {_pct(pq.loc[cat, 'oracle_regret'].sum() / total)} of all regret")
    J["regret_buckets"] = {k: {"queries": int(t.loc[k, "queries"]),
                               "regret_mass": float(mass[k])} for k in _REGRET_NAMES}

    # 2. was a good model available? ---------------------------------------
    rec = mis & (pq["oracle_score"] >= args.recoverable_at)
    rec_share = pq.loc[rec, "oracle_regret"].sum() / total
    print(f"\n--- recoverable mis-routes (oracle_score >= {args.recoverable_at}) ---")
    print(f"{int(rec.sum()):,} queries carrying {_pct(rec_share)} of all regret "
          f"had a pool model scoring >= {args.recoverable_at} that NIRT passed over")
    J["recoverable_regret_share"] = float(rec_share)

    # 3. by dataset / source / metric ------------------------------------
    def _grp(col, head=None):
        g = (pq.assign(_m=mis).groupby(col)
             .agg(q=("query_id", "size"), any_best=("oracle_hit_any_best", "mean"),
                  mean_regret=("oracle_regret", "mean"),
                  regret_mass=("oracle_regret", "sum"),
                  oracle_q=("oracle_score", "mean"),
                  sel_q=("selected_actual_score", "mean"))
             .sort_values("regret_mass", ascending=False))
        g["share"] = g["regret_mass"] / total
        disp = g.head(head) if head else g
        out = disp.copy()
        out["any_best"] = out["any_best"].map(_pct)
        out["share"] = out["share"].map(_pct)
        with float_table(160):
            print(out.round(3).to_string())
        return g

    print("\n--- by dataset (top by regret mass) ---")
    gd = _grp("dataset", head=15)
    J["by_dataset"] = {k: {"queries": int(v.q), "any_best_rate": float(v.any_best),
                           "regret_mass": float(v.regret_mass), "regret_share": float(v.share)}
                       for k, v in gd.head(15).iterrows()}
    for c in ["source", "metric_type"]:
        print(f"\n--- by {c} ---")
        _grp(c)

    # 4. model confusion on mis-routes ----------------------------------
    print(f"\n{'=' * 68}\n  MODEL CONFUSION (mis-routed queries)\n{'=' * 68}")
    mq = set(pq.loc[mis, "query_id"])
    sel = pq.loc[mis, ["query_id", "selected_model_id", "oracle_regret", "oracle_model_id"]]
    best_opts = pqm[(pqm.query_id.isin(mq)) & (pqm.oracle_flag == 1)]
    picked = sel["selected_model_id"].value_counts()
    skipped = (best_opts.merge(sel[["query_id", "selected_model_id"]], on="query_id")
               .query("model_id != selected_model_id")["model_id"].value_counts())
    conf = (pd.DataFrame({"wrongly_picked": picked, "wrongly_skipped": skipped})
            .fillna(0).astype(int))
    conf["net"] = conf.wrongly_picked - conf.wrongly_skipped
    print("\n" + conf.sort_values("net", ascending=False).to_string())

    swaps = (sel.groupby(["oracle_model_id", "selected_model_id"])
             .agg(queries=("query_id", "size"), regret_mass=("oracle_regret", "sum"))
             .sort_values("regret_mass", ascending=False).head(12))
    print("\n--- oracle_model -> selected_model, by regret mass ---")
    print(swaps.round(1).to_string())
    J["confusion"] = {m: {"wrongly_picked": int(r.wrongly_picked),
                          "wrongly_skipped": int(r.wrongly_skipped)}
                      for m, r in conf.iterrows()}

    # 5. how NIRT ranked the model it should have picked -----------------
    print(f"\n{'=' * 68}\n  RANKING QUALITY on mis-routes\n{'=' * 68}")
    rk = best_opts.groupby("query_id")["model_rank"].min().reindex(list(mq))
    rb = pd.cut(rk, [0, 1, 2, 3, 5, 8, 99], labels=["1", "2", "3", "4-5", "6-8", "9+"])
    rt = pd.DataFrame({"queries": rb.value_counts().reindex(["1", "2", "3", "4-5", "6-8", "9+"], fill_value=0)})
    rmass = pq.loc[mis].set_index("query_id")["oracle_regret"].reindex(rk.index)
    rt["regret mass"] = rmass.groupby(rb.values, observed=False).sum().reindex(rt.index).round(1)
    rt["%"] = (rt["queries"] / len(mq)).map(_pct)
    print("\nNIRT's predicted rank of the best model it could have picked:")
    print(rt.to_string())
    print(f"\nmedian rank of that model on mis-routes: {rk.median():.0f} of "
          f"{pqm.model_id.nunique()}")
    J["best_available_rank"] = {"median": float(rk.median()),
                                "at_rank_2_share": float((rk == 2).mean())}

    # 6. utilization vs a constant 'always best-average-model' policy ---
    print(f"\n{'=' * 68}\n  UTILIZATION (all {n:,} queries)\n{'=' * 68}")
    piv = pqm.pivot(index="query_id", columns="model_id", values="actual_score")
    oq = piv.max(axis=1)
    fixed = {m: float((oq - piv[m]).clip(lower=0).mean()) for m in piv.columns}
    best_fixed = min(fixed, key=fixed.get)
    selall = pq["selected_model_id"].value_counts()
    flag_all = pqm[pqm.oracle_flag == 1].groupby("model_id").query_id.nunique()
    u = pd.DataFrame({"NIRT picks": selall, "is a best option": flag_all}).fillna(0).astype(int)
    print("\n" + u.sort_values("NIRT picks", ascending=False).to_string())
    print(f"\nbest single fixed model:      {best_fixed}  (mean regret {fixed[best_fixed]:.4f})")
    print(f"NIRT mean regret:             {reg.mean():.4f}")
    print(f"  => NIRT beats 'always {best_fixed}' by {fixed[best_fixed] - reg.mean():+.4f}")
    J["vs_best_fixed_model"] = {"model": best_fixed, "fixed_mean_regret": fixed[best_fixed],
                               "nirt_mean_regret": float(reg.mean())}

    out = write_json(run_dir / f"misroute_{args.split}.json", J, announce=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
