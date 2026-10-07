"""Where do our routers fail to capture the savings the oracle gets, and on which prompts?

The oracle (evaluation-only) is the cheapest model whose observed score is within
``--tol`` of the query's best. Each router is compared against it, query by query:

    overspend     the router's pick is good enough but the oracle's is >= --min-savings cheaper
    quality_loss  the router's pick scores more than --tol below the best
    ok            neither

    python investigations/oracle_savings_gap/build_dataset.py
    python investigations/oracle_savings_gap/build_dataset.py --lam 50 --zoib artifacts/phase2/zoib

Routers: the NIRT run (--nirt-run) and, optionally, a ZOIB / Bernoulli baseline
checkpoint (--zoib). Selection is ``argmax_m pred - lam * C(m)`` with C(m) the
per-model mean cost (``evaluation.nirt.routing.route``). NOTE: costs are ~1e-4..3e-3
dollars/query, so ``--lam`` below ~5 barely moves a router off its quality argmax --
see the pick mix printed for each router.

Pool and queries follow scripts/route_compare.py: RouterBench models only (so every
query has every model observed), 0-shot items only unless --include-multishot.

Writes to --out (default investigations/oracle_savings_gap/output/):
    gap_all.parquet          every query, incl. the ones no router failed (the contrast set)
    gap_dataset.parquet/.csv the queries at least one router failed on
    by_model.csv             oracle model x router model x failure type, with counts
    failure_rates_by_dataset.csv / failure_rates_by_metric.csv
    reading_sample.md        failed prompts grouped for reading by hand
    summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gap  # noqa: E402

from evaluation.nirt.routing import align, eval_matrices, route  # noqa: E402
from router.config import load_config  # noqa: E402
from router.nirt.checkpoint import load_run  # noqa: E402
from router.nirt.predict import predict_matrix  # noqa: E402
from training.cli import raw_parser, resolve, write_json, zeroshot_only  # noqa: E402
from training.data.facade import load_training_data  # noqa: E402
from training.nirt.baseline.checkpoint import load_baseline_run  # noqa: E402
from training.nirt.baseline.data import checkpoint_matrix  # noqa: E402

DEFAULT_OUT = "investigations/oracle_savings_gap/output"


def _nirt_predictions(run: str, data, split: str):
    model, run_cfg, midx = load_run(run)
    dcfg = run_cfg.get("data", {}) or {}
    frame = predict_matrix(
        model, midx, data, split, pathway=dcfg.get("pathway", "irt"),
        query_pathway=dcfg.get("query_pathway") or None,
        query_features=dcfg.get("query_features") or None,
    )
    return frame, set(midx)


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--config", default=None, help="phase0 / data config")
    ap.add_argument("--split", default="test")
    ap.add_argument("--source", default="routerbench",
                    help="observation source whose models form the pool")
    ap.add_argument("--nirt-run", default="nirt-2d-projected")
    ap.add_argument("--zoib", default=None,
                    help="optional ZOIB checkpoint (scored as E[Y]); must match the current taxonomy")
    ap.add_argument("--lam", type=float, default=0.0, help="cost weight of the routers' selection")
    ap.add_argument("--tol", type=float, default=0.02, help="score within this of the best is 'good enough'")
    ap.add_argument("--min-savings", type=float, default=0.25,
                    help="oracle must be this fraction of the router's cost cheaper to flag overspend")
    ap.add_argument("--include-multishot", action="store_true", help="keep ':5shot' items")
    ap.add_argument("--per-group", type=int, default=3, help="prompts sampled per group in reading_sample.md")
    ap.add_argument("--max-prompts", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = load_training_data(cfg)

    # -- predictions (the pool is what every router can score) ------------------
    nirt_pred, pool_ok = _nirt_predictions(args.nirt_run, data, args.split)
    if args.zoib:
        _, blob, _ = load_baseline_run(args.zoib)
        pool_ok &= set(blob["model_index"])
    obs = data.nirt_observations()
    obs = obs[(obs["split"] == args.split) & (obs["source"] == args.source)]
    pool = sorted(set(obs["model_id"]) & pool_ok)

    true_df, cost_df = eval_matrices(data, split=args.split, models=pool)
    if not args.include_multishot:
        true_df, cost_df = zeroshot_only(true_df, cost_df)
    if true_df.empty:
        sys.exit(f"no fully observed {args.split!r} queries for source {args.source!r} "
                 f"over pool {pool}")
    true, cost = true_df.to_numpy(np.float64), cost_df.to_numpy(np.float64)

    preds = {"nirt": align(nirt_pred, true_df)}
    if args.zoib:
        preds["zoib"] = align(checkpoint_matrix(args.zoib, cfg, args.split, "mean"), true_df)
    for name, p in preds.items():
        if np.isnan(p).any():
            sys.exit(f"router {name!r} has NaN predictions on {int(np.isnan(p).any(axis=1).sum())} "
                     "queries in the pool; refusing to route on missing scores")
    selections = {name: route(p, cost, args.lam) for name, p in preds.items()}
    routers = list(selections)

    # -- compare against the oracle, join the prompt + metadata ---------------
    table = gap.build_gap_table(true_df, cost_df, selections, tol=args.tol,
                                min_savings=args.min_savings)
    queries = data.queries.set_index("query_id")
    metric = obs.groupby("query_id")["metric_type"].first()
    table = table.join(queries[["query", "dataset", "source"]]).join(metric)
    table["prompt_chars"] = table["query"].str.len()
    table = table.reset_index()
    flagged = table[table["failed_by"] != "none"]

    # -- outputs -----------------------------------------------------------------
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "gap_all.parquet", index=False)
    flagged.to_parquet(out / "gap_dataset.parquet", index=False)
    flagged.to_csv(out / "gap_dataset.csv", index=False, encoding="utf-8")
    gap.by_model_table(table, routers).to_csv(out / "by_model.csv", index=False)
    gap.failure_rates_by(table, routers, by="dataset").to_csv(
        out / "failure_rates_by_dataset.csv", index=False)
    gap.failure_rates_by(table, routers, by="metric_type").to_csv(
        out / "failure_rates_by_metric.csv", index=False)
    (out / "reading_sample.md").write_text(
        gap.reading_sample(flagged, routers, per_group=args.per_group,
                           max_prompts=args.max_prompts, seed=args.seed),
        encoding="utf-8",
    )

    n = len(table)
    summary = {
        "config": args.config, "split": args.split, "source": args.source, "pool": list(true_df.columns),
        "nirt_run": args.nirt_run, "zoib": args.zoib, "lam": args.lam, "tol": args.tol,
        "min_savings": args.min_savings, "include_multishot": args.include_multishot,
        "n_queries": n, "n_flagged": len(flagged),
        "oracle": {"mean_quality": float(table["oracle_score"].mean()),
                   "mean_cost_per_1k": float(table["oracle_cost"].mean() * 1000),
                   "model_mix": table["oracle_model"].value_counts().to_dict()},
        "routers": {
            name: {
                "mean_quality": float(table[f"{name}_score"].mean()),
                "mean_cost_per_1k": float(table[f"{name}_cost"].mean() * 1000),
                "failures": table[f"{name}_failure"].value_counts().to_dict(),
                "model_mix": table[f"{name}_model"].value_counts().to_dict(),
                "total_savings_gap": float(table[f"{name}_gap"].clip(lower=0).sum()),
            }
            for name in routers
        },
    }
    write_json(out / "summary.json", summary, announce=False)

    # -- console summary (ASCII: the Windows console is cp1252) ------------------
    print(f"pool: {len(true_df.columns)} models | {n:,} queries | flagged {len(flagged):,} "
          f"({_pct(len(flagged) / n)}) | lam={args.lam} tol={args.tol} min_savings={args.min_savings}")
    print(f"oracle: quality {summary['oracle']['mean_quality']:.4f}  "
          f"${summary['oracle']['mean_cost_per_1k']:.3f}/1k")
    for name, r in summary["routers"].items():
        f = r["failures"]
        top = ", ".join(f"{m} {_pct(c / n).strip()}" for m, c in list(r["model_mix"].items())[:3])
        print(f"{name:>5}: quality {r['mean_quality']:.4f}  ${r['mean_cost_per_1k']:.3f}/1k | "
              f"overspend {_pct(f.get('overspend', 0) / n)}  quality_loss {_pct(f.get('quality_loss', 0) / n)}  "
              f"ok {_pct(f.get('ok', 0) / n)} | picks: {top}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
