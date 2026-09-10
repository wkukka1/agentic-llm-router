"""Oracle-routing evaluation for a trained NIRT / ZOIB predictor.

SEPARATE from predictive (NLL / calibration) evaluation -- see
`scripts/nirt/eval_nirt.py` and `scripts/evaluate_continuous.py` for that. This
script turns a predictor into routing decisions and scores them against the
oracle derived from the split's ACTUAL observed outcomes (evaluation-only; never
fed back into the model). See `docs/routing_evaluation.md`.

    python scripts/nirt/route_eval.py --run irtrouter-irt-25d-projected --split test \
        --config configs/irt_router.yaml --lam 0.3 --tolerance 0.02 \
        --zoib artifacts/irt_router/zoib --bernoulli artifacts/irt_router/bernoulli \
        --oracle-classifier

Writes  <runs_dir>/<run>/route_eval_<split>.json
        <runs_dir>/<run>/route_eval_<split>_per_query.csv
        <runs_dir>/<run>/oracle_labels_<split>.parquet
"""

from __future__ import annotations

import sys

import numpy as np
import yaml

from router.cli import float_table, raw_parser, resolve, write_json
from router.config import load_config
from router.data.phase1 import load_phase1
from router.nirt.baseline.data import checkpoint_matrix
from router.nirt.evaluate import predict_matrix
from router.nirt.routing import align, eval_matrices
from router.nirt.routing_eval import (
    compare_routing_strategies,
    oracle_classifier_matrix,
    oracle_labels,
    routing_evaluation,
)
from router.nirt.train import load_run as nirt_load_run


def _zoib_matrix(ckpt: str, cfg, split: str, field: str, true_df) -> np.ndarray:
    """Predicted [Q,M] for a continuous / Bernoulli checkpoint (field: mean|proba|lower)."""
    return align(checkpoint_matrix(ckpt, cfg, split, field), true_df)


def _fmt_pct(x):
    return "n/a" if x is None else f"{100 * x:5.1f}%"


def _fmt(x, w=7, p=4):
    return "n/a" if x is None else f"{x:{w}.{p}f}"


def _summary(run: str, split: str, nirt_q: dict, nirt_ca: dict, n_models: int) -> str:
    ca = nirt_ca.get("cost_aware", {})
    L = [
        "=== NIRT ROUTING EVALUATION ===",
        "",
        f"Run:                    {run}",
        f"Split:                  {split}",
        f"Candidate models:       {n_models}",
        f"Test queries:           {nirt_q['n_queries']:,}",
        f"Avg candidates/query:   {nirt_q['n_candidates']['mean']:.1f}",
        "",
        "Quality routing (argmax predicted score)",
        "----------------------------------------",
        f"Selected quality:       {_fmt(nirt_q['mean_selected_quality'])}",
        f"Oracle quality:         {_fmt(nirt_q['mean_oracle_quality'])}",
        f"Oracle hit rate:        {_fmt_pct(nirt_q['oracle_hit_rate'])}   "
        f"(any-best {_fmt_pct(nirt_q['oracle_hit_rate_any_best'])})",
        f"Mean regret:            {_fmt(nirt_q['mean_regret'])}",
        f"Median regret:          {_fmt(nirt_q['median_regret'])}",
        f"P90 regret:             {_fmt(nirt_q['p90_regret'])}",
        f"Zero-regret rate:       {_fmt_pct(nirt_q['zero_regret_rate'])}",
        f"Within-tol rate (<={nirt_q['tolerance']:.3f}):  {_fmt_pct(nirt_q['within_tolerance_rate'])}",
    ]
    if "mean_selected_cost_per_1k" in nirt_q:
        L.append(f"Selected cost:          ${nirt_q['mean_selected_cost_per_1k']:.4f} / 1k")
    if ca:
        L += [
            "",
            f"Cost-aware routing (lambda = {nirt_ca['lam']:.3f})",
            "------------------------------------",
            f"Selected quality:       {_fmt(nirt_ca['mean_selected_quality'])}",
            f"Selected cost:          ${nirt_ca.get('mean_selected_cost_per_1k', float('nan')):.4f} / 1k",
            f"Selected utility:       {_fmt(ca['mean_selected_utility'])}",
            f"Cost-aware oracle util: {_fmt(ca['mean_cost_aware_oracle_utility'])}",
            f"Utility regret:         {_fmt(ca['mean_utility_regret'])}",
            f"  quality-oracle:       q {_fmt(ca['quality_oracle_mean_quality'])}  "
            f"${ca['quality_oracle_mean_cost_per_1k']:.4f}/1k",
            f"  cost-aware-oracle:    q {_fmt(ca['cost_aware_oracle_mean_quality'])}  "
            f"${ca['cost_aware_oracle_mean_cost_per_1k']:.4f}/1k",
        ]
    return "\n".join(L)


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--run", required=True, help="NIRT run name under runs_dir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default=None, help="phase0 / data config")
    ap.add_argument("--nirt-config", default="configs/nirt.yaml")
    ap.add_argument("--lam", type=float, default=0.0, help="cost weight for the cost-aware router/oracle")
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="regret <= this counts as 'within tolerance' of the oracle")
    ap.add_argument("--zoib", default=None, help="optional ZOIB checkpoint to compare (E[Y])")
    ap.add_argument("--bernoulli", default=None, help="optional Bernoulli checkpoint to compare")
    ap.add_argument("--oracle-classifier", action="store_true",
                    help="also fit the direct hard-oracle classifier baseline (train-split labels only)")
    ap.add_argument("--per-query-model", action="store_true",
                    help="also dump the full [Q x M] predicted/actual/rank table")
    args = ap.parse_args()

    ncfg = yaml.safe_load(resolve(args.nirt_config).read_text(encoding="utf-8"))
    runs_dir = resolve(ncfg.get("runs_dir", "data/processed/nirt_runs"))

    p0 = load_config(args.config)
    d = load_phase1(p0)

    model, run_cfg, midx = nirt_load_run(args.run, runs_dir=runs_dir)
    pathway = (run_cfg.get("data", {}) or {}).get("pathway", "irt")
    pool = sorted(midx, key=midx.get)

    true_df, cost_df = eval_matrices(d, split=args.split, models=pool)
    model_ids = list(true_df.columns)
    true = true_df.to_numpy(np.float64)
    cost = cost_df.to_numpy(np.float64)
    query_ids = list(true_df.index)

    # -- oracle labels (evaluation-only artifact, this pool + split) -----------
    labels = oracle_labels(true_df, cost_df)

    # -- predictor matrices --------------------------------------------------
    preds: dict[str, np.ndarray] = {
        f"NIRT ({args.run})": align(predict_matrix(model, midx, d, args.split, pathway=pathway), true_df),
    }
    if args.zoib:
        preds["ZOIB E[Y]"] = _zoib_matrix(args.zoib, p0, args.split, "mean", true_df)
    if args.bernoulli:
        preds["Bernoulli"] = _zoib_matrix(args.bernoulli, p0, args.split, "proba", true_df)
    if args.oracle_classifier:
        q_store = d.query_embeddings(pathway)
        tr_true, _ = eval_matrices(d, split="train", models=pool)
        tr_emb = q_store.gather([q for q in tr_true.index if q in q_store])
        tr_true = tr_true.loc[[q for q in tr_true.index if q in q_store]]
        ev_emb = q_store.gather(query_ids)
        preds["oracle-classifier (baseline D)"] = oracle_classifier_matrix(
            tr_emb, tr_true, ev_emb, model_ids
        )

    # -- headline: the NIRT run, quality + cost-aware ----------------------
    nirt_key = f"NIRT ({args.run})"
    nirt_q = routing_evaluation(preds[nirt_key], true, cost, model_ids,
                                lam=0.0, tolerance=args.tolerance, query_ids=query_ids)
    nirt_ca = routing_evaluation(preds[nirt_key], true, cost, model_ids,
                                 lam=args.lam, tolerance=args.tolerance, query_ids=query_ids)
    per_query = nirt_ca.pop("_per_query")
    nirt_q.pop("_per_query", None)

    summary_df, per_strategy = compare_routing_strategies(
        preds, true, cost, model_ids, lam=args.lam, tolerance=args.tolerance,
        query_ids=query_ids,
    )

    print(_summary(args.run, args.split, nirt_q, nirt_ca, len(model_ids)))
    print("\n=== routing strategies (§9) ===")
    with float_table(220, **{"display.max_colwidth": 40}):
        print(summary_df.to_string(index=False))

    # -- artifacts --------------------------------------------------------
    out_dir = runs_dir / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    lab_path = out_dir / f"oracle_labels_{args.split}.parquet"
    labels.to_parquet(lab_path, index=False)
    pq_path = out_dir / f"route_eval_{args.split}_per_query.csv"
    per_query.to_csv(pq_path, index=False)

    payload = {
        "run": args.run, "split": args.split, "config": args.config,
        "pool": model_ids, "n_queries": int(true.shape[0]),
        "lam": args.lam, "tolerance": args.tolerance,
        "target": "y_soft (chance-corrected graded score, split via eval_matrices)",
        "oracle_note": (
            "oracle = cheapest model attaining each query's max actual score; "
            "oracle_flag is tie-inclusive. evaluation-only -- never used in training/"
            "features/embeddings/candidate-selection."
        ),
        "nirt_quality_routing": nirt_q,
        "nirt_cost_aware_routing": nirt_ca,
        "strategy_comparison": summary_df.to_dict(orient="records"),
        "strategy_detail": per_strategy,
        "oracle_label_summary": {
            "n_queries": int(len(query_ids)),
            "mean_oracle_score": float(labels.groupby("query_id")["oracle_score"].first().mean()),
            "mean_n_oracle_ties": float(labels.groupby("query_id")["n_oracle_ties"].first().mean()),
            "frac_queries_with_ties": float(
                (labels.groupby("query_id")["n_oracle_ties"].first() > 1).mean()
            ),
        },
    }
    ev_path = write_json(out_dir / f"route_eval_{args.split}.json", payload, announce=False)

    if args.per_query_model:
        from router.nirt.routing_eval import per_query_model_table

        pqm = per_query_model_table(preds[nirt_key], true, cost, model_ids, query_ids)
        pqm.to_parquet(out_dir / f"route_eval_{args.split}_per_query_model.parquet", index=False)

    print(f"\nwrote {ev_path}\n      {pq_path}\n      {lab_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
