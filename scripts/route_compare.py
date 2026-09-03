"""Routing report: the Phase 2 ZOIB model vs the Phase 1 Bernoulli baseline vs our
NIRT model (``query_latent``) vs the IRT-Router paper baseline (``model_latent``,
arXiv 2506.01048) vs "just always use the best model".

Turns each per-(query, model) predictor into a policy `argmax_m U(q, m)`
(`U = pred - lam * cost_norm`) on the evaluation split(s), and reports achieved
accuracy / graded quality / cost and the savings vs a fixed reference model.
Analysis only -- the routing orchestrator itself is Phase 4.

    # RouterBench (Phase 2 default)
    python scripts/route_compare.py
    python scripts/route_compare.py --nirt-run nirt-4d-projected --lam 0.5

    # IRT-Router benchmark: in-distribution (test) + out-of-distribution (ood)
    python scripts/route_compare.py --config configs/irt_router.yaml \
        --reference gpt_4o --splits test,ood --lam 0.3 \
        --zoib artifacts/irt_router/zoib --baseline artifacts/irt_router/bernoulli \
        --nirt-run irtrouter-nirt-2d-projected --irt-run irtrouter-irt-25d-projected
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from router.config import load_config
from router.data.phase1 import load_phase1
from router.nirt.baseline_data import batched_forward, build_arrays
from router.nirt.checkpoint import load_run
from router.nirt.routing import (
    add_reward_columns,
    aiq,
    align,
    eval_matrices,
    pareto,
    routing_report,
    train_quality,
)

DEFAULT_REFERENCE = "gpt-4-1106-preview"


def _pred_matrix(ckpt, cfg, split, field, level=0.9):
    model, blob, s = load_run(ckpt)
    ev = build_arrays(cfg, split=split, pathway=s["pathway"], binary_threshold=s["binary_threshold"],
                      score_kind=s["score_kind"], use_relevance=model.use_relevance,
                      use_warmup=model.use_warmup, model_index=blob["model_index"])
    v = batched_forward(model, ev, fields=(field,), level=level)[field]
    return pd.DataFrame({"query_id": ev.query_ids, "model_id": ev.model_ids, "v": v}) \
        .pivot_table(index="query_id", columns="model_id", values="v", aggfunc="mean")


def _nirt_matrix(run_name, data, split):
    from router.nirt.evaluate import predict_matrix
    from router.nirt.train import load_run as nirt_load_run

    model, cfg, midx = nirt_load_run(run_name)
    dcfg = cfg.get("data", {}) or {}
    pathway = dcfg.get("pathway", "irt")
    query_pathway = dcfg.get("query_pathway") or None
    return predict_matrix(model, midx, data, split, pathway=pathway, query_pathway=query_pathway)


def _run_split(split: str, label: str, d, cfg, args, pool) -> dict:
    """Build every policy for one split, print the table + ZOIB frontier, and
    return the JSON-serialisable payload."""
    true_df, cost_df = eval_matrices(d, split=split, models=pool)
    if not args.include_multishot:
        keep = [not str(q).endswith(":5shot") for q in true_df.index]
        true_df, cost_df = true_df.loc[keep], cost_df.loc[keep]
    model_ids = list(true_df.columns)
    true, cost = true_df.to_numpy(np.float64), cost_df.to_numpy(np.float64)
    tq_acc = train_quality(d, model_ids, metric="accuracy")

    preds = {
        f"NIRT ({args.nirt_run})": align(_nirt_matrix(args.nirt_run, d, split), true_df),
    }
    if str(args.irt_run).lower() != "none":
        preds[f"IRT-Router (paper, {args.irt_run})"] = \
            align(_nirt_matrix(args.irt_run, d, split), true_df)
    preds.update({
        "baseline NIRT (Bernoulli)": align(_pred_matrix(args.baseline, cfg, split, "proba"), true_df),
        "ZOIB  E[Y]": align(_pred_matrix(args.zoib, cfg, split, "mean"), true_df),
        f"ZOIB  LCB(level={args.lcb_level})":
            align(_pred_matrix(args.zoib, cfg, split, "lower", level=args.lcb_level), true_df),
    })

    rep = routing_report(preds, true, cost, model_ids=model_ids, train_quality=tq_acc,
                         reference=args.reference, lam=args.lam)
    rep = add_reward_columns(rep, pool_mean_costs=cost.mean(axis=0))

    show = ["policy", "accuracy", "quality", "cost_per_1k_queries", "cost_savings_vs_ref",
            "d_accuracy_vs_ref", "d_quality_vs_oracle", "reward@0.8"]
    show = [c for c in show if c in rep.columns]
    with pd.option_context("display.width", 200, "display.max_colwidth", 40,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(f"\n=== {label}: routing policies, {split} split ({true.shape[0]} queries x "
              f"{true.shape[1]} models, lam={args.lam}, ref={args.reference}) ===")
        print(rep[show].to_string(index=False))

    pr = pareto(preds["ZOIB  E[Y]"], true, cost)
    print(f"\n=== {label}: ZOIB E[Y] cost-aware frontier (lambda sweep) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(pr[["lam", "accuracy", "quality", "cost_per_1k_queries"]].to_string(index=False))
    frontier = aiq(pr, pool_mean_costs=cost.mean(axis=0),
                   pool_quality=[true_df[m].mean() for m in model_ids])
    print(f"\n{label}: ZOIB AIQ (area under cost/quality frontier): "
          f"absolute {frontier['aiq_absolute']:.4f}, "
          f"improvement over linear {frontier['aiq_improvement']:.4f}")

    return {
        "split": split, "label": label,
        "n_queries": int(true.shape[0]), "models": model_ids,
        "policies": rep.to_dict(orient="records"),
        "zoib_frontier": pr.to_dict(orient="records"),
        "zoib_aiq": frontier,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default=None, help="single eval split (legacy alias for --splits)")
    ap.add_argument("--splits", default=None,
                    help="comma list of eval splits; 'test' is in-distribution, 'ood' is "
                         "the held-out datasets (IRT-Router). Default: test")
    ap.add_argument("--nirt-run", default="nirt-2d-projected", help="a run under data/processed/nirt_runs/")
    ap.add_argument("--irt-run", default="irt-25d-projected",
                    help="the IRT-Router (model_latent) run to include as the paper baseline; "
                         "'none' to omit that row")
    ap.add_argument("--zoib", default="artifacts/phase2/zoib")
    ap.add_argument("--baseline", default="artifacts/phase1/baseline")
    ap.add_argument("--reference", default=DEFAULT_REFERENCE,
                    help="fixed model that cost savings are quoted against "
                         f"(default {DEFAULT_REFERENCE}; use 'gpt_4o' for IRT-Router)")
    ap.add_argument("--lam", type=float, default=0.0, help="cost weight for the headline table")
    ap.add_argument("--lcb-level", type=float, default=0.8, help="ZOIB lower-bound coverage for the LCB policy")
    ap.add_argument("--include-multishot", action="store_true",
                    help="keep ':5shot' items (default: 0-shot only, as in Phase 2)")
    ap.add_argument("--out", default=None, help="output JSON path (default: alongside --zoib parent)")
    args = ap.parse_args()

    splits_arg = args.splits or args.split or "test"
    splits = [s.strip() for s in splits_arg.split(",") if s.strip()]

    cfg = load_config(args.config)
    d = load_phase1(cfg)

    # Pin the routing pool to what the ZOIB checkpoint was trained on -- the
    # global NIRT dataset may since have grown (pool-expansion phases), and a
    # stale checkpoint has no predictions for the new columns.
    _zm, _zblob, _ = load_run(args.zoib)
    pool = sorted(_zblob["model_index"], key=_zblob["model_index"].get)

    labels = {"test": "ID (test)", "ood": "OOD (held-out datasets)",
              "validation": "validation"}
    results = {}
    for split in splits:
        payload = _run_split(split, labels.get(split, split), d, cfg, args, pool)
        results[split] = payload

    out = cfg.resolve(args.out) if args.out else (cfg.root / "artifacts/phase2/routing_comparison.json")
    if not args.out and "irt_router" in str(cfg.path("processed")):
        out = cfg.root / "artifacts/irt_router/routing_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": args.config, "lam": args.lam, "reference": args.reference,
        "splits": results,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
