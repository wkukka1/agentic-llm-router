"""Cost-aware routing benchmark: ZOIB router (and optionally IRT-Router) vs always-GPT-4
vs the oracle, on the RouterBench + IRT-Router test pools and both combined.

Routers (each picks argmax_m [pred_quality - lam * pred_cost] in the query's own pool):

  ZOIB                 ZOIB E[Y]        - lam * C_hat(q, m)   (the checkpoint's cost head)
  IRT-Router           P(correct)       - lam * C(m)          (paper policy: per-model train-mean cost)
  IRT-Router + C_hat   P(correct)       - lam * C_hat(q, m)   (same cost model as ZOIB)
  ZOIB + C(m)          ZOIB E[Y]        - lam * C(m)          (isolates the cost model)

The last three need --irt-run. Scored on realized quality and realized per-query cost;
see evaluation.nirt.routing_benchmark for the reference / oracle policies.

    python scripts/nirt/baseline/routing_benchmark.py --zoib-ckpt artifacts/phase2/zoib_cost_length_v2
    python scripts/nirt/baseline/routing_benchmark.py --zoib-ckpt artifacts/phase2/zoib_cost_length_v2 \\
        --irt-run irt-25d-projected-29m --out artifacts/phase2/zoib_vs_irt_router_test.json

Writes <zoib-ckpt>/routing_benchmark_<split>.json unless --out is given.
"""

from __future__ import annotations

import sys

import numpy as np

from evaluation.nirt.routing import align, eval_matrices
from evaluation.nirt.routing_benchmark import (
    DEFAULT_LAMS,
    REFERENCE_MODELS,
    baseline_cell_predictions,
    combine,
    per_model_mean_cost,
    source_picks,
    summarize,
)
from router.nirt.frames import pivot_qm
from training.cli import get_phase0_config, raw_parser, resolve, write_json
from training.data.facade import load_training_data


def _print_tables(name: str, s: dict, n: int, routers: list[str], lams) -> None:
    ref = s["reference"]
    print(f"\n## {name}: {n} test queries")
    print(f"Always reference model: quality {ref['quality']:.3f} at ${ref['cost_per_1k']:.2f} per 1k queries")
    for k, label in (("oracle_match_reference", "Oracle, cheapest model matching the reference"),
                     ("oracle_best_quality", "Oracle, best quality")):
        o = s[k]
        print(f"{label}: {o['quality']:.3f} ({o['q_vs_reference']:+.3f}) at "
              f"${o['cost_per_1k']:.2f} ({o['cost_vs_reference']:+.0%})")
    for r in routers:
        print(f"\n### {r}\n")
        print("| lam | Quality (vs ref) | Cost per 1k (vs ref) | Share to ref | Oracle@lam quality / cost |")
        print("|---|---|---|---|---|")
        for lam in lams:
            p, o = s[f"{r}@{lam:g}"], s[f"oracle@{lam:g}"]
            print(f"| {lam:g} | {p['quality']:.3f} ({p['q_vs_reference']:+.3f}) "
                  f"| ${p['cost_per_1k']:.2f} ({p['cost_vs_reference']:+.0%}) "
                  f"| {p['reference_share']:.0%} | {o['quality']:.3f} / ${o['cost_per_1k']:.2f} |")


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--zoib-ckpt", required=True, help="BaselineNIRT checkpoint dir with a cost head")
    ap.add_argument("--irt-run", default=None,
                    help="NIRT/IRT-Router run name under runs_dir (e.g. irt-25d-projected-29m)")
    ap.add_argument("--nirt-runs", default="",
                    help="comma list of further NIRT runs, each routed as P(correct) - lam * C(m) "
                         "(the IRT-Router policy)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--lams", default=",".join(f"{x:g}" for x in DEFAULT_LAMS),
                    help="comma list of lambda values (USD-per-query cost weight)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--phase0-config", default=None)
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]

    cfg = get_phase0_config(args)
    d = load_training_data(cfg)
    ckpt = resolve(args.zoib_ckpt, cfg.root)
    long, _ = baseline_cell_predictions(ckpt, cfg, d, args.split)
    def _run_pred(run):
        from router.nirt.checkpoint import load_run
        from router.nirt.predict import predict_matrix

        model, run_cfg, midx = load_run(run)
        return predict_matrix(model, midx, d, args.split,
                              pathway=run_cfg.get("data", {}).get("pathway", "irt"))

    irt_pred = _run_pred(args.irt_run) if args.irt_run else None
    nirt_preds = {r: _run_pred(r) for r in (x.strip() for x in args.nirt_runs.split(",")) if r}
    c_model = per_model_mean_cost(d.nirt_observations())

    by_source, sizes = {}, {}
    for src, ref in REFERENCE_MODELS.items():
        true_df, cost_df = eval_matrices(d, args.split, models=sorted(long["model_id"].unique()),
                                         source=src)
        models = list(true_df.columns)
        zq = align(pivot_qm(long, "quality_pred"), true_df)
        zc = align(pivot_qm(long, "cost_pred"), true_df)
        Y, C = true_df.to_numpy(np.float64), cost_df.to_numpy(np.float64)
        mats = [zq, zc, Y, C]
        if irt_pred is not None:
            iq = align(irt_pred, true_df)
            mats.append(iq)
        nq = {r: align(p, true_df) for r, p in nirt_preds.items()}
        mats += list(nq.values())
        keep = np.logical_and.reduce([np.isfinite(m).all(1) for m in mats])
        zq, zc, Y, C = zq[keep], zc[keep], Y[keep], C[keep]
        routers = {"ZOIB": (zq, zc)}
        Cm = c_model.reindex(models).to_numpy(np.float64)
        if irt_pred is not None:
            iq = iq[keep]
            routers |= {"IRT-Router": (iq, Cm), "IRT-Router + C_hat": (iq, zc), "ZOIB + C(m)": (zq, Cm)}
        routers |= {r: (p[keep], Cm) for r, p in nq.items()}
        by_source[src] = source_picks(Y, C, models, ref, routers, lams)
        sizes[src] = {"n_queries": int(keep.sum()), "n_dropped": int((~keep).sum()),
                      "n_models": len(models), "reference": ref}

    results = {src: {**sizes[src], "policies": summarize(p)} for src, p in by_source.items()}
    results["combined"] = {"n_queries": sum(v["n_queries"] for v in sizes.values()),
                           "reference": " | ".join(REFERENCE_MODELS.values()),
                           "policies": summarize(combine(by_source))}
    router_names = list(routers)
    for name in ("combined", *REFERENCE_MODELS):
        _print_tables(name, results[name]["policies"], results[name]["n_queries"], router_names, lams)

    out = args.out or ckpt / f"routing_benchmark_{args.split}.json"
    write_json(resolve(out, cfg.root), {"zoib_ckpt": str(ckpt), "irt_run": args.irt_run,
                                        "nirt_runs": list(nirt_preds),
                                        "split": args.split, "lams": lams, "results": results})
    return 0


if __name__ == "__main__":
    sys.exit(main())
