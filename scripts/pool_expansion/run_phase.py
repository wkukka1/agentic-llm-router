"""Run the fixed pool-expansion battery (§1) for one phase and update the ledger.

    python scripts/pool_expansion/run_phase.py --phase E0
    python scripts/pool_expansion/run_phase.py --phase E1 --added-models yi-34b-chat code-llama-34b-instruct \
        --sources routerbench --notes "un-hold the two RouterBench cold-start models"

Writes artifacts/pool_expansion/<phase>/{battery,pool,prediction,routing,cold_start,ablation,provenance}.json,
appends a row to artifacts/pool_expansion/ledger.json, and re-renders
docs/pool_expansion_results.md.
"""

from __future__ import annotations

import argparse
import json
import sys

from router.config import load_config
from router.pool_expansion import run_battery


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, help="phase id, e.g. E0")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--zoib", default="artifacts/phase2/zoib")
    ap.add_argument("--bernoulli", default="artifacts/phase1/baseline")
    ap.add_argument("--zoib-projected", default="artifacts/phase2/zoib_projected")
    ap.add_argument("--nirt-run", default="nirt-2d-projected",
                    help="legacy Phase-0 NIRT run for the policy table; "
                         "pass 'none' to skip (e.g. once the pool no longer matches it)")
    ap.add_argument("--reference", default="gpt-4-1106-preview")
    ap.add_argument("--lcb-level", type=float, default=0.8)
    ap.add_argument("--added-models", nargs="*", default=[],
                    help="models this phase added to the pool (drives the §1f ablation)")
    ap.add_argument("--ablation-drop-query-suffix", default=None,
                    help="query-subset ablation: re-run routing with queries whose id "
                         "ends with this suffix removed (E2: ':5shot')")
    ap.add_argument("--sources", nargs="*", default=["routerbench"])
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    nirt_run = None if str(args.nirt_run).lower() in ("", "none") else args.nirt_run
    result = run_battery(
        args.phase, cfg=cfg, split=args.split, zoib=args.zoib, bernoulli=args.bernoulli,
        zoib_projected=args.zoib_projected, nirt_run=nirt_run, reference=args.reference,
        lcb_level=args.lcb_level, added_models=args.added_models, sources=args.sources,
        ablation_drop_query_suffix=args.ablation_drop_query_suffix,
        notes=args.notes, write=not args.no_write,
    )

    der = result["routing"]["derived"]
    pool = result["pool"]
    print(f"\n=== pool-expansion battery — {args.phase} ({args.split}) ===")
    print(f"  warm models              {pool['n_warm_models']}")
    print(f"  GPT-4 : cheapest ratio   {pool['gpt4_cheapest_cost_ratio']:.1f}x")
    print(f"  ZOIB test NLL / MAE      {result['prediction']['zoib']['nll_mean']:.4f} / "
          f"{result['prediction']['zoib']['mae']:.4f}")
    print(f"  oracle cost-save ceiling {der['oracle_cost_saving_ceiling']*100:+.1f}%  "
          f"(oracle routes {der['oracle_offbest_fraction']*100:.0f}% off {der['best_single_model']})")
    for tag, key in (("-1pt", "router_saving_at_minus1pt"), ("-3pt", "router_saving_at_minus3pt")):
        s = der.get(key)
        if s:
            print(f"  ZOIB saving @ {tag:>4}       {s['cost_saving_vs_ref']*100:+.1f}%  "
                  f"(acc {s['accuracy']:.4f}, d {s['d_accuracy_vs_best_single']*100:+.2f}pt, lam {s['lam']})")
        else:
            print(f"  ZOIB saving @ {tag:>4}       none")
    print(f"  ZOIB AIQ improvement     {result['routing']['zoib_aiq'].get('aiq_improvement'):+.4f}")
    cs = result["cold_start"]
    if cs.get("status") == "ran":
        print(f"  cold-start beats global-mean BCE?  {cs['beats_global_mean_on_bce']}")
    else:
        print(f"  cold-start: {cs.get('reason')}")
    abl = result["ablation"]
    if abl.get("status") == "ran" and abl.get("mode") == "query_subset":
        print(f"  ablation (drop {abl['removed_query_suffix']} queries): "
              f"ceiling {abl['reduced_oracle_cost_saving_ceiling']*100:+.1f}% "
              f"(d {abl['delta_oracle_cost_saving_ceiling']*100:+.1f}pp), "
              f"d -3pt saving {_g(abl['delta_zoib_saving_at_minus3pt'])}")
    elif abl.get("status") == "ran":
        print(f"  ablation (remove {', '.join(abl['added_models'])}): "
              f"d ceiling {abl['delta_oracle_cost_saving_ceiling']*100:+.1f}pp, "
              f"d -3pt saving {_g(abl['delta_zoib_saving_at_minus3pt'])}")
        for m, c in (abl.get("leave_one_out") or {}).items():
            print(f"    - {m:<26} d ceiling {c['delta_oracle_cost_saving_ceiling']*100:+.1f}pp, "
                  f"d -3pt saving {_g(c['delta_zoib_saving_at_minus3pt'])}")
    else:
        print(f"  ablation: {abl.get('reason')}")
    sb = result.get("shot_breakdown")
    if sb:
        print(f"  5-shot lift (mean d-acc {sb['mean_d_acc']*100:+.1f}pt over {sb['n_suffix_queries']} queries):")
        for r in sb["per_model"]:
            print(f"    - {r['model_id']:<26} 0shot {r['acc_base']:.3f} -> 5shot {r['acc_suffix']:.3f} "
                  f"({r['d_acc']*100:+.1f}pt)")
    print(f"\nwrote artifacts/pool_expansion/{args.phase}/  +  ledger.json  +  docs/pool_expansion_results.md")
    return 0


def _g(d):
    v = d.get("delta")
    return f"{v*100:+.1f}pp" if v is not None else "n/a"


if __name__ == "__main__":
    sys.exit(main())
