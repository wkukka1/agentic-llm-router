"""Projected NIRT runs on the aar-gateway graded prompts, scoring the Claude tiers
from their OWN profiles (no stand-in benchmark models).

Each run scores every graded prompt's route text against the three Claude tiers
via ``NIRTRouter.predict_pool_scores_text`` -- the tier's curated profile in
``configs/model_profiles.yaml`` rendered and embedded with the run's own encoder,
so a profile-manifold run's encoder places the tier on its manifold. The
gateway's margin rule (cheapest tier within ``margin`` of the best prediction)
turns scores into picks; ``evaluation.gateway.router_bench.margin_sweep`` reports
quality, savings vs always-Opus with a paired CI, and a split-half margin pick.

For context the two shipped gateway router files (MLP default, llmrouterbench-nirt)
are replayed too -- they CAN'T score the tiers directly, so they use the gateway's
``proxy`` stand-ins and are labelled as such.

    python scripts/nirt/gateway_replay.py --graded ~/.aar-gateway/eval-rules/graded.jsonl \\
        --graded ~/.aar-gateway/eval40/graded.jsonl \\
        --runs nirt-2d-projected,pm-ft-d1024-s42 --out investigations/profile_manifold/gateway_replay.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from evaluation.gateway import router_bench as rb
from evaluation.gateway.pool_eval import TIERS, lambda_pick, load_prompts, margin_pick

MARGINS = (0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
LAMS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graded", action="append", required=True)
    ap.add_argument("--runs", required=True, help="comma list of projected NIRT runs under runs_dir")
    ap.add_argument("--group-seeds", action="store_true",
                    help="average runs that differ only in -s<seed> into one router (mean prediction)")
    ap.add_argument("--no-proxy-refs", action="store_true", help="skip the shipped proxy-based router files")
    ap.add_argument("--files", default="",
                    help="comma list of gateway router files (.npz) to replay through the gateway's own "
                         "loader; a file with Claude-tier columns is read directly, else via proxies")
    ap.add_argument("--lams", default=",".join(f"{x:g}" for x in LAMS),
                    help="lambda values on relative cost to sweep ('' to skip)")
    ap.add_argument("--out", default="investigations/profile_manifold/gateway_replay.json")
    args = ap.parse_args(argv)

    seen, prompts = set(), []
    for g in args.graded:
        for p in load_prompts(Path(g).expanduser()):
            if p["prompt_hash"] not in seen:
                seen.add(p["prompt_hash"])
                prompts.append(p)
    texts = [p["route_text"] for p in prompts]
    print(f"{len(prompts)} graded prompts with all three tiers")

    from router.routing.routers import NIRTRouter

    preds: dict[str, np.ndarray] = {}
    data = None
    for run in [r.strip() for r in args.runs.split(",") if r.strip()]:
        router = NIRTRouter.from_run(run, data=data)
        data = router._data                       # load the training data once, share it
        pred = router.predict_pool_scores_text(texts, list(TIERS))[list(TIERS)].to_numpy(float)
        key = re.sub(r"-s\d+(?=-|$)", "", run) if args.group_seeds else run
        preds.setdefault(key, []).append(pred)
    preds = {k: np.mean(v, axis=0) for k, v in preds.items()}

    for f in [x.strip() for x in args.files.split(",") if x.strip()]:
        from aar_gateway.routing import load_router_model

        frame = load_router_model({"artifact": str(Path(f).expanduser())}).predict_scores_text(texts)
        if all(t in frame.columns for t in TIERS):
            preds[f"{Path(f).stem} (file)"] = frame[list(TIERS)].to_numpy(float)
        else:
            from evaluation.gateway.zoib_replay import proxy_scores

            preds[f"{Path(f).stem} (file, proxy)"] = proxy_scores(frame)

    if not args.no_proxy_refs:
        from aar_gateway.routing import load_router_model

        from evaluation.gateway.zoib_replay import proxy_scores

        for name, f in (("MLP default (proxy)", Path.home() / ".aar-gateway" / "router.npz"),
                        ("llmrouterbench-nirt (proxy)", rb.AAR_ROUTER_FILES["aar-nirt"])):
            if Path(f).exists():
                preds[name] = proxy_scores(load_router_model({"artifact": str(f)}).predict_scores_text(texts))

    rows, labels = [], {}
    for name, pred in preds.items():
        key = "replay:" + name
        labels[key] = name
        for m in MARGINS:
            pick = margin_pick(pred, m)
            rows += [{"router": key, "track": "margin", "condition": f"margin {m:g}",
                      "prompt_hash": p["prompt_hash"], "tier": TIERS[k], "error": None}
                     for p, k in zip(prompts, pick, strict=True)]
    rb.ROUTERS.update({k: (v, "offline") for k, v in labels.items()})
    lines: list[str] = []
    sweep = rb.margin_sweep(prompts, rows, MARGINS, list(labels), lines, f"all ({len(prompts)} prompts)")
    lams = [float(x) for x in args.lams.split(",") if x.strip()]
    lam_sweep = {}
    if lams:
        cost = np.stack([p["cost"] for p in prompts])
        lrows, lkeys = [], {}
        for name, pred in preds.items():
            key = "lambda:" + name
            lkeys[key] = name
            for lam in lams:   # margin_sweep reads `condition` as "margin <x>": x is lambda here
                lrows += [{"router": key, "track": "margin", "condition": f"margin {lam:g}",
                           "prompt_hash": p["prompt_hash"], "tier": TIERS[k], "error": None}
                          for p, k in zip(prompts, lambda_pick(pred, cost, lam), strict=True)]
        rb.ROUTERS.update({k: (v, "offline") for k, v in lkeys.items()})
        llines: list[str] = []
        lam_sweep = rb.margin_sweep(prompts, lrows, lams, list(lkeys), llines, f"all ({len(prompts)} prompts)")
        lines += [ln.replace("cost aversion sweep (router.margin via x-aar-quality-margin)",
                             "lambda sweep (quality - lambda x cost / Opus cost)").replace(" margin ", " lambda ")
                  for ln in llines]
    real = np.stack([q["score"] for q in prompts])
    gaps = {name: {"opus_minus_haiku_median": float(np.median(p[:, 2] - p[:, 0])),
                   "opus_minus_sonnet_median": float(np.median(p[:, 2] - p[:, 1])),
                   "corr_with_real_opus_minus_haiku": float(np.corrcoef(p[:, 2] - p[:, 0],
                                                                        real[:, 2] - real[:, 0])[0, 1])}
            for name, p in preds.items()}
    print("\n".join(lines))
    print("\npredicted tier gaps:")
    for name, g in gaps.items():
        print(f"  {name:32s} Opus-Haiku median {g['opus_minus_haiku_median']:+.3f}  Opus-Sonnet median "
              f"{g['opus_minus_sonnet_median']:+.3f}  corr with real Opus-Haiku gap "
              f"{g['corr_with_real_opus_minus_haiku']:+.2f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_prompts": len(prompts), "margins": MARGINS, "gaps": gaps, "sweep": sweep,
                               "lams": lams, "lambda_sweep": lam_sweep},
                              indent=1, default=float), encoding="utf-8")
    print(f"\nresults: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
