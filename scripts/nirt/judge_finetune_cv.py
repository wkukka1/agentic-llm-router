"""Does fine-tuning the profile-manifold NIRT on judged gateway prompts help? Cross-validated.

The served manifold file (``manifold-nirt-claude``) never saw a labelled Claude 5.5
answer, so it predicts the tiers almost level (Opus - Haiku ~ +0.01) while the
LLM judge has them 0.18 apart. Here each variant is fine-tuned on 4/5 of the
graded prompts and predicts the held-out 1/5 (5 folds), so every prediction
scored is out-of-fold; the out-of-fold matrix then goes through the gateway's
lambda rule and ``router_bench.margin_sweep`` (savings vs Opus, vs mix, split-half pick).

Variants, all trained on the judge's score per (prompt, tier) with soft BCE:

* ``none``        -- the run as served (no fine-tuning)
* ``tiers``       -- only per-tier offsets on (a_m, b_m); query side and profiles frozen
* ``tiers+head``  -- the offsets plus the query head's last layer
* ``full``        -- every weight at a HIGH lr (1e-3, 10x the fine-tune lr) plus the offsets
* ``mlp+bias``    -- the shipped MLP default (proxy stand-ins) with a per-tier logit bias, for reference

    python scripts/nirt/judge_finetune_cv.py --graded ~/.cache/aar/routing-pool/multiturn/graded.jsonl \\
        --runs pm-ft-d1024-s42,pm-ft-d1024-s43,pm-ft-d1024-s44
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from evaluation.gateway import router_bench as rb
from evaluation.gateway.pool_eval import TIERS, lambda_pick, load_prompts
from training.nirt.judge_finetune import VARIANTS, fit_judged, predict_judged

LAMS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))




def bias_predict(p_tr, y_tr, p_te, *, steps: int = 300) -> np.ndarray:
    """A per-tier logit bias fitted on train prompts, applied to fixed predictions."""
    lt, lte = torch.from_numpy(_logit(p_tr)), torch.from_numpy(_logit(p_te))
    Y = torch.from_numpy(y_tr)
    db = torch.zeros(len(TIERS), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([db], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(lt + db, Y).backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(lte + db).numpy()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graded", action="append", required=True)
    ap.add_argument("--runs", default="pm-ft-d1024-s42,pm-ft-d1024-s43,pm-ft-d1024-s44")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split", choices=("group", "source"), default="group",
                    help="group: k folds over conversations (two turns of one conversation never straddle "
                         "train/test); source: leave one prompt source out (own, swe-chat, ...)")
    ap.add_argument("--steps", type=int, default=300, help="full-batch Adam steps per fold")
    ap.add_argument("--out", default="investigations/profile_manifold/judge_finetune_cv.json")
    args = ap.parse_args(argv)

    from aar_gateway.config import read_jsonl

    seen, prompts, group_of = set(), [], {}
    for g in args.graded:
        for r in read_jsonl(Path(g).expanduser()):    # the conversation a turn came from
            group_of.setdefault(r["prompt_hash"], r.get("group") or r["prompt_hash"])
        for p in load_prompts(Path(g).expanduser()):
            if p["prompt_hash"] not in seen:
                seen.add(p["prompt_hash"])
                prompts.append(p)
    texts = [p["route_text"] for p in prompts]
    S = np.stack([p["score"] for p in prompts])
    C = np.stack([p["cost"] for p in prompts])
    n = len(prompts)
    if args.split == "source":
        names = sorted({p["source"] for p in prompts})
        fold = np.array([names.index(p["source"]) for p in prompts])
        args.folds = len(names)
        print(f"{n} graded prompts, leave-one-source-out over {names}")
    else:
        groups = sorted({group_of[p["prompt_hash"]] for p in prompts})
        gfold = dict(zip(groups, np.random.default_rng(0).permutation(len(groups)) % args.folds, strict=True))
        fold = np.array([gfold[group_of[p["prompt_hash"]]] for p in prompts])
        print(f"{n} graded prompts in {len(groups)} conversations, {args.folds} folds by conversation")

    from router.routing.routers import NIRTRouter, _encode_texts

    preds: dict[str, np.ndarray] = {}
    data = None
    for run in [r.strip() for r in args.runs.split(",") if r.strip()]:
        router = NIRTRouter.from_run(run, data=data)
        data = router._data
        e_q = np.ascontiguousarray(_encode_texts(data, texts, router._query_pathway,
                                                 fallback_pathway=router._pathway,
                                                 cache=router._encoder_cache), dtype=np.float32)
        refs = router.resolve_pool_profile_matrix(list(TIERS))
        for variant in VARIANTS:
            oof = np.empty((n, len(TIERS)))
            for f in range(args.folds):
                tr, te = fold != f, fold == f
                fit = fit_judged(router._model, e_q[tr], S[tr], refs, variant=variant, steps=args.steps, seed=f)
                oof[te] = predict_judged(fit, e_q[te], refs)
            preds[f"{variant} [{run}]"] = oof
            print(f"  {run:18s} {variant:11s} done")

    # the seeds' logits averaged -- what the exported file serves (training.trainers.
    # nirt_router_export.ensemble_fold is this exactly)
    for variant in VARIANTS:
        per_seed = [p for k, p in preds.items() if k.startswith(f"{variant} [")]
        if len(per_seed) > 1:
            preds[f"{variant} [mean of {len(per_seed)} seeds]"] = 1 / (1 + np.exp(-np.mean(
                [_logit(p) for p in per_seed], axis=0)))

    from aar_gateway.routing import load_router_model

    from evaluation.gateway.zoib_replay import proxy_scores

    mlp = proxy_scores(load_router_model({"artifact": str(Path.home() / ".aar-gateway" / "router.npz")})
                       .predict_scores_text(texts))
    preds["mlp (no fit)"] = mlp
    oof = np.empty_like(mlp)
    for f in range(args.folds):
        tr, te = fold != f, fold == f
        oof[te] = bias_predict(mlp[tr], S[tr], mlp[te])
    preds["mlp+bias"] = oof

    rows, keys = [], {}
    for name, pred in preds.items():
        key = "cv:" + name
        keys[key] = name
        for lam in LAMS:   # margin_sweep reads `condition` as "margin <x>": x is lambda here
            rows += [{"router": key, "track": "margin", "condition": f"margin {lam:g}",
                      "prompt_hash": p["prompt_hash"], "tier": TIERS[k], "error": None}
                     for p, k in zip(prompts, lambda_pick(pred, C, lam), strict=True)]
    rb.ROUTERS.update({k: (v, "offline") for k, v in keys.items()})
    lines: list[str] = []
    sweep = rb.margin_sweep(prompts, rows, LAMS, list(keys), lines, f"all ({n} prompts, out-of-fold)")
    print("\n".join(ln.replace("cost aversion sweep (router.margin via x-aar-quality-margin)",
                               "lambda sweep, out-of-fold predictions").replace(" margin ", " lambda ")
                     for ln in lines))
    real_gap = S[:, 2] - S[:, 0]
    diag = {name: {"pred_opus_minus_haiku_mean": float((p[:, 2] - p[:, 0]).mean()),
                   "corr_with_real_gap": float(np.corrcoef(p[:, 2] - p[:, 0], real_gap)[0, 1]),
                   "brier": float(((p - S) ** 2).mean())}
            for name, p in preds.items()}
    print(f"\nreal Opus-Haiku mean gap {real_gap.mean():+.3f}")
    for name, g in diag.items():
        print(f"  {name:36s} predicted gap {g['pred_opus_minus_haiku_mean']:+.3f}  corr {g['corr_with_real_gap']:+.2f}"
              f"  brier {g['brier']:.4f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": n, "folds": args.folds, "steps": args.steps, "lams": LAMS,
                               "diagnostics": diag, "sweep": sweep}, indent=1, default=float), encoding="utf-8")
    print(f"\nresults: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
