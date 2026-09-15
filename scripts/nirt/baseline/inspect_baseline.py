"""Inspect the learned Phase 1 parameters for individual model / query pairs.

    python scripts/inspect_baseline.py --checkpoint artifacts/phase1/baseline
    python scripts/inspect_baseline.py --query-id routerbench:mmlu:xxxx --model-id gpt-4-1106-preview

model -> theta_m ; query -> a_q / b_q / r_q ; parameter distributions ; and
P(correct | q, m) with the base IRT score for a chosen pair.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from training.cli import raw_parser
from router.config import load_config
from training.nirt.baseline.checkpoint import load_run
from training.nirt.baseline.data import batched_forward, build_arrays
from training.nirt.baseline.diagnostics import parameter_summary
from training.nirt.baseline.eval import theta_matrix


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--checkpoint", default="artifacts/phase1/baseline")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--query-id", default=None)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, blob, s = load_run(args.checkpoint)
    mi = blob["model_index"]
    order = sorted(mi, key=mi.get)

    arr = build_arrays(cfg, split=args.split, pathway=s["pathway"],
                       binary_threshold=s["binary_threshold"], score_kind=s["score_kind"],
                       use_relevance=model.use_relevance, use_warmup=model.use_warmup, model_index=mi)
    theta = theta_matrix(model, cfg, mi, s["pathway"])
    fwd = batched_forward(model, arr, fields=("proba", "a_q", "b_q"))
    a_all, b_all, prob = fwd["a_q"], fwd["b_q"], fwd["proba"]

    print("=== model -> theta_m ===")
    for m, t in zip(order, theta):
        print(f"  {m:28s} |theta|={np.linalg.norm(t):.3f}  theta={np.round(t, 3).tolist()}")

    print("\n=== parameter distributions ===")
    print(json.dumps(parameter_summary(theta=theta, a_q=a_all, b_q=b_all,
                                       discrimination_constrained=bool(model.disc_head.constrain)),
                     indent=2, default=float))

    seen = {}
    for i, q in enumerate(arr.query_ids):
        seen.setdefault(q, i)
    idx = np.fromiter(seen.values(), dtype=int)
    ordered = idx[np.argsort(b_all[idx])]
    print("\n=== easiest / hardest queries (b_q) ===")
    for tag, sel in (("easiest", ordered[:args.top]), ("hardest", ordered[-args.top:])):
        for i in sel:
            print(f"  [{tag:8s}] b={b_all[i]:+.3f} |a|={np.linalg.norm(a_all[i]):.3f} {arr.query_ids[i]}")

    if args.query_id and args.model_id:
        j = mi.get(args.model_id)
        rows = np.where(arr.query_ids == args.query_id)[0]
        if j is None or len(rows) == 0:
            print(f"\n(no observation for {args.query_id} x {args.model_id})")
        else:
            r = rows[0]
            print(f"\n=== P(correct | q, m) for {args.query_id} x {args.model_id} ===")
            print(f"  a_q = {np.round(a_all[r], 3).tolist()}")
            print(f"  b_q = {b_all[r]:+.3f}")
            if model.use_relevance and arr.r_q is not None:
                top = np.argsort(arr.r_q[r])[::-1][:5]
                print(f"  r_q top clusters = {[(int(c), round(float(arr.r_q[r][c]), 3)) for c in top]}")
            print(f"  theta_m = {np.round(theta[j], 3).tolist()}")
            print(f"  base IRT score = {float(a_all[r] @ theta[j] - b_all[r]):+.3f}")
            print(f"  predicted P(correct) = {prob[r]:.4f}   observed y = {arr.y[r]:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
