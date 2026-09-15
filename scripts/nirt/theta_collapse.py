"""Is the router just routing on the per-model bias b_m, ignoring the query?

Loads a trained NIRTModel (query_latent, logit = a_m . theta_q - b_m) and checks
whether argmax_m is actually query-dependent, or has collapsed onto the constant
argmax_m(-b_m) -- i.e. "always route to the best-on-average model". Reports:

  * theta_q effective rank (PCA eigenvalues of the query-head output)
  * a_m spread across models, per latent dimension
  * agreement rate between the full router and the constant bias-only argmax
  * the top1-vs-top2 logit gap (how much query signal it'd take to flip it)

    python scripts/nirt/theta_collapse.py --run nirt-2d-projected --split test

Writes <runs_dir>/<run>/theta_collapse_<split>.json
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import yaml

from training.cli import raw_parser, resolve, write_json
from router.config import load_config
from router.nirt.checkpoint import load_run
from training.data.facade import load_training_data


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default=None)
    ap.add_argument("--nirt-config", default="configs/nirt.yaml")
    args = ap.parse_args()

    ncfg = yaml.safe_load(resolve(args.nirt_config).read_text(encoding="utf-8"))
    runs_dir = resolve(ncfg.get("runs_dir", "data/processed/nirt_runs"))

    model, cfg, midx = load_run(args.run, runs_dir=runs_dir)
    if model.orientation != "query_latent":
        sys.exit(f"{args.run} is orientation={model.orientation}; this diagnostic assumes query_latent")
    model.eval()

    d = load_training_data(load_config(args.config))
    pathway = (cfg.get("data", {}) or {}).get("pathway", "irt")
    q_store, m_store = d.query_embeddings(pathway), d.profile_embeddings(pathway)
    pool = sorted(midx, key=midx.get)
    query_ids = [q for q in d.split_query_ids(args.split) if q in q_store]

    Eq = torch.tensor(q_store.gather(query_ids), dtype=torch.float32)
    Em = torch.tensor(m_store.gather(pool), dtype=torch.float32)
    with torch.no_grad():
        theta_q = model.latent_query(Eq)
        a_m, b_m = model.model_parameters(Em)
        logit = theta_q @ a_m.T - b_m[None, :]

    argmax_full = logit.argmax(1)
    bias_only = int((-b_m).argmax())
    agree = float((argmax_full == bias_only).float().mean())
    top2 = logit.topk(2, dim=1).values
    gap = top2[:, 0] - top2[:, 1]

    tc = theta_q - theta_q.mean(0)
    ev = torch.linalg.eigvalsh((tc.T @ tc) / len(tc)).numpy()
    eff_rank = float((ev.sum() ** 2) / (ev ** 2).sum())   # 1 = a line, K = isotropic

    print(f"\n=== theta collapse check -- {args.run} / {args.split} ({len(query_ids):,} queries) ===")
    print(f"theta_q covariance eigenvalues: {np.round(ev, 4)}  "
          f"(effective rank {eff_rank:.2f} of {model.dim})")
    print("\na_m by model (should differ meaningfully across models if the router can reorder them):")
    for j, mid in enumerate(pool):
        print(f"  {mid:26s} a={a_m[j].numpy().round(3)}  b={b_m[j].item():+.3f}")
    print(f"\nconstant bias-only argmax (ignoring the query entirely): {pool[bias_only]!r}")
    print(f"the ACTUAL router agrees with that constant on {100 * agree:.1f}% of queries")
    print(f"top1-vs-top2 logit gap: mean {gap.mean():.3f}  median {gap.median():.3f}")
    print("  -> for the query term to flip the decision it must swing the gap past this on its own")

    out = {
        "run": args.run, "split": args.split, "n_queries": len(query_ids),
        "theta_q_eigenvalues": ev.tolist(), "theta_q_effective_rank": eff_rank,
        "constant_bias_argmax": pool[bias_only],
        "agreement_with_constant_bias_argmax": agree,
        "top1_top2_gap": {"mean": float(gap.mean()), "median": float(gap.median())},
        "a_m": {mid: a_m[j].tolist() for j, mid in enumerate(pool)},
        "b_m": {mid: float(b_m[j]) for j, mid in enumerate(pool)},
    }
    out_path = write_json(runs_dir / args.run / f"theta_collapse_{args.split}.json", out, announce=False)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
