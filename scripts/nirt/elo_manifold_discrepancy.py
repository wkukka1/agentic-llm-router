"""Where the public numbers are wrong: per-model discrepancy of an Elo-manifold NIRT run.

For a run trained with configs/nirt_elo_manifold.yaml (``model.discrepancy``),
reports per pool model, on the validation queries:

* ``manifold_p``  -- mean P(correct) from the manifold heads alone (z_m -> a_m, b_m)
* ``fitted_p``    -- mean P(correct) with the learned discrepancy added
* ``observed``    -- the model's actual mean score
* ``delta_logit`` -- mean logit shift the discrepancy applies

A large |delta| means the response data disagrees with where Arena / Open LLM
Leaderboard put the model.

    python scripts/nirt/elo_manifold_discrepancy.py --run em-manifold-delta \
        --phase0-config configs/irt_router.yaml
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from router.config import load_config
from router.nirt.checkpoint import load_run
from router.nirt.predict import predict_dataset
from training.data.facade import load_training_data
from training.data.nirt import nirt_dataset_from_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="em-manifold-delta")
    ap.add_argument("--phase0-config", default="configs/irt_router.yaml")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default="investigations/elo_manifold/discrepancy.csv")
    args = ap.parse_args()

    p0 = load_config(args.phase0_config)
    loaded = load_run(args.run)
    model, cfg, model_index = loaded
    if not getattr(model, "discrepancy", False):
        raise SystemExit(f"run {args.run} has no per-model discrepancy (model.discrepancy)")
    dcfg = cfg.get("data", {})
    d = load_training_data(p0)
    ds = nirt_dataset_from_config(p0, data=d, split=args.split, pathway=dcfg.get("pathway"),
                                  query_pathway=dcfg.get("query_pathway"))
    y, p_fit = predict_dataset(model, ds, model_index)
    y, p_man = predict_dataset(model, ds, {})          # no index -> manifold only
    mids = np.asarray(ds.model_ids).astype(str)
    logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7)) - np.log1p(-np.clip(p, 1e-7, 1 - 1e-7))
    df = pd.DataFrame({"model_id": mids, "y": y, "manifold_p": p_man, "fitted_p": p_fit,
                       "delta_logit": logit(p_fit) - logit(p_man)})
    out = df.groupby("model_id").agg(observed=("y", "mean"), manifold_p=("manifold_p", "mean"),
                                     fitted_p=("fitted_p", "mean"),
                                     delta_logit=("delta_logit", "mean"))
    out["manifold_err"] = out["manifold_p"] - out["observed"]
    out["fitted_err"] = out["fitted_p"] - out["observed"]
    out = out.sort_values("delta_logit")
    path = p0.resolve(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(path)
    print(out.round(3).to_string())
    print(f"\nmean |manifold_err| {out['manifold_err'].abs().mean():.3f}   "
          f"mean |fitted_err| {out['fitted_err'].abs().mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
