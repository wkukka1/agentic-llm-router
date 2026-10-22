"""Grid search for the Elo-prior NIRT: training hyperparameters x routing lambda.

Trains every combination of ``train.elo_prior.weight`` x ``model.dim`` x seed
(configs/nirt_elo_prior.yaml; weight 0 also turns ``model.elo_prior`` off), plus
a no-d_q baseline, then sweeps the routing cost weight ``lam`` on each run's
predicted [query x model] matrix.

Selection is on the VALIDATION split only; test (ID) and ood are reported at the
selected point. Two scores per run:

* ``aiq``          -- area under the cost/quality frontier over the lam grid
  (lam-free; :func:`evaluation.nirt.routing.aiq`).
* ``reward@a``     -- IRT-Router's ``a * quality - (1-a) * linear(cost)`` at each
  lam; the best lam per (config, a) is picked on validation.

``lam`` is in raw USD per query: this pool's mean costs span 0 -- 0.0018, so
lam < ~1 is a no-op and the grid is log-spaced up to 10000.

    python scripts/nirt/elo_grid_search.py
    python scripts/nirt/elo_grid_search.py --weights 0,1 --dims 2 --seeds 42

Resumable: runs already under runs_dir/<prefix>-* are reused, not retrained.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd
import yaml

from router.config import load_config
from router.nirt.checkpoint import load_run
from router.nirt.predict import predict_matrix
from training.data.facade import load_training_data
from training.models.manifold import arena_ratings
from training.nirt.train import fit
from evaluation.nirt.routing import (
    _policy_row, add_reward_columns, aiq, align, eval_matrices, oracle_choice, route,
)

LAMS = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
ALPHAS = (0.5, 0.7, 0.8, 0.9)
SPLITS = ("validation", "test", "ood")


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _configs(weights, dims, seeds, dq_baseline: bool):
    out = []
    for w, k, seed in itertools.product(weights, dims, seeds):
        out.append({"weight": w, "dim": int(k), "seed": int(seed), "query_difficulty": True})
    if dq_baseline:
        for seed in seeds:
            out.append({"weight": 0.0, "dim": 2, "seed": int(seed), "query_difficulty": False})
    return out


def _name(prefix, c) -> str:
    dq = "" if c["query_difficulty"] else "-nodq"
    return f"{prefix}-w{c['weight']:g}-k{c['dim']}{dq}-s{c['seed']}"


def _train(base_cfg, c, name, ratings, runs_dir, p0) -> None:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["seed"] = c["seed"]
    cfg["model"]["dim"] = c["dim"]
    cfg["model"]["query_difficulty"] = c["query_difficulty"]
    cfg["model"]["elo_prior"] = c["weight"] > 0
    cfg["train"]["device"] = "auto"
    cfg["train"]["elo_prior"] = {"weight": c["weight"], "ratings": ratings if c["weight"] > 0 else None}
    fit(cfg, phase0_cfg=p0, name=name, runs_dir=runs_dir, verbose=False)


def _sweep(pred, true, cost) -> pd.DataFrame:
    """One row per lam: quality / accuracy / cost / reward@alpha."""
    rows = []
    for lam in LAMS:
        pick = route(pred, cost, lam)
        r = _policy_row(f"lam={lam:g}", pick, true, cost)
        r["lam"] = lam
        r["n_models_used"] = int(len(np.unique(pick)))
        rows.append(r)
    return add_reward_columns(pd.DataFrame(rows), pool_mean_costs=cost.mean(axis=0), alphas=ALPHAS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase0-config", default="configs/irt_router.yaml")
    ap.add_argument("--config", default="configs/nirt_elo_prior.yaml")
    ap.add_argument("--weights", default="0,0.1,1,10")
    ap.add_argument("--dims", default="1,2,4")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--no-dq-baseline", action="store_true")
    ap.add_argument("--prefix", default="grid-elo")
    ap.add_argument("--out", default="investigations/elo_manifold/grid")
    args = ap.parse_args()

    p0 = load_config(args.phase0_config)
    base_cfg = yaml.safe_load(p0.resolve(args.config).read_text(encoding="utf-8"))
    runs_dir = p0.resolve(base_cfg.get("runs_dir", "data/processed/nirt_runs"))
    ratings = arena_ratings(p0, (base_cfg["train"].get("elo_prior") or {}).get(
        "manifold", "configs/model_manifold.yaml"))
    configs = _configs(_floats(args.weights), _floats(args.dims), _floats(args.seeds),
                       not args.no_dq_baseline)

    # -- train ------------------------------------------------------------
    for i, c in enumerate(configs):
        name = _name(args.prefix, c)
        if (runs_dir / name / "model.pt").exists():
            continue
        print(f"[grid] train {i + 1}/{len(configs)} {name}", flush=True)
        _train(base_cfg, c, name, ratings, runs_dir, p0)

    # -- evaluate ---------------------------------------------------------
    d = load_training_data(p0)
    mats = {s: eval_matrices(d, split=s) for s in SPLITS}
    rows, refs = [], []
    for s, (true_df, cost_df) in mats.items():
        true, cost = true_df.to_numpy(np.float64), cost_df.to_numpy(np.float64)
        ids = list(true_df.columns)
        ref = pd.DataFrame([_policy_row(label, pick, true, cost) for label, pick in (
            ("oracle", oracle_choice(true, cost, model_ids=ids)),
            ("cheapest", np.full(len(true), int(np.argmin(cost.mean(0))))),
            ("gpt_4o", np.full(len(true), ids.index("gpt_4o"))),
            ("deepseek_coder", np.full(len(true), ids.index("deepseek_coder"))),
        )])
        refs.append(add_reward_columns(ref, pool_mean_costs=cost.mean(axis=0), alphas=ALPHAS)
                    .assign(split=s))
    refs = pd.concat(refs, ignore_index=True)

    for c in configs:
        name = _name(args.prefix, c)
        model, rcfg, midx = load_run(name, runs_dir=runs_dir)
        dcfg = rcfg.get("data", {}) or {}
        for s, (true_df, cost_df) in mats.items():
            pred = align(predict_matrix(model, midx, d, s, pathway=dcfg.get("pathway", "irt"),
                                        query_pathway=dcfg.get("query_pathway") or None), true_df)
            true, cost = true_df.to_numpy(np.float64), cost_df.to_numpy(np.float64)
            sw = _sweep(pred, true, cost)
            fr = aiq(sw, pool_mean_costs=cost.mean(axis=0), pool_quality=true.mean(axis=0))
            for r in sw.to_dict("records"):
                rows.append({"run": name, **c, "split": s, **r, **fr})
        print(f"[grid] evaluated {name}", flush=True)

    df = pd.DataFrame(rows)
    out = p0.resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sweep.csv", index=False)
    refs.to_csv(out / "references.csv", index=False)

    # -- summaries --------------------------------------------------------
    keys = ["weight", "dim", "query_difficulty"]
    val = df[df["split"] == "validation"]
    # lam-free: AIQ per config (mean +- std over seeds), every split
    aiq_tbl = (df.drop_duplicates(["run", "split"])
               .groupby(keys + ["split"])["aiq_improvement"].agg(["mean", "std"]).unstack("split"))
    print("\n=== AIQ improvement over linear (lam-free; mean/std over seeds) ===")
    print(aiq_tbl.round(4).to_string())

    # per alpha: pick (config, lam) by mean validation reward over seeds,
    # then report that same point on test / ood
    summary = []
    for a in ALPHAS:
        col = f"reward@{a}"
        g = val.groupby(keys + ["lam"])[col].mean()
        best = g.idxmax()
        hit = np.ones(len(df), bool)
        for k, v in zip(keys + ["lam"], best):
            hit &= (df[k] == v).to_numpy()
        sel = df[hit]
        for s in SPLITS:
            part = sel[sel["split"] == s]
            summary.append({"alpha": a, **dict(zip(keys + ["lam"], best)), "split": s,
                            "reward": part[col].mean(), "reward_std": part[col].std(),
                            "quality": part["quality"].mean(), "cost": part["cost"].mean(),
                            "n_models_used": part["n_models_used"].mean()})
    summ = pd.DataFrame(summary)
    summ.to_csv(out / "selected.csv", index=False)
    print("\n=== fixed / oracle references ===")
    print(refs[["split", "policy", "quality", "cost"] + [f"reward@{a}" for a in ALPHAS]]
          .round(5).to_string(index=False))
    print("\n=== best (config, lam) per alpha, chosen on validation ===")
    print(summ.round(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
