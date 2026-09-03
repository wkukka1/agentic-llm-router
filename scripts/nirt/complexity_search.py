"""Random architecture search for the NIRT (query_latent) head on the
kNN-imputed query representation.

Explores latent width `K` (theta_q / a_m in R^K), the query-head MLP width, and
**multidimensional difficulty** (`model.difficulty="vector"` -> b_m in R^K,
logit = sum_k a_k (theta_k - b_k), vs the scalar intercept). Samples N random
configs, trains each on `data.query_pathway` (default knn10w), and tables test
prediction / ranking / routing metrics + the train->test BCE gap against the
K=2 scalar baseline.

    python scripts/nirt/complexity_search.py --n 15
    python scripts/nirt/complexity_search.py --n 20 --query-pathway knn10w --seed 1

Writes artifacts/phase2/complexity_search.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from router.config import load_config
from router.data.phase1 import load_phase1
from router.nirt.evaluate import predict_dataset, predict_matrix, ranking_metrics
from router.nirt.metrics import prediction_metrics
from router.nirt.routing import align, eval_matrices
from router.nirt.routing_eval import routing_evaluation
from router.nirt.train import fit, load_run

SPACE = {
    "dim": [4, 8, 16, 32, 48, 64],
    "query_hidden": [None, 64, 128, 256],
    "difficulty": ["scalar", "vector"],
    "lr": [1.0e-3, 2.0e-3],
    "weight_decay": [1.0e-5, 1.0e-4],
}
_KEEP = ("bce", "brier", "auc", "acc@0.5", "spearman_r")


def _sample(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    seen, out = set(), []
    tries = 0
    while len(out) < n and tries < n * 200:
        tries += 1
        c = {k: rng.choice(v) for k, v in SPACE.items()}
        key = tuple(sorted(c.items(), key=lambda kv: kv[0]))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _nirt_cfg(base: dict, combo: dict, query_pathway: str | None) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg.setdefault("model", {})
    cfg["model"].update(
        orientation="query_latent", model_params="projected",
        dim=int(combo["dim"]), difficulty=combo["difficulty"],
        query_hidden=("none" if combo["query_hidden"] is None else int(combo["query_hidden"])),
        constrain_discrimination="auto",
    )
    cfg.setdefault("train", {})
    cfg["train"].update(lr=float(combo["lr"]), weight_decay=float(combo["weight_decay"]))
    cfg.setdefault("data", {})["query_pathway"] = query_pathway
    return cfg


def _zeroshot_mask(ids) -> np.ndarray:
    return ~pd.Series(ids).astype(str).str.endswith(":5shot").to_numpy()


def _evaluate(model, midx, d, true_df, cost_df, pw, qp) -> dict:
    pred = align(predict_matrix(model, midx, d, "test", pathway=pw, query_pathway=qp), true_df)
    true, cost = true_df.to_numpy(np.float64), cost_df.to_numpy(np.float64)
    ok = np.isfinite(pred).all(axis=1)
    t, p, c = true[ok], pred[ok], cost[ok]
    pm = prediction_metrics(t.ravel(), p.ravel())
    rm = ranking_metrics(p, t)
    re = routing_evaluation(p, t, c, list(true_df.columns), lam=0.0)

    tr = d.nirt_dataset(split="train", pathway=pw, query_pathway=qp)
    y, pr = predict_dataset(model, tr, midx)
    m0 = _zeroshot_mask(tr.query_ids)
    train_bce = prediction_metrics(y[m0], pr[m0])["bce"]
    return {
        **{k: pm.get(k) for k in _KEEP},
        "train_bce": train_bce, "gap_bce": pm["bce"] - train_bce,
        "optimal_rate": rm["optimal_rate"], "regret": rm["regret"],
        "oracle_hit_any_best": re["oracle_hit_rate_any_best"],
        "n_test": int(ok.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/nirt.yaml")
    ap.add_argument("--phase0-config", default=None)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--query-pathway", default="knn10w", help="'none' for the raw irt store")
    ap.add_argument("--baseline-run", default="nirt-knn10w-2d-projected")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--out", default="artifacts/phase2/complexity_search.json")
    args = ap.parse_args()

    root = Path(load_config().root)
    cfg_path = Path(args.config)
    base = yaml.safe_load((cfg_path if cfg_path.is_absolute() else root / cfg_path).read_text("utf-8"))
    p0 = load_config(args.phase0_config) if args.phase0_config else load_config()
    qp = None if str(args.query_pathway).lower() == "none" else args.query_pathway
    d = load_phase1(p0)
    true_df, cost_df = eval_matrices(d, split="test")
    keep = [not str(q).endswith(":5shot") for q in true_df.index]
    true_df, cost_df = true_df.loc[keep], cost_df.loc[keep]

    combos = _sample(args.n, args.seed)
    rows: list[dict] = []

    # reference: the existing K=2 scalar baseline
    try:
        bm, bcfg, bmi = load_run(args.baseline_run)
        bqp = (bcfg.get("data", {}) or {}).get("query_pathway") or None
        rows.append({"run": args.baseline_run, "dim": 2, "query_hidden": 64,
                     "difficulty": "scalar", "lr": 1e-3, "weight_decay": 1e-5,
                     "params": sum(p.numel() for p in bm.parameters()),
                     **_evaluate(bm, bmi, d, true_df, cost_df, "irt", bqp)})
    except FileNotFoundError:
        print(f"[cx] baseline run '{args.baseline_run}' not found; skipping reference row")

    for i, combo in enumerate(combos):
        h = "none" if combo["query_hidden"] is None else combo["query_hidden"]
        name = f"nirt-cx-K{combo['dim']}h{h}{'v' if combo['difficulty']=='vector' else 's'}" \
               f"-lr{combo['lr']:g}wd{combo['weight_decay']:g}"
        print(f"\n[cx {i+1}/{len(combos)}] {name}")
        cfg = _nirt_cfg(base, combo, qp)
        if not args.retrain:
            try:
                model, rcfg, midx = load_run(name)
                print("  [reuse]")
            except FileNotFoundError:
                res = fit(cfg, phase0_cfg=p0, name=name, verbose=False)
                model, midx = res.model, res.model_index
        else:
            res = fit(cfg, phase0_cfg=p0, name=name, verbose=False)
            model, midx = res.model, res.model_index

        row = {"run": name, **{k: combo[k] for k in SPACE},
               "params": sum(p.numel() for p in model.parameters()),
               **_evaluate(model, midx, d, true_df, cost_df, "irt", qp)}
        rows.append(row)
        print(f"  test bce {row['bce']:.4f}  gap {row['gap_bce']:.4f}  "
              f"auc {row['auc']:.4f}  regret {row['regret']:.4f}")

    df = pd.DataFrame(rows).sort_values("bce").reset_index(drop=True)
    show = ["run", "dim", "query_hidden", "difficulty", "lr", "weight_decay", "params",
            "train_bce", "bce", "gap_bce", "auc", "spearman_r", "optimal_rate", "regret",
            "oracle_hit_any_best"]
    with pd.option_context("display.width", 240, "display.max_colwidth", 44,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print("\n=== NIRT complexity search (test, sorted by BCE) ===")
        print(df[[c for c in show if c in df]].to_string(index=False))

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "query_pathway": args.query_pathway, "n": args.n, "seed": args.seed,
        "space": {k: [str(x) for x in v] for k, v in SPACE.items()},
        "rows": rows,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
