"""Router bake-off: score every router on the same queries with routing metrics.

Step 1 of docs/router_benchmark_experiment.md. Reads the bundle written by
``export_bakeoff.py``, our routers' predictions (computed here, then cached
under ``<bundle>/ours/``) and the external routers' scores (written by
``run_llmrouter.py`` in its own venv), and reports decision quality, not BCE.

    python scripts/bench/router_bakeoff.py --config configs/irt_router.yaml \
        --bundle data/processed/router_bakeoff/primary --seed 0

Every router becomes a set of frontier points, one per cost weight lambda in
the shared grid:

* routers that predict per-model quality (ours): ``argmax_m pred - lam * C_m``,
  where ``C_m`` is the model's train-mean cost (the served decision rule,
  ``routing_decision``);
* llmrouter-lib routers: retrained per lambda on the utility label, then
  ``argmax_m score``.

Metrics per (router, split), all on the observed graded score (no BCE):

* ``aiq_log`` (headline) / ``aiq``: area under the quality-vs-cost frontier
  over the pool's cost range on a log / linear cost axis, for the row-weighted
  frontier and the macro (per-dataset mean) frontier;
* ``q@0.5%`` ... ``q@100%``: frontier quality at that share of the reference
  model's cost.

Each router's curve is the upper concave hull of its points plus the "always
the cheapest model" point (RouterBench's convention: any spend in between is
reachable by randomly mixing two operating points), so a router is never
credited with quality below the cheapest spend it can actually reach.
* at lambda=0: ``quality``, ``regret`` (oracle score - chosen score, mean and
  p90), ``any_best`` (chose a tied-best model);
* ranking quality for score routers: per-query Spearman / NDCG@3 against the
  observed scores.

Paired bootstrap (queries resampled within dataset) gives 95 % CIs for each
router's difference to ``--anchor``.

Outputs ``investigations/router_bakeoff/<bundle name>/``: ``points.csv``
(every frontier point), ``summary.csv`` and ``per_dataset.csv``.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from router.config import load_config
from router.nirt.routing_decision import routing_decision
from training.data.facade import load_training_data

LAMBDAS = (0.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
COST_FRACS = (0.005, 0.01, 0.02, 0.05, 1.0)

OUR_NIRT_RUNS = {
    "nirt (K=2, BERT profile)": "em-bert",
    "nirt elo-manifold + discrepancy": "em-manifold-delta",
    "irt-router K=25 (paper, our impl.)": "irtrouter-irt-25d-projected",
}
OUR_BASELINES = {"bernoulli": ("artifacts/irt_router/bernoulli", "proba"),
                 "zoib E[Y]": ("artifacts/irt_router/zoib", "mean")}


# --------------------------------------------------------------------------- #
# predictions                                                                  #
# --------------------------------------------------------------------------- #
def _our_matrix(name, kind, spec, d, cfg, split, qids, cache: Path) -> pd.DataFrame | None:
    f = cache / f"{name.replace(' ', '_').replace('/', '-')}__{split}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    if kind == "nirt":
        from router.nirt.checkpoint import load_run
        from router.nirt.predict import predict_matrix

        if not (cfg.resolve("data/processed/nirt_runs") / spec).exists():
            return None
        model, rc, midx = load_run(spec)
        dc = rc.get("data", {}) or {}
        m = predict_matrix(model, midx, d, split, pathway=dc.get("pathway", "irt"),
                           query_pathway=dc.get("query_pathway") or None,
                           query_features=dc.get("query_features") or None)
    elif kind == "baseline":
        from training.nirt.baseline.data import checkpoint_matrix

        path, field = spec
        if not cfg.resolve(path).exists():
            return None
        m = checkpoint_matrix(path, cfg, split, field)
    elif kind == "knn":
        from router.nirt.baselines_infer import knn_router_matrix

        m = knn_router_matrix(d, list(qids), k=spec, pathway="irt")
    elif kind == "mlp":
        from training.trainers.mlp_router import fit_mlp_router_as_router

        r = getattr(_our_matrix, "_mlp", None) or fit_mlp_router_as_router(d, pathway="irt")
        _our_matrix._mlp = r
        m = r.predict_scores(list(qids))
    else:
        raise ValueError(kind)
    m = m.reindex(index=list(qids))
    m.columns = [str(c) for c in m.columns]
    cache.mkdir(parents=True, exist_ok=True)
    m.to_parquet(f)
    return m


def _ext_scores(bundle: Path, seed: int) -> dict:
    """{router: {lam: DataFrame[query x model]}} from run_llmrouter.py outputs."""
    out: dict = {}
    files = [(f, "ext") for f in glob.glob(str(bundle / "llmrouter" / "*" / f"seed{seed}" / "lam*.npz"))]
    files += [(f, "litellm") for f in glob.glob(str(bundle / "litellm" / "*" / f"seed{seed}" / "lam*.npz"))]
    for f, src in sorted(files):
        kind = Path(f).parts[-3]
        lam = float(Path(f).stem[3:])
        z = np.load(f)
        out.setdefault(f"{src}: {kind}", {})[lam] = pd.DataFrame(
            z["scores"], index=z["query_ids"].astype(str), columns=z["model_ids"].astype(str))
    return out


# --------------------------------------------------------------------------- #
# metrics                                                                      #
# --------------------------------------------------------------------------- #
def frontier(points: np.ndarray, anchor) -> tuple:
    """Achievable quality-vs-cost curve: the upper concave hull of the router's
    (cost, quality) points plus ``anchor`` = (cost, quality) of always routing to
    the pool's cheapest model, made non-decreasing. Every point on it is
    achievable by randomly mixing two adjacent operating points (RouterBench's
    convention), and nothing is credited below what the router can reach.
    Returns the hull vertices ``(xs, ys)``; evaluate with ``np.interp``."""
    p = np.vstack([points, np.asarray(anchor, float)[None]])
    p = p[np.lexsort((-p[:, 1], p[:, 0]))]
    hull: list = []
    for x, y in p:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0:   # hull[-1] under the chord
                hull.pop()
            else:
                break
        hull.append((x, y))
    xs, ys = np.array(hull).T
    ys = np.maximum.accumulate(ys)
    return xs, ys


def aiq(points, anchor, lo, hi, n=400, log=False) -> float:
    """Mean frontier quality over the pool's cost range [lo, hi], on a linear or
    log(cost) axis. ``log=True`` needs ``lo`` > 0 (pass the cheapest non-zero
    model cost). On this pool the strongest model, deepseek_coder, costs ~3.6 %
    of gpt_4o, so on a linear axis ~96 % of the range is a plateau and AIQ ~= max
    quality; the log axis weights each decade of spend equally."""
    xs, ys = frontier(points, anchor)
    grid = np.geomspace(lo, hi, n) if log else np.linspace(lo, hi, n)
    q = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
    x = np.log(grid) if log else grid
    return float(np.trapezoid(q, x) / (x[-1] - x[0]))


def quality_at(points, anchor, cost) -> float:
    """Best achievable quality at a cost budget (mixing allowed)."""
    xs, ys = frontier(points, anchor)
    return float(np.interp(cost, xs, ys, left=ys[0], right=ys[-1]))


def _spearman_ndcg(pred: np.ndarray, true: np.ndarray, k: int = 3):
    from scipy.stats import rankdata

    rho, nd = [], []
    for p, t in zip(pred, true):
        if np.ptp(t) == 0 or not np.isfinite(p).any():
            continue
        p = np.where(np.isfinite(p), p, np.nanmin(p[np.isfinite(p)]) - 1)
        rp, rt = rankdata(p), rankdata(t)
        rho.append(np.corrcoef(rp, rt)[0, 1])
        top = np.argsort(-p)[:k]
        ideal = np.sort(t)[::-1][:k]
        disc = 1 / np.log2(np.arange(2, k + 2))
        nd.append(float((t[top] * disc).sum() / max((ideal * disc).sum(), 1e-12)))
    return float(np.nanmean(rho)), float(np.mean(nd))


def evaluate_split(policies: dict, true: pd.DataFrame, cost: pd.DataFrame, datasets: np.ndarray,
                   ref_cost: float, n_boot: int, anchor_router: str, rng) -> tuple:
    """policies: {name: {lam: choice index array}} -> (points, summary, per_dataset)."""
    T, C = true.to_numpy(np.float64), cost.to_numpy(np.float64)
    n = len(T)
    qi = np.arange(n)
    oracle = T.max(axis=1)
    pool_cost = C.mean(axis=0)
    lo, hi = float(pool_cost.min()), float(pool_cost.max())
    lo_pos = float(pool_cost[pool_cost > 0].min())
    cheap = int(np.argmin(pool_cost))
    anchor = (pool_cost[cheap], T[:, cheap].mean())
    ds_list = sorted(set(datasets))
    ds_idx = {g: np.where(datasets == g)[0] for g in ds_list}
    mac_anchor = (np.mean([C[ix, cheap].mean() for ix in ds_idx.values()]),
                  np.mean([T[ix, cheap].mean() for ix in ds_idx.values()]))

    chosen_q = {nm: {lam: T[qi, ch] for lam, ch in pol.items()} for nm, pol in policies.items()}
    chosen_c = {nm: {lam: C[qi, ch] for lam, ch in pol.items()} for nm, pol in policies.items()}

    def stats(nm, idx_by_ds):
        """Row-weighted and macro frontiers for one (possibly resampled) query set."""
        lams = sorted(chosen_q[nm])
        allidx = np.concatenate(list(idx_by_ds.values()))
        pts = np.array([[chosen_c[nm][l][allidx].mean(), chosen_q[nm][l][allidx].mean()] for l in lams])
        mac = np.array([[np.mean([chosen_c[nm][l][ix].mean() for ix in idx_by_ds.values()]),
                         np.mean([chosen_q[nm][l][ix].mean() for ix in idx_by_ds.values()])]
                        for l in lams])
        return pts, mac

    points, rows, per_ds = [], [], []
    for nm, pol in policies.items():
        pts, mac = stats(nm, ds_idx)
        for (c, q), (mc, mq), lam in zip(pts, mac, sorted(pol)):
            points.append({"router": nm, "lam": lam, "cost": c, "quality": q,
                           "macro_cost": mc, "macro_quality": mq})
        lam0 = min(pol)
        reg = oracle - chosen_q[nm][lam0]
        r = {"router": nm, "aiq_log": aiq(pts, anchor, lo_pos, hi, log=True),
             "macro_aiq_log": aiq(mac, mac_anchor, lo_pos, hi, log=True),
             "aiq": aiq(pts, anchor, lo, hi), "macro_aiq": aiq(mac, mac_anchor, lo, hi),
             **{f"q@{f * 100:g}%": quality_at(pts, anchor, f * ref_cost) for f in COST_FRACS},
             **{f"macro_q@{f * 100:g}%": quality_at(mac, mac_anchor, f * ref_cost) for f in COST_FRACS},
             "quality_lam0": float(chosen_q[nm][lam0].mean()),
             "cost_lam0": float(chosen_c[nm][lam0].mean()),
             "regret_mean": float(reg.mean()), "regret_p90": float(np.quantile(reg, 0.9)),
             "any_best": float((reg <= 1e-9).mean()), "n_points": len(pol)}
        rows.append(r)
        for g, ix in ds_idx.items():
            per_ds.append({"router": nm, "dataset": g, "n": len(ix),
                           "quality_lam0": float(chosen_q[nm][lam0][ix].mean()),
                           "cost_lam0": float(chosen_c[nm][lam0][ix].mean()),
                           "regret_lam0": float(reg[ix].mean()),
                           "aiq_log": aiq(np.array([[chosen_c[nm][l][ix].mean(), chosen_q[nm][l][ix].mean()]
                                                    for l in sorted(pol)]), (C[ix, cheap].mean(), T[ix, cheap].mean()),
                                               lo_pos, hi, log=True)})

    # paired bootstrap of differences to the anchor, resampling queries within dataset
    if n_boot and anchor_router in policies:
        names = list(policies)
        keys = ["aiq_log", "macro_aiq_log", "q@1%", "macro_q@1%"]
        draws = {nm: {k: [] for k in keys} for nm in names}
        for _ in range(n_boot):
            res = {g: rng.choice(ix, size=len(ix), replace=True) for g, ix in ds_idx.items()}
            allr = np.concatenate(list(res.values()))
            b_anchor = (C[allr, cheap].mean(), T[allr, cheap].mean())
            b_mac = (np.mean([C[ix, cheap].mean() for ix in res.values()]),
                     np.mean([T[ix, cheap].mean() for ix in res.values()]))
            vals = {}
            for nm in names:
                pts, mac = stats(nm, res)
                vals[nm] = {"aiq_log": aiq(pts, b_anchor, lo_pos, hi, 120, log=True),
                            "macro_aiq_log": aiq(mac, b_mac, lo_pos, hi, 120, log=True),
                            "q@1%": quality_at(pts, b_anchor, 0.01 * ref_cost),
                            "macro_q@1%": quality_at(mac, b_mac, 0.01 * ref_cost)}
            for nm in names:
                for k in keys:
                    draws[nm][k].append(vals[nm][k] - vals[anchor_router][k])
        for r in rows:
            for k in keys:
                a = np.array(draws[r["router"]][k], dtype=float)
                a = a[np.isfinite(a)]
                if len(a):
                    r[f"d_{k}_vs_anchor_lo"], r[f"d_{k}_vs_anchor_hi"] = np.quantile(a, [0.025, 0.975])
                    r[f"d_{k}_vs_anchor_p_le0"] = float((a <= 0).mean())
    return pd.DataFrame(points), pd.DataFrame(rows), pd.DataFrame(per_ds)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/irt_router.yaml")
    ap.add_argument("--bundle", default="data/processed/router_bakeoff/primary")
    ap.add_argument("--splits", default="test,ood")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", default="gpt_4o")
    ap.add_argument("--anchor", default="nirt (K=2, BERT profile)")
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--out", default="investigations/router_bakeoff")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = load_training_data(cfg)
    bundle = cfg.resolve(args.bundle)
    queries = pd.read_parquet(bundle / "queries.parquet")
    obs = pd.read_parquet(bundle / "observations.parquet")
    models = pd.read_parquet(bundle / "models.parquet")
    model_ids = models["model_id"].tolist()
    train_cost = models.set_index("model_id")["train_mean_cost"].reindex(model_ids).to_numpy()
    train_q = obs[obs["query_id"].isin(queries.loc[queries["split"] == "train", "query_id"])]
    tq = train_q.merge(queries[["query_id", "dataset"]], on="query_id")
    best_single = tq.groupby("model_id")["target"].mean().idxmax()
    per_ds_best = tq.groupby(["dataset", "model_id"])["target"].mean().unstack().idxmax(axis=1)
    ext = _ext_scores(bundle, args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = cfg.resolve(args.out) / bundle.name
    out_dir.mkdir(parents=True, exist_ok=True)
    all_pts, all_sum, all_ds = [], [], []
    for split in args.splits.split(","):
        qs = queries[queries["split"] == split]
        o = obs[obs["query_id"].isin(qs["query_id"])]
        true = o.pivot(index="query_id", columns="model_id", values="target").reindex(columns=model_ids)
        cost = o.pivot(index="query_id", columns="model_id", values="cost").reindex(columns=model_ids)
        keep = true.notna().all(axis=1) & cost.notna().all(axis=1)
        true, cost = true[keep], cost[keep]
        qids = true.index.tolist()
        datasets = qs.set_index("query_id").loc[qids, "dataset"].to_numpy()
        n = len(qids)
        ref_cost = float(cost[args.reference].mean())

        policies: dict = {}
        ranking: dict = {}
        # reference policies
        T = true.to_numpy()
        C = cost.to_numpy()
        at_max = T >= T.max(axis=1, keepdims=True) - 1e-9
        # cost-aware oracle: per query argmax of the TRUE utility y - lam * c (realized
        # cost); at lam=0 the cheapest model with the best score
        oracle = {0.0: np.where(at_max, C, np.inf).argmin(axis=1)}
        for lam in LAMBDAS[1:]:
            U = T - lam * C
            oracle[lam] = np.where(U >= U.max(axis=1, keepdims=True) - 1e-12, C, np.inf).argmin(axis=1)
        policies["oracle (cheapest best)"] = oracle
        for m in (best_single, args.reference, model_ids[int(np.argmin(train_cost))]):
            policies[f"fixed: {m}"] = {0.0: np.full(n, model_ids.index(m))}
        policies["random"] = {0.0: rng.integers(0, len(model_ids), n)}
        pdb = [per_ds_best.get(g, best_single) for g in datasets]   # unseen dataset -> best single
        policies["per-dataset best (uses dataset label)"] = {0.0: np.array([model_ids.index(m) for m in pdb])}

        # ours: per-model quality predictions, swept over lambda
        ours = {**{k: ("nirt", v) for k, v in OUR_NIRT_RUNS.items()},
                **{k: ("baseline", v) for k, v in OUR_BASELINES.items()},
                f"knn (k={args.knn_k}, ours)": ("knn", args.knn_k),
                "mlp (ours)": ("mlp", None)}
        for nm, (kind, spec) in ours.items():
            m = _our_matrix(nm, kind, spec, d, cfg, split, qids, bundle / "ours")
            if m is None:
                print(f"[skip] {nm}: no checkpoint", flush=True)
                continue
            P = m.reindex(index=qids, columns=model_ids).to_numpy(np.float64)
            policies[nm] = {lam: routing_decision(P, lam=lam, model_costs=train_cost) for lam in LAMBDAS}
            ranking[nm] = P
        # external: retrained per lambda
        for nm, by_lam in ext.items():
            pol = {}
            for lam, S in by_lam.items():
                S = S.reindex(index=qids, columns=model_ids).to_numpy(np.float64)
                pol[lam] = np.nanargmax(np.where(np.isfinite(S), S, -np.inf), axis=1)
                if lam == 0.0:
                    ranking[nm] = S
            policies[nm] = pol

        pts, summ, per_ds = evaluate_split(policies, true, cost, datasets, ref_cost,
                                           args.n_boot, args.anchor, rng)
        for nm, P in ranking.items():
            rho, nd = _spearman_ndcg(P, T)
            summ.loc[summ["router"] == nm, ["spearman_lam0", "ndcg@3_lam0"]] = [rho, nd]
            if nm in ours and np.nanmin(P) >= 0 and np.nanmax(P) <= 1:
                summ.loc[summ["router"] == nm, "mse"] = float(np.nanmean((P - T) ** 2))
        for df in (pts, summ, per_ds):
            df.insert(0, "split", split)
        all_pts.append(pts); all_sum.append(summ); all_ds.append(per_ds)

        show = ["router", "macro_aiq_log", "aiq_log", "q@0.5%", "q@1%", "q@2%", "q@100%",
                "regret_mean", "any_best", "spearman_lam0"]
        print(f"\n=== {split}: {n} queries, {len(model_ids)} models, ref {args.reference} "
              f"cost {ref_cost * 1000:.3f}$/1k ===")
        print(summ.sort_values("macro_aiq_log", ascending=False)[show].round(4).to_string(index=False))

    pd.concat(all_pts).to_csv(out_dir / f"points_seed{args.seed}.csv", index=False)
    pd.concat(all_sum).to_csv(out_dir / f"summary_seed{args.seed}.csv", index=False)
    pd.concat(all_ds).to_csv(out_dir / f"per_dataset_seed{args.seed}.csv", index=False)
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
