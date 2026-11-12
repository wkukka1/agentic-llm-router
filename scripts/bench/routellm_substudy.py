"""Router bake-off, 2-model sub-study: gpt_4o (strong) vs gpt_4o_mini (weak).

RouteLLM's setting and RouteLLM's own metrics. Every router gives each query one
"strong score"; sending the top-k % of queries by score to gpt_4o and the rest to
gpt_4o_mini traces quality against the share of strong calls (k = 0..100 %).

* RouteLLM 0.2.0 (``scripts/bench/run_routellm.py``): pretrained BERT (Arena) and
  MF retrained on our data. Score = strong win rate.
* Our 20-model routers, restricted to the two columns: score = P(gpt_4o) - P(gpt_4o_mini)
  (cached by ``router_bakeoff.py``).
* llmrouter-lib routers at λ=0: score = S(gpt_4o) - S(gpt_4o_mini).
* References: random (score = noise), oracle (strong iff gpt_4o strictly beats mini).

Metrics (RouteLLM paper, Ong et al. 2024)
* PGR(k) = (q(k) - q_weak) / (q_strong - q_weak), the performance gap recovered.
* **APGR** = mean of PGR over k = 0..100 %, i.e. the area under the PGR curve. Random = 0.5.
* CPT(x) = the smallest strong-call share with PGR >= x (x = 50 %, 80 %).
Pooled over queries (RouteLLM's convention) plus a macro version: thresholds are
global, quality is averaged per dataset. Ties in a score are broken at random
(seeded), so a constant score behaves like random routing.
Paired bootstrap (queries resampled within dataset) for APGR differences.

    python scripts/bench/routellm_substudy.py --bundle data/processed/router_bakeoff/primary
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRONG, WEAK = "gpt_4o", "gpt_4o_mini"
GRID = np.linspace(0, 1, 101)


def curve(score: np.ndarray, gain: np.ndarray, qw: float, n: int, tie: np.ndarray) -> np.ndarray:
    """Mean quality at each strong share in GRID: top-k by score go strong."""
    order = np.lexsort((tie, -score))                # descending score, random tie-break
    cum = np.concatenate([[0.0], np.cumsum(gain[order])])
    k = np.round(GRID * n).astype(int)
    return qw + cum[k] / n


def macro_curve(score, gain, base, ds_codes, n_ds, tie) -> np.ndarray:
    """Global thresholds, per-dataset mean quality, averaged over datasets."""
    order = np.lexsort((tie, -score))
    n = len(score)
    k = np.round(GRID * n).astype(int)
    out = np.zeros(len(GRID))
    for g in range(n_ds):
        m = (ds_codes[order] == g)
        cum = np.concatenate([[0.0], np.cumsum(gain[order] * m)])
        out += (base[g] * m.sum() + cum[k]) / m.sum()
    return out / n_ds


def metrics(q: np.ndarray, qw: float, qs: float) -> dict:
    pgr = (q - qw) / (qs - qw)
    cpt = {x: (float(GRID[np.argmax(pgr >= x)]) if (pgr >= x).any() else np.nan) for x in (0.5, 0.8)}
    return {"apgr": float(np.trapezoid(pgr, GRID)), "cpt50": cpt[0.5], "cpt80": cpt[0.8]}


def load_scores(bundle: Path, split_q: pd.Index, seed: int) -> dict[str, np.ndarray]:
    sc: dict[str, np.ndarray] = {}
    for f in sorted(glob.glob(str(bundle / "routellm" / "*.parquet"))):
        r = pd.read_parquet(f)
        s = r.set_index("query_id")["strong_win_rate"].reindex(split_q)
        if s.notna().all():
            sc[f"routellm: {r['router'].iloc[0]}"] = s.to_numpy()
    return sc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default="data/processed/router_bakeoff/primary")
    ap.add_argument("--splits", default="test,ood")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--anchor", default="routellm: bert (pretrained, Arena)")
    ap.add_argument("--out", default="investigations/router_bakeoff")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    q = pd.read_parquet(bundle / "queries.parquet")
    obs = pd.read_parquet(bundle / "observations.parquet")
    obs = obs[obs["model_id"].isin([STRONG, WEAK])]
    T = obs.pivot(index="query_id", columns="model_id", values="target")
    Cc = obs.pivot(index="query_id", columns="model_id", values="cost")
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out) / bundle.name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, curves = [], []
    for split in args.splits.split(","):
        qq = q[q["split"] == split]
        qids = pd.Index(qq["query_id"])
        ds = qq["dataset"].to_numpy()
        ds_list = sorted(set(ds))
        ds_codes = np.array([ds_list.index(x) for x in ds])
        ts, tw = T.loc[qids, STRONG].to_numpy(), T.loc[qids, WEAK].to_numpy()
        gain = ts - tw
        n = len(qids)

        scores = load_scores(bundle, qids, args.seed)
        for f in sorted(glob.glob(str(bundle / "ours" / f"*__{split}.parquet"))):
            name = Path(f).name.rsplit("__", 1)[0].replace("_", " ")
            P = pd.read_parquet(f).reindex(qids)
            scores[f"ours: {name}"] = (P[STRONG] - P[WEAK]).to_numpy()
        for f in sorted(glob.glob(str(bundle / "llmrouter" / "*" / f"seed{args.seed}" / "lam0.npz"))):
            z = np.load(f)
            S = pd.DataFrame(z["scores"], index=z["query_ids"].astype(str),
                             columns=z["model_ids"].astype(str)).reindex(qids)
            d = (S[STRONG] - S[WEAK]).to_numpy()
            scores[f"llmrouter: {Path(f).parts[-3]}"] = np.where(np.isfinite(d), d, 0.0)
        scores["ref: oracle"] = np.sign(gain)                       # strong only when strictly better
        scores["ref: random"] = rng.random(n)

        tie = rng.random(n)
        base = np.array([tw[ds_codes == g].mean() for g in range(len(ds_list))])
        qw, qs = tw.mean(), ts.mean()
        mqw, mqs = base.mean(), np.mean([ts[ds_codes == g].mean() for g in range(len(ds_list))])
        cost_s, cost_w = Cc.loc[qids, STRONG].mean(), Cc.loc[qids, WEAK].mean()
        print(f"\n=== {split}: {n} queries; q_weak {qw:.4f}, q_strong {qs:.4f} "
              f"(gpt_4o better on {np.mean(gain > 0):.1%}, mini better on {np.mean(gain < 0):.1%}) ===")

        res = {}
        for nm, s in scores.items():
            c = curve(s, gain, qw, n, tie)
            mc = macro_curve(s, gain, base, ds_codes, len(ds_list), tie)
            m, mm = metrics(c, qw, qs), metrics(mc, mqw, mqs)
            res[nm] = m["apgr"], mm["apgr"]
            rows.append({"split": split, "router": nm, **m, **{f"macro_{k}": v for k, v in mm.items()},
                         "q@20%": float(c[20]), "q@50%": float(c[50])})
            for share, qv, mqv in zip(GRID, c, mc):
                curves.append({"split": split, "router": nm, "strong_share": share,
                               "quality": qv, "macro_quality": mqv,
                               "cost_per_1k": 1000 * (share * cost_s + (1 - share) * cost_w)})

        # paired bootstrap: APGR minus the anchor's, resampling within dataset
        if args.n_boot and args.anchor in scores:
            idx_by = [np.where(ds_codes == g)[0] for g in range(len(ds_list))]
            draws = {nm: [] for nm in scores}
            mdraws = {nm: [] for nm in scores}
            for _ in range(args.n_boot):
                ix = np.concatenate([rng.choice(i, len(i), replace=True) for i in idx_by])
                g_, tw_, ts_, dc_ = gain[ix], tw[ix], ts[ix], ds_codes[ix]
                b_ = np.array([tw_[dc_ == g].mean() for g in range(len(ds_list))])
                ms_ = np.mean([ts_[dc_ == g].mean() for g in range(len(ds_list))])
                vals = {}
                for nm, s in scores.items():
                    a = metrics(curve(s[ix], g_, tw_.mean(), len(ix), tie[ix]), tw_.mean(), ts_.mean())["apgr"]
                    b = metrics(macro_curve(s[ix], g_, b_, dc_, len(ds_list), tie[ix]), b_.mean(), ms_)["apgr"]
                    vals[nm] = (a, b)
                for nm in scores:
                    draws[nm].append(vals[nm][0] - vals[args.anchor][0])
                    mdraws[nm].append(vals[nm][1] - vals[args.anchor][1])
            for r in rows:
                if r["split"] == split:
                    lo, hi = np.quantile(draws[r["router"]], [0.025, 0.975])
                    mlo, mhi = np.quantile(mdraws[r["router"]], [0.025, 0.975])
                    r.update({"d_apgr_lo": lo, "d_apgr_hi": hi, "d_macro_apgr_lo": mlo, "d_macro_apgr_hi": mhi})

        t = pd.DataFrame([r for r in rows if r["split"] == split]).sort_values("apgr", ascending=False)
        show = [c for c in ["router", "apgr", "d_apgr_lo", "d_apgr_hi", "macro_apgr", "cpt50", "cpt80",
                            "q@20%", "q@50%"] if c in t.columns]
        print(t[show].round(4).to_string(index=False))

    pd.DataFrame(rows).to_csv(out_dir / f"routellm_summary_seed{args.seed}.csv", index=False)
    pd.DataFrame(curves).to_csv(out_dir / f"routellm_curves_seed{args.seed}.csv", index=False)
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
