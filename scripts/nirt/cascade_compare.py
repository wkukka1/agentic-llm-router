"""Learned routers vs hardness-threshold policies and hardness-gated cascades.

Every policy is traced as a cost/quality frontier on the eval split:

* ``route:<predictor>``   -- argmax(pred - lam * C(m)) over a wide log ``lam`` grid
  (lam < ~5 is a no-op at these per-query costs, so the usual 0..5 grid is not
  enough to trace a frontier).
* ``threshold:<signal>`` -- rank queries by a hardness score; the easiest
  fraction ``f`` go to the cheap model, the rest to the strong one (no execution
  feedback).
* ``cascade:<signal>:v=<acc>`` -- the easiest fraction ``f`` TRY the cheap model;
  a verifier of accuracy ``v`` checks it, and a rejected answer escalates to the
  strong model (both calls paid). ``f = 1`` is "always cascade".
* ``chain3:<signal>:v=<acc>`` -- the same with a cheap -> mid -> strong ladder; two
  thresholds pick each query's entry rung.

Hardness signals (query-only at decision time, except ``oracle``):
``bern_pool`` / ``nirt_pool`` = 1 - mean_m P(correct) under that predictor,
``bern_cheap`` = 1 - P(cheap model correct), ``knn`` = mean difficulty
(1 - mean_m score) of the k nearest TRAIN queries (BERT query embeddings,
model-free), ``oracle`` = the true 1 - mean_m score (a ceiling, not a policy).
The verifier is simulated: it accepts a correct answer w.p. ``v`` and a wrong
one w.p. ``1 - v`` -- ``v = 1`` is a perfect checker (upper bound).

    python scripts/nirt/cascade_compare.py --config configs/irt_router.yaml --splits test,ood
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from evaluation.nirt.routing import align, aiq, eval_matrices, route, train_quality
from router.config import load_config
from training.cli import float_table, raw_parser
from training.data.facade import load_training_data
from training.nirt.baseline.data import checkpoint_matrix

LAMS = np.concatenate([[0.0], np.logspace(0, 5, 31)])
FRACS = np.round(np.linspace(0.0, 1.0, 21), 3)


def _nirt(run, d, split):
    from router.nirt.checkpoint import load_run
    from router.nirt.predict import predict_matrix

    model, cfg, midx = load_run(run)
    dc = cfg.get("data", {}) or {}
    return predict_matrix(model, midx, d, split, pathway=dc.get("pathway", "irt"),
                          query_pathway=dc.get("query_pathway") or None,
                          query_features=dc.get("query_features") or None)


def _knn_hardness(d, eval_ids, k=20, chunk=2048) -> np.ndarray:
    """Mean difficulty of the k most cosine-similar train queries."""
    obs = d.nirt_observations()
    tr = obs[obs["split"] == "train"]
    diff = 1.0 - tr.groupby("query_id")["target"].mean()
    store = d.query_embeddings("irt")
    idx = store._index
    tr_ids = [q for q in diff.index if q in idx]
    T = np.asarray(store.matrix[[idx[q] for q in tr_ids]], dtype=np.float32)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    dv = diff.reindex(tr_ids).to_numpy(np.float64)
    E = np.asarray(store.matrix[[idx[q] for q in eval_ids]], dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    out = np.empty(len(eval_ids))
    for i in range(0, len(E), chunk):
        S = E[i:i + chunk] @ T.T
        top = np.argpartition(-S, k, axis=1)[:, :k]
        w = np.clip(np.take_along_axis(S, top, 1), 0, None) + 1e-6
        out[i:i + chunk] = (w * dv[top]).sum(1) / w.sum(1)
    return out


def _row(policy, family, knob, q, c) -> dict:
    return {"policy": policy, "family": family, "knob": knob, "quality": float(q.mean()),
            "accuracy": float((q >= 0.5).mean()), "cost_per_1k": float(c.mean() * 1000)}


def _entry(h, fracs) -> np.ndarray:
    """Rung index per query: rung j for queries whose hardness rank falls in
    [fracs[j-1], fracs[j]) (fracs ascending, len = rungs - 1)."""
    r = np.argsort(np.argsort(h, kind="stable"), kind="stable") / len(h)
    return np.searchsorted(np.asarray(fracs), r, side="right")


def _ladder(true, cost, rungs, entry, v, rng_u, verify_cost=None):
    """Execute the ladder from each query's entry rung; the verifier accepts a
    rung's answer with prob v if correct, 1-v if wrong; the top rung is final.
    ``verify_cost`` (per-query vector) is paid for every non-top rung checked."""
    n = len(true)
    qi = np.arange(n)
    q = np.zeros(n)
    c = np.zeros(n)
    done = np.zeros(n, bool)
    for j, m in enumerate(rungs):
        here = (~done) & (entry <= j)
        c[here] += cost[here, m]
        if verify_cost is not None and j < len(rungs) - 1:
            c[here] += verify_cost[here]
        correct = true[:, m] >= 0.5
        accept = np.where(correct, rng_u[:, j] < v, rng_u[:, j] >= v)
        stop = here & (accept | (j == len(rungs) - 1))
        q[stop] = true[stop, m]
        done |= stop
    assert done.all()
    del qi
    return q, c


def _min_cost_at(df, acc) -> float:
    ok = df[df["accuracy"] >= acc]
    return float(ok["cost_per_1k"].min()) if len(ok) else float("nan")


def _max_acc_at(df, cost) -> float:
    ok = df[df["cost_per_1k"] <= cost + 1e-12]
    return float(ok["accuracy"].max()) if len(ok) else float("nan")


def run_split(split, d, cfg, args, pool) -> pd.DataFrame:
    true_df, cost_df = eval_matrices(d, split=split, models=pool)
    ids = list(true_df.columns)
    true, cost = true_df.to_numpy(np.float64), cost_df.to_numpy(np.float64)
    n = len(true)

    tq = train_quality(d, ids, metric="accuracy")
    mean_c = cost.mean(0)
    cheap = int(np.argmin(mean_c))
    strong = ids.index(max(tq, key=tq.get))
    mids = [j for j in range(len(ids)) if j not in (cheap, strong) and mean_c[j] < 0.25 * mean_c[strong]]
    mid = max(mids, key=lambda j: tq[ids[j]])
    print(f"\n[{split}] {n} queries; cheap={ids[cheap]}, mid={ids[mid]}, strong={ids[strong]}")

    preds = {
        "bernoulli": align(checkpoint_matrix(args.baseline, cfg, split, "proba"), true_df),
        "zoib": align(checkpoint_matrix(args.zoib, cfg, split, "mean"), true_df),
        "nirt_bert": align(_nirt(args.nirt_run, d, split), true_df),
        "nirt_elo_k3": align(_nirt(args.elo_run, d, split), true_df),
        "irt_router": align(_nirt(args.irt_run, d, split), true_df),
    }
    rows = []
    for name, P in preds.items():
        for lam in LAMS:
            ch = route(P, cost, lam)
            rows.append(_row(f"route:{name}", "router", float(lam),
                             true[np.arange(n), ch], cost[np.arange(n), ch]))
    for j, label in ((cheap, "cheap"), (mid, "mid"), (strong, "strong")):
        rows.append(_row(f"fixed:{ids[j]}", f"fixed_{label}", 0.0, true[:, j], cost[:, j]))

    hard = {
        "bern_pool": 1.0 - preds["bernoulli"].mean(1),
        "nirt_pool": 1.0 - preds["nirt_bert"].mean(1),
        "bern_cheap": 1.0 - preds["bernoulli"][:, cheap],
        "knn": _knn_hardness(d, list(true_df.index), k=args.knn_k),
        "oracle": 1.0 - true.mean(1),
    }
    hard = {k: v for k, v in hard.items() if k in args.signals}
    # a judge call priced like the verifier model answering the same query
    # (same prompt in, a short verdict out -- an upper-ish estimate)
    vcost = None if args.verifier_model == "none" else cost[:, ids.index(args.verifier_model)]
    rng_u = np.random.default_rng(args.seed).random((n, 3))
    for sig, h in hard.items():
        for f in FRACS:
            e = _entry(h, [f])                       # 0 -> cheap, 1 -> strong
            m = np.where(e == 0, cheap, strong)
            rows.append(_row(f"threshold:{sig}", "threshold", float(f),
                             true[np.arange(n), m], cost[np.arange(n), m]))
            for v in args.verifier:
                q, c = _ladder(true, cost, [cheap, strong], e, v, rng_u, vcost)
                rows.append(_row(f"cascade:{sig}:v={v}", "cascade", float(f), q, c))
        for v in args.verifier:
            for f1 in FRACS[::2]:
                for f2 in FRACS[::2]:
                    if f2 < f1:
                        continue
                    q, c = _ladder(true, cost, [cheap, mid, strong], _entry(h, [f1, f2]), v, rng_u, vcost)
                    rows.append(_row(f"chain3:{sig}:v={v}", "chain3", float(f1 * 100 + f2), q, c))

    oracle_ch = true.argmax(1)
    df = pd.DataFrame(rows)
    df.insert(0, "split", split)

    # -- summary: one line per policy --------------------------------------
    s_acc = float((true[:, strong] >= 0.5).mean())
    s_cost = float(cost[:, strong].mean() * 1000)
    targets = [s_acc, 0.85, 0.90]          # exact: a rounded-up target excludes the strong model itself
    summ = []
    pool_q = [true[:, j].mean() for j in range(len(ids))]
    for pol, g in df.groupby("policy", sort=False):
        r = {"policy": pol, "best_accuracy": g["accuracy"].max(),
             f"acc@cost<={s_cost:.3f}": _max_acc_at(g, s_cost)}
        for t in targets:
            r[f"cost@acc>={t:.4f}"] = _min_cost_at(g, t - 1e-12)
        if len(g) > 1:
            pr = g.rename(columns={"cost_per_1k": "cost"}).assign(cost=lambda x: x["cost"] / 1000)
            r["aiq_improvement"] = aiq(pr, pool_mean_costs=mean_c, pool_quality=pool_q)["aiq_improvement"]
        summ.append(r)
    summ = pd.DataFrame(summ)
    with float_table(250, **{"display.max_colwidth": 40}):
        print(f"[{split}] strong fixed model {ids[strong]}: accuracy {s_acc:.4f}, "
              f"cost/1k {s_cost:.4f}; oracle accuracy {float((true.max(1) >= 0.5).mean()):.4f}")
        print(summ.round(4).to_string(index=False))
    del oracle_ch
    return df, summ


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--config", default="configs/irt_router.yaml")
    ap.add_argument("--splits", default="test,ood")
    ap.add_argument("--baseline", default="artifacts/irt_router/bernoulli")
    ap.add_argument("--zoib", default="artifacts/irt_router/zoib")
    ap.add_argument("--nirt-run", default="irtrouter-nirt-2d-projected")
    ap.add_argument("--elo-run", default="irtrouter-nirt-elo-k3")
    ap.add_argument("--irt-run", default="irtrouter-irt-25d-projected")
    ap.add_argument("--verifier", default="1.0,0.9,0.8",
                    help="comma list of simulated verifier accuracies")
    ap.add_argument("--verifier-model", default="none",
                    help="pool model whose per-query cost is charged for each verification "
                         "(e.g. glm_4_flash, gpt_4o_mini); 'none' = free verifier")
    ap.add_argument("--signals", default="bern_pool,nirt_pool,bern_cheap,knn,oracle")
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="investigations/cascade")
    args = ap.parse_args()
    args.verifier = [float(v) for v in args.verifier.split(",")]
    args.signals = [s.strip() for s in args.signals.split(",") if s.strip()]

    cfg = load_config(args.config)
    d = load_training_data(cfg)
    pool = sorted(d.nirt_observations()["model_id"].astype(str).unique())
    out = cfg.resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames, summaries = [], {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        df, summ = run_split(split, d, cfg, args, pool)
        frames.append(df)
        summaries[split] = summ.to_dict(orient="records")
    pd.concat(frames).to_csv(out / "frontiers.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summaries, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {out / 'frontiers.csv'} and summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
