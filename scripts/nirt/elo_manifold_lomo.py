"""Strict leave-one-model-out (LOMO) zero-shot cold start: Elo ability prior vs BERT profile.

For every pool model m: drop ALL of m's observations, retrain NIRT on the
other 19, then score m zero-shot on the test (ID) and ood splits from its
public representation alone:

* ``eloprior``  -- configs/nirt_elo_prior.yaml (Elo as a prior on ability,
  per-query difficulty d_q) trained on the other 19. m has no data, so its
  ability is the prior mean ``alpha + beta * elo_m`` when Arena rates it, and
  its BERT-profile ability otherwise; the per-query terms (a_m, theta_q, d_q)
  are unchanged.
* ``bert``      -- m's BERT profile-text embedding (configs/nirt.yaml, the
  ``irtrouter-nirt-2d-projected`` recipe).
* ``warm_mean`` -- m predicted as the per-query mean of the 19 warm models
  (the documented zero-shot floor), from the same fold's model.

Reports per held-out model: BCE/AUC on m's own rows, and the routing regret
(lam=0) over the full 20-model pool with m's column replaced by the cold
prediction.

    python scripts/nirt/elo_manifold_lomo.py --phase0-config configs/irt_router.yaml
    python scripts/nirt/elo_manifold_lomo.py --phase0-config configs/irt_router.yaml \
        --methods eloprior --models gpt_4o,qwq_32b_preview
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import yaml

from router.config import load_config
from router.nirt.predict import predict_dataset
from training.data.facade import load_training_data
from training.data.nirt import nirt_dataset_from_config
from training.data.public_benchmarks import fetch_arena_category_elo
from training.models.manifold import build_feature_matrix
from training.nirt.metrics import prediction_metrics
from training.nirt.train import fit

METHODS = {
    "eloprior": "configs/nirt_elo_prior.yaml",
    "bert": "configs/nirt.yaml",
}


def _elo_ability_shift(model, d, pathway, held_out, ratings) -> float:
    """Logit shift that swaps the held-out model's BERT-profile ability for its
    Elo prior mean (0 when Arena does not rate it). Elo is standardized over the
    TRAINED models' ratings, exactly as the trainer did."""
    import torch

    if held_out not in ratings:
        return 0.0
    rated = np.array([v for m, v in ratings.items() if m != held_out])
    z = (ratings[held_out] - rated.mean()) / rated.std()
    e_m = np.asarray(d.profile_embeddings(pathway).get(held_out), dtype=np.float32)
    with torch.no_grad():
        ability = float(model.ability(torch.from_numpy(e_m[None]))[0])
        prior = float(model.elo_prior_mean(torch.tensor(z, dtype=torch.float32)))
    return prior - ability


def _scores(model, midx, p0, d, obs, split, pathway, query_pathway,
            held_out=None, shift=0.0) -> pd.DataFrame:
    ds = nirt_dataset_from_config(p0, data=d, split=split, pathway=pathway,
                                  query_pathway=query_pathway, observations=obs)
    y, p = predict_dataset(model, ds, midx)
    mids = np.asarray(ds.model_ids).astype(str)
    if shift:
        rows = mids == held_out
        pr = np.clip(p[rows], 1e-7, 1 - 1e-7)
        p = p.copy()
        p[rows] = 1.0 / (1.0 + np.exp(-(np.log(pr) - np.log1p(-pr) + shift)))
    return pd.DataFrame({"query_id": ds.query_ids, "model_id": mids, "y": y, "p": p})


def _evaluate(frame: pd.DataFrame, held_out: str) -> dict:
    Y = frame.pivot(index="query_id", columns="model_id", values="y")
    P = frame.pivot(index="query_id", columns="model_id", values="p")
    ok = Y.notna().all(axis=1) & P.notna().all(axis=1)
    Y, P = Y[ok], P[ok]
    others = [c for c in P.columns if c != held_out]
    warm = P.copy()
    warm[held_out] = P[others].mean(axis=1)
    y_m = Y[held_out].to_numpy()
    out = {"n_queries": int(len(Y))}
    oracle = Y.max(axis=1).to_numpy()
    for tag, Pm in (("cold", P), ("warm_mean", warm)):
        met = prediction_metrics(y_m, Pm[held_out].to_numpy())
        out[f"{tag}_bce"], out[f"{tag}_auc"] = met["bce"], met.get("auc")
        pick = Pm.to_numpy().argmax(axis=1)
        chosen = Y.to_numpy()[np.arange(len(Y)), pick]
        out[f"{tag}_regret"] = float(np.mean(oracle - chosen))
        out[f"{tag}_pick_rate"] = float(np.mean(np.asarray(Pm.columns)[pick] == held_out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase0-config", default="configs/irt_router.yaml")
    ap.add_argument("--manifold", default="configs/model_manifold.yaml",
                    help="pool id -> Arena key mapping (and the pinned snapshot)")
    ap.add_argument("--methods", default="eloprior,bert")
    ap.add_argument("--models", default=None, help="comma list (default: whole pool)")
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--splits", default="test,ood")
    ap.add_argument("--out", default="investigations/elo_manifold/lomo.csv")
    args = ap.parse_args()

    p0 = load_config(args.phase0_config)
    d = load_training_data(p0)
    obs = d.nirt_observations()
    pool = sorted(obs["model_id"].astype(str).unique())
    held = args.models.split(",") if args.models else pool

    mcfg = yaml.safe_load(p0.resolve(args.manifold).read_text(encoding="utf-8"))
    arena = mcfg["arena"]
    elo = pd.read_csv(fetch_arena_category_elo(p0, snapshot_date=arena["snapshot_date"],
                                               filename=arena["filename"], revision=arena["revision"]))
    ids, X, _ = build_feature_matrix(mcfg, elo)
    if ids != pool:
        raise SystemExit(f"manifold ids {ids} != observation pool {pool}")
    ratings = {m: float(x) for m, x in zip(ids, X[:, 0]) if np.isfinite(x)}

    out_path = p0.resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out_path.exists():
        rows = pd.read_csv(out_path).to_dict("records")
    done = {(r["method"], r["model_id"]) for r in rows}

    for method in args.methods.split(","):
        ncfg = yaml.safe_load(p0.resolve(METHODS[method]).read_text(encoding="utf-8"))
        ncfg["model"]["dim"] = args.dim
        ncfg["model"]["model_params"] = "projected"
        pathway = ncfg["data"].get("pathway", "irt")
        query_pathway = ncfg["data"].get("query_pathway") or None
        for m in held:
            if (method, m) in done:
                continue
            if method == "eloprior":
                # m is absent from training, so its own rating plays no part there
                ncfg["train"]["elo_prior"]["ratings"] = {k: v for k, v in ratings.items() if k != m}
            sub = obs[obs["model_id"].astype(str) != m]
            tr = nirt_dataset_from_config(p0, data=d, split="train", pathway=pathway,
                                          query_pathway=query_pathway, observations=sub)
            va = nirt_dataset_from_config(p0, data=d, split="validation", pathway=pathway,
                                          query_pathway=query_pathway, observations=sub)
            res = fit(ncfg, phase0_cfg=p0, datasets=(tr, va), save=False, verbose=False)
            shift = (_elo_ability_shift(res.model, d, pathway, m, ratings)
                     if method == "eloprior" else 0.0)
            for split in args.splits.split(","):
                frame = _scores(res.model, res.model_index, p0, d, obs, split, pathway,
                                query_pathway, held_out=m, shift=shift)
                r = {"method": method, "model_id": m, "split": split, **_evaluate(frame, m)}
                rows.append(r)
                print(f"[{method}] {m:28s} {split:5s} cold bce {r['cold_bce']:.4f} "
                      f"(warm-mean {r['warm_mean_bce']:.4f})  regret {r['cold_regret']:.4f} "
                      f"(warm-mean {r['warm_mean_regret']:.4f})  pick {r['cold_pick_rate']:.3f}",
                      flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)   # resumable

    df = pd.DataFrame(rows)
    summary = df.groupby(["method", "split"])[
        ["cold_bce", "warm_mean_bce", "cold_auc", "cold_regret", "warm_mean_regret"]].mean()
    print("\n=== LOMO zero-shot cold start (mean over held-out models) ===")
    print(summary.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
