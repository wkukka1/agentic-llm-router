"""kNN-imputed query embedding sweep for the NIRT response model.

Each query's ``e_q`` is replaced by the similarity-weighted mean of its ``k``
nearest TRAIN queries (self-excluded) before the query head sees it. This trains
one NIRT run per ``k`` on the ID split and one on the OOD (held-out families)
split, evaluates prediction + ranking + oracle-routing metrics, and tables them
against the raw (k=0) control and the existing reference models.

    python scripts/nirt/knn_impute_sweep.py --k 1,5,10,25
    python scripts/nirt/knn_impute_sweep.py --k 5,10 --weighting uniform --no-ood

Requires the FAISS query bank (scripts/retrieval/build_query_bank.py). The OOD
bank (indexes/query_bank__ood/) is built in-process if missing. Writes
artifacts/phase2/knn_impute_sweep.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from router.config import load_config
from router.data.phase1 import load_phase1
from router.embeddings import EmbeddingStore
from router.nirt.evaluate import predict_matrix, predict_matrix_from_dataset, ranking_metrics
from router.nirt.metrics import prediction_metrics
from router.nirt.ood import ood_datasets, ood_families, ood_matrices
from router.nirt.routing import align, eval_matrices
from router.nirt.routing_eval import routing_evaluation
from router.nirt.train import fit, load_run
from router.retrieval.knn_impute import build_knn_imputed_store, imputed_pathway_name
from router.retrieval.query_bank import QueryBank, build_query_bank, query_bank_dir

_KEEP = ("bce", "brier", "auc", "acc@0.5", "spearman_r")


def _zeroshot(true_df, cost_df):
    """Drop RouterBench ':5shot' variants so every row is scored on the same
    query set regardless of which query store covers the multishot ids
    (matches scripts/route_compare.py)."""
    keep = [not str(q).endswith(":5shot") for q in true_df.index]
    return true_df.loc[keep], cost_df.loc[keep]


def _metrics_row(true_df, cost_df, pred_df) -> dict:
    true = true_df.to_numpy(np.float64)
    cost = cost_df.to_numpy(np.float64)
    pred = align(pred_df, true_df)
    ok = np.isfinite(pred).all(axis=1)
    t, p, c = true[ok], pred[ok], cost[ok]
    pm = prediction_metrics(t.ravel(), p.ravel())
    rm = ranking_metrics(p, t)
    re = routing_evaluation(p, t, c, list(true_df.columns), lam=0.0)
    return {
        **{k: pm.get(k) for k in _KEEP},
        "optimal_rate": rm["optimal_rate"],
        "regret": rm["regret"],
        "oracle_hit_any_best": re["oracle_hit_rate_any_best"],
        "sel_quality": re["mean_selected_quality"],
        "oracle_quality": re["mean_oracle_quality"],
        "n_queries": int(ok.sum()),
    }


def _nirt_cfg(base: dict, *, dim: int, query_pathway) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg.setdefault("model", {})
    cfg["model"].update(orientation="query_latent", dim=int(dim), model_params="projected")
    cfg.setdefault("data", {})["query_pathway"] = query_pathway
    return cfg


def _ensure_ood_bank(cfg, nirt_cfg, verbose=True) -> Path:
    d_ood = query_bank_dir(cfg).parent / "query_bank__ood"
    if (d_ood / "manifest.json").exists():
        return d_ood
    from router.nirt.ood import family_of_query

    fams = list(ood_families(nirt_cfg))
    fam_of = family_of_query(load_phase1(cfg))
    exclude = [q for q, f in fam_of.items() if f in set(fams)]
    if verbose:
        print(f"[sweep] building OOD bank (excluding {len(exclude):,} queries, families {fams})")
    build_query_bank(cfg, split="train", exclude_query_ids=exclude,
                     holdout_families=fams, out_dir=d_ood)
    return d_ood


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/nirt.yaml")
    ap.add_argument("--phase0-config", default=None)
    ap.add_argument("--k", default=None, help="comma list of k (default: knn_impute.k_values)")
    ap.add_argument("--weighting", default=None, choices=["similarity", "uniform"])
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--no-ood", action="store_true", help="skip the OOD arm")
    ap.add_argument("--rebuild-stores", action="store_true")
    ap.add_argument("--retrain", action="store_true",
                    help="retrain even when a run of that name already exists")
    ap.add_argument("--out", default="artifacts/phase2/knn_impute_sweep.json")
    args = ap.parse_args()

    root = Path(load_config().root)
    cfg_path = Path(args.config)
    nirt_cfg = yaml.safe_load((cfg_path if cfg_path.is_absolute() else root / cfg_path).read_text("utf-8"))
    ki = nirt_cfg.get("knn_impute", {}) or {}
    k_values = ([int(x) for x in args.k.split(",")] if args.k else list(ki.get("k_values", [1, 5, 10, 25])))
    weighting = args.weighting or ki.get("weighting", "similarity")
    dim = args.dim or int(ki.get("dim", 2))
    do_ood = bool(ki.get("ood", True)) and not args.no_ood
    ref_runs = list(ki.get("reference_runs", []))
    wt = "w" if weighting == "similarity" else "u"

    cfg = load_config(args.phase0_config) if args.phase0_config else load_config()
    p0 = cfg
    d = load_phase1(cfg)

    def _store_missing(pathway: str) -> bool:
        return not EmbeddingStore.exists(cfg.resolve(f"data/processed/embeddings/query__{pathway}"))

    def _fit_or_load(cfg_dict, name, *, datasets=None):
        if not args.retrain:
            try:
                m, _rc, mi = load_run(name)
                print(f"  [reuse] {name}")
                return m, mi
            except FileNotFoundError:
                pass
        res = fit(cfg_dict, phase0_cfg=p0, datasets=datasets, name=name, verbose=False)
        return res.model, res.model_index

    if not QueryBank.exists(cfg):
        print("[sweep] FAISS query bank missing; run scripts/retrieval/build_query_bank.py first")
        return 1

    id_true, id_cost = _zeroshot(*eval_matrices(d, split="test"))
    ood_true = ood_cost = None
    ood_bank_dir = None
    if do_ood:
        fams = list(ood_families(nirt_cfg))
        ood_true, ood_cost = _zeroshot(*ood_matrices(d, fams))
        ood_bank_dir = _ensure_ood_bank(cfg, nirt_cfg)

    id_rows: list[dict] = []
    ood_rows: list[dict] = []

    for k in [0] + [x for x in k_values if x > 0]:
        tag = "raw" if k == 0 else f"knn{k}{wt}"
        qp = None if k == 0 else imputed_pathway_name(k, weighting)
        print(f"\n########## k={k}  ({tag}) ##########")

        # ---- ID arm ----
        if k > 0 and (args.rebuild_stores or _store_missing(qp)):
            build_knn_imputed_store(cfg, k=k, weighting=weighting, out_pathway=qp, save=True)
        name = f"nirt-{tag}-{dim}d-projected"
        m, mi = _fit_or_load(_nirt_cfg(nirt_cfg, dim=dim, query_pathway=qp), name)
        pred = predict_matrix(m, mi, d, "test", pathway="irt", query_pathway=qp)
        row = {"k": k, "run": name, **_metrics_row(id_true, id_cost, pred)}
        id_rows.append(row)
        print(f"  ID  test: bce {row['bce']:.4f}  regret {row['regret']:.4f}  "
              f"hit* {row['oracle_hit_any_best']:.3f}")

        # ---- OOD arm ----
        if do_ood:
            qp_ood = None if k == 0 else imputed_pathway_name(k, weighting, ood=True)
            if k > 0 and (args.rebuild_stores or _store_missing(qp_ood)):
                build_knn_imputed_store(cfg, k=k, weighting=weighting, bank_dir=ood_bank_dir,
                                        ood=True, out_pathway=qp_ood, save=True)
            fams = list(ood_families(nirt_cfg))
            tr, va, ood_ds = ood_datasets(d, fams, pathway="irt", query_pathway=qp_ood)
            name_o = f"nirt-{tag}-{dim}d-ood"
            m_o, mi_o = _fit_or_load(_nirt_cfg(nirt_cfg, dim=dim, query_pathway=qp_ood),
                                     name_o, datasets=(tr, va))
            pred_o = predict_matrix_from_dataset(m_o, mi_o, ood_ds)
            row_o = {"k": k, "run": name_o, **_metrics_row(ood_true, ood_cost, pred_o)}
            ood_rows.append(row_o)
            print(f"  OOD     : bce {row_o['bce']:.4f}  regret {row_o['regret']:.4f}  "
                  f"hit* {row_o['oracle_hit_any_best']:.3f}")

    # ---- reference models (ID only) ----
    for r in ref_runs:
        try:
            m, rc, mi = load_run(r)
        except FileNotFoundError:
            print(f"[sweep] reference run '{r}' not found; skipping")
            continue
        rqp = (rc.get("data", {}) or {}).get("query_pathway") or None
        pred = predict_matrix(m, mi, d, "test", pathway=(rc.get("data", {}) or {}).get("pathway", "irt"),
                              query_pathway=rqp)
        id_rows.append({"k": None, "run": r, **_metrics_row(id_true, id_cost, pred)})

    id_df = pd.DataFrame(id_rows)
    ood_df = pd.DataFrame(ood_rows)
    show = ["k", "run", "bce", "brier", "auc", "acc@0.5", "spearman_r",
            "optimal_rate", "regret", "oracle_hit_any_best", "sel_quality"]
    with pd.option_context("display.width", 220, "display.float_format", lambda x: f"{x:.4f}"):
        print("\n=== ID (test) — kNN-imputed query representation sweep ===")
        print(id_df[[c for c in show if c in id_df]].to_string(index=False))
        if not ood_df.empty:
            print("\n=== OOD (held-out families) ===")
            print(ood_df[[c for c in show if c in ood_df]].to_string(index=False))

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "k_values": k_values, "weighting": weighting, "dim": dim,
        "id": id_rows, "ood": ood_rows,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
