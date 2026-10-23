"""Profile-manifold NIRT: pretrain the model-side manifold on Arena-Elo
pseudo-labels, then fine-tune on the real correctness data at a lower lr.

    python scripts/nirt/profile_manifold.py build
    python scripts/nirt/profile_manifold.py pretrain --dim 64 [--seed 42]
    python scripts/nirt/profile_manifold.py finetune --dim 64 [--seed 42] [--control]
    python scripts/nirt/profile_manifold.py sweep [--dims 16,64] [--seeds 42]

``build`` writes ``<out_dir>/pseudo_obs.parquet`` (query_id, model_id, target,
split) and ``<out_dir>/elo_quality_map.json``. Run names: ``pm-pre-d<D>-s<seed>``
(stage 1), ``pm-ft-d<D>-s<seed>`` (stage 2, warm-started from stage 1) and
``pm-ctl-d<D>-s<seed>`` (the same architecture trained from scratch). ``sweep``
skips any run whose ``model.pt`` exists and appends every run's validation
metrics, plus the reference run's, to one CSV. All stages accept ``--set
KEY=VALUE`` overrides on the config (dotted path, YAML value).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from router.config import load_config


def _load_cfg(path: str, sets: list[str], root: Path) -> dict:
    p = Path(path)
    cfg = yaml.safe_load((p if p.is_absolute() else root / p).read_text(encoding="utf-8"))
    for item in sets:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def _stage_cfg(cfg: dict, stage: str, dim: int, seed: int) -> dict:
    """The NIRT config for one stage: ``train`` with the stage's keys merged over it."""
    out = {k: copy.deepcopy(v) for k, v in cfg.items() if k != "profile_manifold"}
    out["seed"] = seed
    out["model"]["profile_latent_dim"] = dim
    out["train"].update(copy.deepcopy(cfg["profile_manifold"].get(stage) or {}))
    return out


#: ``profile_manifold.tag`` -- a run-name suffix so a variant (e.g. another pathway)
#: never overwrites the runs it is compared against; set in ``main``
_TAG = ""


def _run_name(stage: str, dim: int, seed: int) -> str:
    return f"pm-{stage}-d{dim}-s{seed}" + (f"-{_TAG}" if _TAG else "")


# --------------------------------------------------------------------------- #
# build                                                                       #
# --------------------------------------------------------------------------- #
def build(cfg: dict, p0) -> Path:
    from training.data.facade import load_training_data
    from training.data.public_benchmarks import fetch_arena_category_elo
    from training.nirt.elo_pseudo import (
        assign_families,
        dataset_families,
        fit_elo_quality_map,
        pool_category_elo,
        pseudo_observations,
    )

    pm = cfg["profile_manifold"]
    pcfg = pm["pseudo"]
    out_dir = p0.resolve(pm["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- per-model category Elo, pool models only -------------------------
    mcfg = yaml.safe_load(p0.resolve(pm["manifold"]).read_text(encoding="utf-8"))
    arena = mcfg["arena"]
    elo_rows = pd.read_csv(fetch_arena_category_elo(p0, snapshot_date=arena["snapshot_date"],
                                                    filename=arena["filename"],
                                                    revision=arena["revision"]))
    d = load_training_data(p0)
    pool = set(d.nirt_observations()["model_id"].astype(str))
    elo = pool_category_elo(mcfg, elo_rows)
    elo = elo[elo.index.isin(pool)]
    print(f"[build] Arena-rated pool models: {len(elo)}/{len(pool)}; "
          f"unrated: {sorted(pool - set(elo.index))}")

    # -- Elo -> quality map from TRAIN-split mean targets ------------------
    # project split = data/splits/*.json (responses.parquet's `split` is the raw
    # source split); the NIRT target is the chance-corrected score it trains on
    fam_of = dataset_families(p0.to_dict().get("profiles", {}).get("task_families", {}))
    r = pd.read_parquet(p0.path("processed") / "responses.parquet", columns=["query_id", "dataset"])
    family = r.drop_duplicates("query_id").set_index("query_id")["dataset"].map(fam_of).fillna("other")
    nobs = d.nirt_observations()
    tr = nobs[(nobs["split"] == "train") & nobs["source"].isin(pcfg["sources"])].copy()
    tr["family"] = tr["query_id"].map(family).fillna("other")
    perf = tr.groupby(["model_id", "family"])["target"].mean().unstack()
    fmap = fit_elo_quality_map(perf, elo)
    for fam, fit in sorted(fmap.items()):
        print(f"[build] {fam:12s} <- elo:{fit['category']:8s} n={fit['n']:2d} "
              f"beta={fit['beta']:+.3f} r2={fit['r2']:.2f}{'  (fallback)' if fit.get('fallback') else ''}")

    # -- the query pool: every train-eligible row of the query store -------
    store = d.query_embeddings(cfg.get("data", {}).get("pathway", "irt"))
    ids = pd.Series(np.asarray(store.ids).astype(str))
    train_q = set(json.loads((p0.path("splits") / "train.json").read_text())["query_ids"])
    eligible = ids[ids.isin(train_q)].index.to_numpy()
    rng = np.random.default_rng(int(pcfg.get("seed", 0)))
    take = np.sort(rng.choice(eligible, size=min(int(pcfg["n_queries"]), len(eligible)),
                              replace=False))
    q_ids = ids.iloc[take].to_numpy()

    fam_known = tr.drop_duplicates("query_id").set_index("query_id")["family"]
    fam = pd.Series(q_ids).map(fam_known).to_numpy(dtype=object)
    unknown = pd.isna(fam)
    if unknown.any():
        ref_rows = ids[ids.isin(fam_known.index)].index.to_numpy()
        M = np.asarray(store.matrix)
        fam[unknown] = assign_families(M[take[unknown]], M[ref_rows],
                                       fam_known.reindex(ids.iloc[ref_rows]).to_numpy())
    print(f"[build] {len(q_ids):,} queries ({int(unknown.sum()):,} family by nearest centroid): "
          + ", ".join(f"{k}={v}" for k, v in pd.Series(fam).value_counts().items()))

    obs = pseudo_observations(q_ids, fam.astype(str), elo, fmap,
                              models_per_query=pcfg.get("models_per_query"),
                              seed=int(pcfg.get("seed", 0)))
    val_q = set(rng.choice(q_ids, size=int(len(q_ids) * float(pcfg["val_fraction"])),
                           replace=False))
    obs["split"] = np.where(obs["query_id"].isin(val_q), "validation", "train")
    obs.to_parquet(out_dir / "pseudo_obs.parquet", index=False)
    (out_dir / "elo_quality_map.json").write_text(json.dumps(
        {"fmap": fmap, "rated_models": sorted(elo.index),
         "n_queries": int(len(q_ids)), "n_rows": int(len(obs))}, indent=2), encoding="utf-8")
    print(f"[build] wrote {len(obs):,} pseudo rows -> {out_dir / 'pseudo_obs.parquet'}")
    return out_dir


# --------------------------------------------------------------------------- #
# stages                                                                      #
# --------------------------------------------------------------------------- #
def pretrain(cfg: dict, p0, dim: int, seed: int):
    from router.data.nirt import NIRTDataset
    from training.data.facade import load_training_data
    from training.nirt.train import fit

    obs = pd.read_parquet(p0.resolve(cfg["profile_manifold"]["out_dir"]) / "pseudo_obs.parquet")
    pathway = cfg.get("data", {}).get("pathway", "irt")
    d = load_training_data(p0)
    qs, ps = d.query_embeddings(pathway), d.profile_embeddings(pathway)
    ds = (NIRTDataset(obs[obs["split"] == "train"], qs, ps),
          NIRTDataset(obs[obs["split"] == "validation"], qs, ps))
    return fit(_stage_cfg(cfg, "pretrain", dim, seed), phase0_cfg=p0, datasets=ds,
               name=_run_name("pre", dim, seed))


def finetune(cfg: dict, p0, dim: int, seed: int, *, control: bool = False,
             tag: str | None = None, overrides: dict | None = None):
    from training.nirt.train import fit

    stage = "control" if control else "finetune"
    scfg = _stage_cfg(cfg, stage, dim, seed)
    scfg["train"].update(copy.deepcopy(overrides or {}))
    if not control:
        scfg["train"]["init_from"] = _run_name("pre", dim, seed)
    name = _run_name("ctl" if control else "ft", dim, seed) + (f"-{tag}" if tag else "")
    return fit(scfg, phase0_cfg=p0, name=name)


def _row(run_dir: Path, **extra) -> dict:
    r = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    vm = r["val_metrics"]
    return {**extra, "run": r["name"], "best_epoch": r["best_epoch"],
            "elapsed_sec": r.get("elapsed_sec"),
            **{f"val_{k}": vm.get(k) for k in ("bce", "brier", "auc", "spearman_r", "regret")}}


def sweep(cfg: dict, p0, dims: list[int], seeds: list[int]) -> Path:
    from training.nirt.train import _resolve_runs_dir

    sw = cfg["profile_manifold"]["sweep"]
    runs = _resolve_runs_dir(cfg, p0)
    rows = []
    ref = runs / sw["reference_run"]
    if (ref / "run.json").exists():
        rows.append(_row(ref, stage="reference", dim=None, seed=None))
    for dim in dims:
        for seed in seeds:
            for stage in ("pre", "ft", "ctl"):
                run_dir = runs / _run_name(stage, dim, seed)
                if not (run_dir / "model.pt").exists():
                    if stage == "pre":
                        pretrain(cfg, p0, dim, seed)
                    else:
                        finetune(cfg, p0, dim, seed, control=stage == "ctl")
                rows.append(_row(run_dir, stage=stage, dim=dim, seed=seed))
                out = p0.resolve(sw["out"])
                out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(out, index=False)     # rewritten after every run
    df = pd.DataFrame(rows)
    print(df.groupby(["stage", "dim"], dropna=False)[["val_bce", "val_regret"]]
          .agg(["mean", "std"]).round(4).to_string())
    return p0.resolve(sw["out"])


def _run_diagnostics(run_dir: Path, p0, d, pseudo_val, eval_mats: dict) -> dict:
    """Test regret per dense source pool (quality routing, as route_eval.py) and
    drift: Spearman of the run's predictions on the pseudo-label validation rows
    -- how much of the Elo structure survived fine-tuning."""
    from scipy.stats import spearmanr

    from evaluation.nirt.routing import align
    from evaluation.routing.oracle import routing_evaluation
    from router.nirt.checkpoint import load_run
    from router.nirt.predict import predict_dataset, predict_matrix

    model, run_cfg, midx = load_run(run_dir.name, runs_dir=run_dir.parent)
    pathway = (run_cfg.get("data", {}) or {}).get("pathway", "irt")
    out = {}
    pred_all = predict_matrix(model, midx, d, "test", pathway=pathway)
    for source, (true_df, cost_df) in eval_mats.items():
        ev = routing_evaluation(align(pred_all, true_df), true_df.to_numpy(np.float64),
                                cost_df.to_numpy(np.float64), list(true_df.columns), lam=0.0)
        out[f"test_regret_{source}"] = float(ev["mean_regret"])
    y, p = predict_dataset(model, pseudo_val, midx)
    out["pseudo_spearman"] = float(spearmanr(y, p).correlation)
    return out


def lr_sweep(cfg: dict, p0) -> Path:
    """Fine-tune grid at lower learning rates, warm-started from the existing
    ``pm-pre-*`` runs; one CSV row per run with val metrics, test regret and drift."""
    from evaluation.nirt.routing import eval_matrices
    from router.data.nirt import NIRTDataset
    from training.data.facade import load_training_data
    from training.nirt.train import _resolve_runs_dir

    ls = cfg["profile_manifold"]["lr_sweep"]
    runs = _resolve_runs_dir(cfg, p0)
    out = p0.resolve(ls["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    d = load_training_data(p0)
    pathway = cfg.get("data", {}).get("pathway", "irt")
    obs = pd.read_parquet(p0.resolve(cfg["profile_manifold"]["out_dir"]) / "pseudo_obs.parquet")
    pseudo_val = NIRTDataset(obs[obs["split"] == "validation"],
                             d.query_embeddings(pathway), d.profile_embeddings(pathway))
    pool = sorted(json.loads((runs / cfg["profile_manifold"]["sweep"]["reference_run"]
                              / "run.json").read_text())["model_index"])
    eval_mats = {s: eval_matrices(d, split="test", models=pool, source=s)
                 for s in ls.get("eval_sources", ["irt_router", "routerbench"])}

    # the comparison points first: reference, then the lr 1e-4 fine-tunes and controls
    targets = [("reference", None, None, None, runs / cfg["profile_manifold"]["sweep"]["reference_run"])]
    for dim in ls["dims"]:
        for seed in ls["seeds"]:
            targets.append(("ft-lr1e-4", dim, seed, None, runs / _run_name("ft", dim, seed)))
            targets.append(("ctl", dim, seed, None, runs / _run_name("ctl", dim, seed)))
            for entry in ls["grid"]:
                entry = dict(entry)
                tag = entry.pop("tag")
                targets.append((tag, dim, seed, entry,
                                runs / f"{_run_name('ft', dim, seed)}-{tag}"))

    rows = []
    for tag, dim, seed, overrides, run_dir in targets:
        if overrides is not None and not (run_dir / "model.pt").exists():
            finetune(cfg, p0, dim, seed, tag=tag, overrides=overrides)
        if not (run_dir / "model.pt").exists():
            print(f"[lr_sweep] missing {run_dir.name}, skipped")
            continue
        rows.append({**_row(run_dir, stage=tag, dim=dim, seed=seed),
                     **_run_diagnostics(run_dir, p0, d, pseudo_val, eval_mats)})
        pd.DataFrame(rows).to_csv(out, index=False)          # rewritten after every run
    df = pd.DataFrame(rows)
    cols = ["val_bce", "test_regret_irt_router", "test_regret_routerbench", "pseudo_spearman"]
    print(df.groupby(["dim", "stage"], dropna=False)[cols].agg(["mean", "std"]).round(4).to_string())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "pretrain", "finetune", "sweep", "lr_sweep"])
    ap.add_argument("--tag", default=None, help="finetune: run-name suffix (pm-ft-d<D>-s<seed>-<tag>)")
    ap.add_argument("--config", default="configs/nirt_profile_manifold.yaml")
    ap.add_argument("--phase0-config", default=None)
    ap.add_argument("--dim", type=int, default=None, help="profile_latent_dim D")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--control", action="store_true", help="finetune: train from scratch instead")
    ap.add_argument("--dims", default=None, help="sweep: comma list (default: config)")
    ap.add_argument("--seeds", default=None, help="sweep: comma list (default: config)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    p0 = load_config(args.phase0_config) if args.phase0_config else load_config()
    cfg = _load_cfg(args.config, args.set, Path(p0.root))
    global _TAG
    _TAG = str(cfg["profile_manifold"].get("tag") or "")
    dim = args.dim or int(cfg["model"]["profile_latent_dim"])
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))

    if args.command == "build":
        build(cfg, p0)
    elif args.command == "pretrain":
        pretrain(cfg, p0, dim, seed)
    elif args.command == "finetune":
        finetune(cfg, p0, dim, seed, control=args.control, tag=args.tag)
    elif args.command == "lr_sweep":
        print(f"[lr_sweep] wrote {lr_sweep(cfg, p0)}")
    else:
        sw = cfg["profile_manifold"]["sweep"]
        dims = [int(x) for x in args.dims.split(",")] if args.dims else sw["dims"]
        seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else sw["seeds"]
        print(f"[sweep] wrote {sweep(cfg, p0, dims, seeds)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
