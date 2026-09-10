"""Train the Phase 1 plain Bernoulli / BCE NIRT baseline.

    from router.nirt.baseline.train import fit
    res = fit(yaml.safe_load(open("configs/phase1.yaml")))

Phase 1 facade -> ``build_arrays`` (materialised numpy, CPU RAM, never a block on
GPU) -> ``DataLoader`` -> Adam + early stop on validation BCE -> checkpoint.
Correctness signal only (Arena excluded); binary targets.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from router.config import Config, load_config
from router.determinism import seed_everything
from router.nirt.metrics import prediction_metrics
from router.provenance import file_digest

from .checkpoint import load_checkpoint, save_checkpoint
from .data import batched_forward, build_arrays, to_tensors
from .losses import RegConfig, bce_loss, class_balance_pos_weight, regularization
from .model import build_baseline_model

_DEFAULT_DIR = "artifacts/phase1/baseline"


@dataclass
class BaselineRun:
    config: dict
    history: list = field(default_factory=list)
    best_epoch: int = -1
    val_metrics: dict = field(default_factory=dict)
    train_imbalance: dict = field(default_factory=dict)
    model: object = None
    model_index: dict = field(default_factory=dict)
    dims: dict = field(default_factory=dict)
    path: Optional[Path] = None


@dataclass
class TrainCfg:
    seed: int = 42
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 4096
    epochs: int = 60
    patience: int = 8
    device: str = "cpu"
    target: str = "binary"          # {binary, soft}
    class_weighting: bool = False    # BCEWithLogits pos_weight = #neg/#pos
    grad_clip: float = 0.0           # max grad-norm (0 = off); applied only for continuous heads

    @classmethod
    def from_dict(cls, d: dict) -> "TrainCfg":
        d = d or {}
        g = lambda k, alt=None: d.get(k, d.get(alt, getattr(cls, k)))  # noqa: E731
        return cls(
            seed=int(d.get("seed", 42)), lr=float(g("lr", "learning_rate")),
            weight_decay=float(g("weight_decay")), batch_size=int(g("batch_size")),
            epochs=int(g("epochs")), patience=int(g("patience")), device=str(g("device")),
            target=str(g("target")), class_weighting=bool(g("class_weighting")),
            grad_clip=float(g("grad_clip")),
        )


def fit(
    phase1_cfg: dict, *, phase0_cfg: Optional[Config] = None, arrays: Optional[tuple] = None,
    save: bool = True, out_dir: Optional[str] = None, resume_from: Optional[str] = None,
    verbose: bool = True,
) -> BaselineRun:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    cfg = json.loads(json.dumps(phase1_cfg))
    tcfg = TrainCfg.from_dict({**cfg.get("train", {}), "seed": cfg.get("seed", 42)})
    mcfg = cfg.get("model", {}) or {}
    abl = cfg.get("ablation", {}) or {}
    rcfg = RegConfig.from_dict(cfg.get("regularization"))
    resp = cfg.get("response", {}) or {}
    use_relevance = bool(abl.get("use_relevance", mcfg.get("use_relevance", True)))
    use_warmup = bool(abl.get("use_warmup", mcfg.get("use_warmup", False)))
    use_interaction = bool(abl.get("use_interaction", mcfg.get("use_interaction", True)))
    model_params = mcfg.get("model_params", "free")
    seed_everything(tcfg.seed)

    p0 = phase0_cfg or load_config()
    if arrays is not None:
        tr_arr, va_arr = arrays
    else:
        from router.data.phase1 import load_phase1

        d = load_phase1(p0)
        common = dict(data=d, pathway=cfg.get("data", {}).get("pathway", "irt"),
                      binary_threshold=float(resp.get("binary_threshold", 0.5)),
                      score_kind=resp.get("score_kind", "effective"),
                      use_relevance=use_relevance, use_warmup=use_warmup)
        tr_arr = build_arrays(p0, split="train", **common)
        va_arr = build_arrays(p0, split="validation", model_index=tr_arr.meta["model_index"], **common)

    model_index = tr_arr.meta["model_index"]
    dims = {"query_dim": tr_arr.query_dim, "profile_dim": tr_arr.profile_dim,
            "relevance_dim": tr_arr.relevance_dim if use_relevance else 0,
            "n_models": len(model_index)}

    # bake the resolved ablation + response model into the model cfg so a
    # checkpoint rebuilds the exact same architecture.
    mcfg = {**mcfg, "ablation": {"use_relevance": use_relevance, "use_interaction": use_interaction,
                                 "use_warmup": use_warmup},
            "response_model": resp.get("model", mcfg.get("response_model", "bernoulli")),
            "response_cfg": resp.get("cfg", mcfg.get("response_cfg", {}))}
    cfg["model"] = mcfg
    model = build_baseline_model(
        mcfg, n_models=dims["n_models"], query_dim=dims["query_dim"],
        profile_dim=dims["profile_dim"] or dims["query_dim"], relevance_dim=dims["relevance_dim"],
    ).to(tcfg.device)
    if resume_from:
        model.load_state_dict(load_checkpoint(resume_from, map_location=tcfg.device)[0].state_dict())
        if verbose:
            print(f"[phase1] resumed weights from {resume_from}")

    head = model.response_head
    bernoulli = model.response_model == "bernoulli"
    if getattr(head, "interior_only", False):        # Beta: interior observations only
        tr_arr, va_arr = _interior(tr_arr), _interior(va_arr)

    tt = dict(model_params=model_params, use_relevance=use_relevance, use_warmup=use_warmup)
    tr, va = to_tensors(tr_arr, **tt), to_tensors(va_arr, **tt)
    zeros = torch.zeros(len(tr["y"]), 1)
    cols = [tr["e_q"], tr["ref"], tr["y"], tr["y_soft"],
            tr["r_q"] if tr["r_q"] is not None else zeros,
            tr["nbr"] if tr["nbr"] is not None else zeros]
    loader = DataLoader(TensorDataset(*cols), batch_size=tcfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(tcfg.seed))
    has_r, has_n = tr["r_q"] is not None, tr["nbr"] is not None

    pos_weight = (class_balance_pos_weight(tr["y"]).to(tcfg.device)
                  if tcfg.class_weighting else None)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    # identifiability regularisers touch only the ability rows we actually fit
    # (a model absent from training gets no gradient -> no cold-start leakage).
    train_rows = torch.unique(tr["ref"]) if model_params == "free" else None

    history: list = []
    best_val, best_epoch, best_state, since = float("inf"), -1, None, 0
    t0 = time.time()

    soft_target = (not bernoulli) or tcfg.target == "soft"
    for epoch in range(tcfg.epochs):
        model.train()
        running = seen = 0.0
        for e_q, ref, y, y_soft, r_q, nbr in loader:
            e_q, ref = e_q.to(tcfg.device), ref.to(tcfg.device)
            target = (y_soft if soft_target else y).to(tcfg.device)
            out = model(e_q, ref, r_q.to(tcfg.device) if has_r else None,
                        nbr.to(tcfg.device) if has_n else None)
            loss = (bce_loss(out.logit, target, pos_weight=pos_weight) if bernoulli
                    else head.loss(target, out.response))
            theta_all = model.theta.weight[train_rows] if model_params == "free" else out.theta_m
            reg = regularization(rcfg, theta_all=theta_all, a_q=out.a_q, b_q=out.b_q).to(tcfg.device)
            opt.zero_grad()
            (loss + reg).backward()
            if not bernoulli and tcfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            running += float(loss.item()) * len(y)
            seen += len(y)
        train_loss = running / max(seen, 1)

        val_prob = batched_forward(model, va, device=tcfg.device)["proba"]
        vm = prediction_metrics(va_arr.y, val_prob)
        val_nll = _mean_nll(model, va, va_arr, device=tcfg.device) if not bernoulli else vm["bce"]
        stop = vm["bce"] if bernoulli else val_nll
        history.append({"epoch": epoch, "train_loss": train_loss, "val_nll": val_nll,
                        **{f"val_{k}": v for k, v in vm.items() if isinstance(v, (int, float))}})
        if verbose:
            print(f"[phase1] epoch {epoch:3d}  train_loss {train_loss:.4f}  val_nll {val_nll:.4f}  "
                  f"val_acc {vm['accuracy']:.4f}  val_brier {vm['brier']:.4f}")

        if stop < best_val - 1e-5:
            best_val, best_epoch, since = stop, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= tcfg.patience:
                if verbose:
                    print(f"[phase1] early stop at epoch {epoch} (best {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    val_metrics = prediction_metrics(va_arr.y, batched_forward(model, va, device=tcfg.device)["proba"])
    if not bernoulli:
        val_metrics["nll"] = _mean_nll(model, va, va_arr, device=tcfg.device)

    run = BaselineRun(config=cfg, history=history, best_epoch=best_epoch, val_metrics=val_metrics,
                      train_imbalance=tr_arr.imbalance_report(), model=model,
                      model_index=model_index, dims=dims)
    if save:
        out = Path(out_dir or cfg.get("out_dir", _DEFAULT_DIR))
        out = out if out.is_absolute() else p0.root / out
        save_checkpoint(
            out, model=model, config=cfg, model_index=model_index, query_dim=dims["query_dim"],
            profile_dim=dims["profile_dim"] or dims["query_dim"], relevance_dim=dims["relevance_dim"],
            metrics={"validation": val_metrics, "best_epoch": best_epoch,
                     "train_imbalance": run.train_imbalance},
            history=history, provenance=_provenance(p0, cfg, tr_arr, va_arr, tcfg.seed))
        run.path = out
        if verbose:
            print(f"[phase1] wrote {out} ({round(time.time() - t0, 1)}s)")
    return run


def _interior(arr):
    """Rows with ``0 < y_soft < 1`` (Beta support)."""
    from dataclasses import replace

    m = (arr.y_soft > 0.0) & (arr.y_soft < 1.0)
    sub = {f: getattr(arr, f)[m] for f in
           ("e_q", "midx", "y", "y_soft", "query_ids", "model_ids")}
    for f in ("e_m", "r_q", "nbr", "cost"):
        v = getattr(arr, f)
        sub[f] = v[m] if v is not None else None
    return replace(arr, **sub)


def _mean_nll(model, tensors: dict, arr, *, device: str = "cpu") -> float:
    return float(batched_forward(model, tensors, fields=("nll",), y=arr.y_soft,
                                 device=device)["nll"].mean())


def _provenance(cfg: Config, phase1_cfg: dict, tr_arr, va_arr, seed: int) -> dict:
    def _hash(p: Path) -> Optional[str]:
        return file_digest(p, algo="sha1")

    proc, tax = cfg.path("processed"), cfg.path("taxonomy")
    return {
        "seed": seed,
        "phase1_config": phase1_cfg,
        "phase0_artifacts": {
            k: _hash(base / f)
            for k, (base, f) in {
                "nirt_observations": (proc, "nirt_observations.parquet"),
                "queries": (proc, "queries.parquet"),
                "model_profiles": (proc, "model_profiles.parquet"),
                "clusters": (tax, "clusters.parquet"),
                "centroids": (tax, "centroids.npy"),
            }.items()
        },
        "taxonomy_version": int(cfg.get("taxonomy.version", 1)),
        "dataset_stats": {
            "train": tr_arr.imbalance_report(),
            "validation_n": int(len(va_arr)),
            "binary_threshold": tr_arr.meta["binary_threshold"],
            "score_kind": tr_arr.meta["score_kind"],
            "relevance_available": tr_arr.meta["relevance_available"],
            "warmup_available": tr_arr.meta["warmup_available"],
        },
    }
