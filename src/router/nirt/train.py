"""Train the NIRT baseline predictor (Phase 2, v1).

    from router.nirt.train import fit
    res = fit(yaml.safe_load(open("configs/nirt.yaml")))

``fit`` reads the training split through the Phase 1 facade
(:func:`router.data.phase1.load_phase1` -> :meth:`Phase1Data.nirt_dataset`),
gathers it into RAM once (``train.materialize``), runs Adam with early stopping
on the validation split, and writes ``model.pt`` + ``run.json`` under
``runs_dir/<name>/``.

Only the absolute-correctness signal is used -- this is N-IRT, kept isolated from
any pairwise / M-IRT path.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..determinism import seed_everything
from ..provenance import git_sha as _git_sha
from .metrics import marginal_baselines, prediction_metrics
from .model import build_model

_DEFAULT_RUNS_DIR = "data/processed/nirt_runs"

__all__ = ["fit", "load_run", "RunResult", "seed_everything"]


@dataclass
class RunResult:
    name: str
    config: dict
    history: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    val_metrics: dict = field(default_factory=dict)
    baselines: dict = field(default_factory=dict)
    path: Optional[Path] = None
    model: object = None            # the fitted NIRTModel (in-memory; not serialised here)
    model_index: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# data packing                                                                #
# --------------------------------------------------------------------------- #
def _pack(dataset, model_index: dict[str, int]) -> dict:
    """Materialize a :class:`NIRTDataset` into contiguous arrays (one copy)."""
    g = dataset.gather()
    model_ids = np.asarray(dataset.model_ids)
    midx = np.array([model_index.get(str(m), -1) for m in model_ids], dtype=np.int64)
    qids = np.asarray(dataset.query_ids).astype(str)
    # dense per-query group id (0..G-1), for the P5 query-grouped sampler / losses
    _, qgroup = np.unique(qids, return_inverse=True)
    return {
        "q": np.ascontiguousarray(g["query_embedding"], dtype=np.float32),
        "m": np.ascontiguousarray(g["model_embedding"], dtype=np.float32),
        "y": np.ascontiguousarray(g["target"], dtype=np.float32),
        "midx": midx,
        "model_ids": model_ids,
        "qgroup": np.ascontiguousarray(qgroup, dtype=np.int64),
    }


# --------------------------------------------------------------------------- #
# losses  (P5: routing-aware pairwise / listwise, alongside the per-cell BCE)  #
# --------------------------------------------------------------------------- #
_POINTWISE = ("soft_bce", "hard_bce", "mse")
_GROUPED = ("pairwise", "listwise", "bce_pairwise")


def _group_blocks(groups):
    """``groups`` (row-order = grouped, from :func:`_grouped_batches`) -> list of
    ``(start, size)`` per contiguous group, plus a fast-path ``(G, k)`` when every
    group is the same size ``k``."""
    import torch

    boundaries = torch.cat([
        torch.tensor([0], device=groups.device),
        (groups[1:] != groups[:-1]).nonzero(as_tuple=True)[0] + 1,
        torch.tensor([len(groups)], device=groups.device),
    ])
    sizes = (boundaries[1:] - boundaries[:-1]).tolist()
    starts = boundaries[:-1].tolist()
    equal_k = sizes[0] if sizes and all(s == sizes[0] for s in sizes) else None
    return list(zip(starts, sizes)), equal_k


def _pairwise_within_group(logit, y, groups):
    """Mean RankNet / Bradley-Terry loss over model pairs *within* each query
    group: ``BCE(z_m - z_m', 1[y_m > y_m'])``. Ties (``y_m == y_m'``) are skipped."""
    import torch
    from torch.nn import functional as F

    blocks, k = _group_blocks(groups)
    if k is not None and k >= 2:                      # fast path: dense (G, k)
        G = len(blocks)
        Z = logit.view(G, k)
        Y = y.view(G, k)
        dZ = Z.unsqueeze(2) - Z.unsqueeze(1)          # (G, k, k)
        dY = Y.unsqueeze(2) - Y.unsqueeze(1)
        m = torch.triu(torch.ones(k, k, dtype=torch.bool, device=logit.device), 1) & (dY != 0)
        if not m.any():
            return torch.zeros((), dtype=logit.dtype, device=logit.device)
        return F.binary_cross_entropy_with_logits(dZ[m], (dY[m] > 0).to(logit.dtype))
    total, npairs = torch.zeros((), dtype=logit.dtype, device=logit.device), 0
    for s, sz in blocks:
        if sz < 2:
            continue
        z, t = logit[s:s + sz], y[s:s + sz]
        dz = z.unsqueeze(1) - z.unsqueeze(0)
        dt = t.unsqueeze(1) - t.unsqueeze(0)
        mm = torch.triu(torch.ones_like(dz, dtype=torch.bool), 1) & (dt != 0)
        if mm.any():
            total = total + F.binary_cross_entropy_with_logits(dz[mm], (dt[mm] > 0).to(logit.dtype),
                                                               reduction="sum")
            npairs += int(mm.sum())
    return total / max(npairs, 1)


def _listwise_within_group(logit, y, groups, *, tau: float = 0.1):
    """ListNet top-1: cross-entropy between ``softmax(z_q)`` and the soft label
    distribution ``softmax(y_q / tau)`` over the models scored on each query."""
    import torch
    from torch.nn import functional as F

    blocks, k = _group_blocks(groups)
    if k is not None and k >= 2:
        G = len(blocks)
        logp = F.log_softmax(logit.view(G, k), dim=1)
        q = F.softmax(y.view(G, k) / tau, dim=1)
        return -(q * logp).sum(1).mean()
    total, ng = torch.zeros((), dtype=logit.dtype, device=logit.device), 0
    for s, sz in blocks:
        if sz < 2:
            continue
        total = total - (F.softmax(y[s:s + sz] / tau, dim=0)
                         * F.log_softmax(logit[s:s + sz], dim=0)).sum()
        ng += 1
    return total / max(ng, 1)


def _loss_fn(kind: str, *, aux_weight: float = 0.5, tau: float = 0.1):
    """Return ``loss(logit, y, groups=None)``. Pointwise losses ignore ``groups``;
    the grouped losses (P5) require it (query-group ids per row)."""
    import torch
    from torch.nn import functional as F

    if kind == "soft_bce":
        return lambda logit, y, groups=None: F.binary_cross_entropy_with_logits(logit, y)
    if kind == "hard_bce":
        return lambda logit, y, groups=None: F.binary_cross_entropy_with_logits(
            logit, (y >= 0.5).float())
    if kind == "mse":
        return lambda logit, y, groups=None: F.mse_loss(torch.sigmoid(logit), y)
    if kind == "pairwise":
        return lambda logit, y, groups: _pairwise_within_group(logit, y, groups)
    if kind == "listwise":
        return lambda logit, y, groups: _listwise_within_group(logit, y, groups, tau=tau)
    if kind == "bce_pairwise":
        return lambda logit, y, groups: (
            F.binary_cross_entropy_with_logits(logit, y)
            + aux_weight * _pairwise_within_group(logit, y, groups))
    raise ValueError(f"unknown loss {kind!r}; use {_POINTWISE + _GROUPED}")


def _grouped_batches(qgroup: np.ndarray, batch_queries: int, gen):
    """Yield row-index arrays, ``batch_queries`` whole query groups per batch
    (shuffled). Every model row for a sampled query is in the same batch, so the
    pairwise / listwise losses see complete per-query model lists."""
    import torch

    g_ids = np.unique(qgroup)
    order = torch.randperm(len(g_ids), generator=gen).numpy()
    rows_of = {g: np.where(qgroup == g)[0] for g in g_ids}
    for i in range(0, len(g_ids), batch_queries):
        chunk = g_ids[order[i : i + batch_queries]]
        yield np.concatenate([rows_of[g] for g in chunk])


def _val_regret(packed: dict, prob: np.ndarray) -> float:
    """Mean per-query routing regret: ``mean_q( max_m y[q,m] - y[q, argmax_m prob[q,m]] )``.
    Only queries with >= 2 models scored contribute. For ``val_metric: regret`` (P5)."""
    y = np.asarray(packed["y"], np.float64)
    g = np.asarray(packed["qgroup"])
    p = np.asarray(prob, np.float64)
    reg, nq = 0.0, 0
    for gid in np.unique(g):
        m = g == gid
        if m.sum() < 2:
            continue
        yy, pp = y[m], p[m]
        reg += float(yy.max() - yy[int(pp.argmax())])
        nq += 1
    return reg / max(nq, 1)


def _predict_packed(model, packed: dict, batch_size: int = 8192) -> np.ndarray:
    import torch

    projected = model.model_params == "projected"
    q = torch.from_numpy(packed["q"])
    ref_src = torch.from_numpy(packed["m"]) if projected else torch.from_numpy(packed["midx"])
    out = np.empty(len(packed["y"]), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(out), batch_size):
            sl = slice(i, i + batch_size)
            out[sl] = torch.sigmoid(model(q[sl], ref_src[sl])).numpy()
    return out


# --------------------------------------------------------------------------- #
# fit                                                                         #
# --------------------------------------------------------------------------- #
def fit(
    nirt_cfg: dict,
    *,
    phase0_cfg=None,
    datasets: Optional[tuple] = None,
    name: Optional[str] = None,
    runs_dir: Optional[os.PathLike] = None,
    save: bool = True,
    verbose: bool = True,
) -> RunResult:
    import torch

    nirt_cfg = json.loads(json.dumps(nirt_cfg))  # deep copy, plain types
    seed = int(nirt_cfg.get("seed", 42))
    seed_everything(seed)

    mcfg = nirt_cfg.get("model", {}) or {}
    tcfg = nirt_cfg.get("train", {}) or {}
    dcfg = nirt_cfg.get("data", {}) or {}
    pathway = dcfg.get("pathway", "irt")
    query_pathway = dcfg.get("query_pathway") or None

    # -- data ---------------------------------------------------------------
    if datasets is not None:
        train_ds, val_ds = datasets
    else:
        from ..config import load_config
        from ..data.phase1 import load_phase1

        p0 = phase0_cfg or load_config()
        d = load_phase1(p0)
        train_ds = d.nirt_dataset(split="train", pathway=pathway, query_pathway=query_pathway)
        val_ds = d.nirt_dataset(split="validation", pathway=pathway, query_pathway=query_pathway)

    all_ids = sorted(set(map(str, train_ds.model_ids)) | set(map(str, val_ds.model_ids)))
    model_index = {m: i for i, m in enumerate(all_ids)}

    tr = _pack(train_ds, model_index)
    va = _pack(val_ds, model_index)
    if (tr["midx"] < 0).any():  # pragma: no cover - defensive
        raise RuntimeError("training observation references a model id not in the index")

    # -- model ------------------------------------------------------------
    model = build_model(
        mcfg,
        n_models=len(model_index),
        query_dim=int(train_ds.query_dim),
        profile_dim=int(train_ds.model_dim),
    )
    projected = model.model_params == "projected"

    lr = float(tcfg.get("lr", 1e-3))
    weight_decay = float(tcfg.get("weight_decay", 1e-5))
    batch_size = int(tcfg.get("batch_size", 4096))
    epochs = int(tcfg.get("epochs", 50))
    patience = int(tcfg.get("patience", 6))
    val_key = tcfg.get("val_metric", "bce")
    grad_clip = float(tcfg.get("grad_clip", 0.0) or 0.0)          # 0 -> off (legacy)
    lr_schedule = str(tcfg.get("lr_schedule", "none") or "none")  # {none, cosine, plateau}
    loss_kind = tcfg.get("loss", "soft_bce")
    sampler = str(tcfg.get("sampler", "cell") or "cell")          # {cell, query} (P5)
    if loss_kind in _GROUPED and sampler != "query":
        sampler = "query"   # a within-query loss needs whole query groups per batch
    loss_fn = _loss_fn(loss_kind, aux_weight=float(tcfg.get("loss_aux_weight", 0.5)),
                       tau=float(tcfg.get("listwise_tau", 0.1)))
    batch_queries = int(tcfg.get("batch_queries", 512))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = None
    if lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    elif lr_schedule == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=max(1, patience // 2)
        )
    elif lr_schedule != "none":
        raise ValueError(f"lr_schedule must be one of none|cosine|plateau, got {lr_schedule!r}")

    qt = torch.from_numpy(tr["q"])
    mt = torch.from_numpy(tr["m"])
    yt = torch.from_numpy(tr["y"])
    it = torch.from_numpy(tr["midx"])
    gt = torch.from_numpy(tr["qgroup"])
    n = len(yt)
    gen = torch.Generator().manual_seed(seed)

    baselines = marginal_baselines(tr["y"], tr["model_ids"], va["y"], va["model_ids"])

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    since_improved = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        if sampler == "query":
            batches = [torch.from_numpy(b) for b in
                       _grouped_batches(tr["qgroup"], batch_queries, gen)]
        else:
            perm = torch.randperm(n, generator=gen)
            batches = [perm[i : i + batch_size] for i in range(0, n, batch_size)]
        running = seen = 0.0
        for idx in batches:
            ref = mt[idx] if projected else it[idx]
            logit = model(qt[idx], ref)
            loss = loss_fn(logit, yt[idx], gt[idx])
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += float(loss.item()) * len(idx)
            seen += len(idx)
        train_loss = running / max(seen, 1)

        val_prob = _predict_packed(model, va, batch_size=max(batch_size, 8192))
        vm = prediction_metrics(va["y"], val_prob)
        if val_key == "regret":
            vm["regret"] = _val_regret(va, val_prob)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in vm.items()}}
        history.append(row)
        if verbose:
            print(
                f"[nirt] epoch {epoch:3d}  train_loss {train_loss:.4f}  "
                f"val_bce {vm['bce']:.4f}  val_spearman {vm['spearman_r']:.3f}"
                + (f"  val_regret {vm['regret']:.4f}" if "regret" in vm else "")
            )

        score = vm.get(val_key, vm["bce"])
        if sched is not None:
            sched.step(score) if lr_schedule == "plateau" else sched.step()
        if score < best_val - 1e-5:
            best_val = score
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                if verbose:
                    print(f"[nirt] early stop at epoch {epoch} (best {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    val_prob = _predict_packed(model, va, batch_size=max(batch_size, 8192))
    val_metrics = prediction_metrics(va["y"], val_prob)
    val_metrics["delta_bce_vs_model_mean"] = val_metrics["bce"] - baselines["model_mean"]["bce"]
    val_metrics["regret"] = _val_regret(va, val_prob)

    tag = "irt" if model.orientation == "model_latent" else "nirt"
    name = name or f"{tag}-{model.dim}d-{model.model_params}"
    result = RunResult(
        name=name,
        config=nirt_cfg,
        history=history,
        best_epoch=best_epoch,
        val_metrics=val_metrics,
        baselines=baselines,
        model=model,
        model_index=model_index,
    )

    if save:
        base = Path(runs_dir) if runs_dir is not None else _resolve_runs_dir(nirt_cfg, phase0_cfg)
        out = base / name
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": nirt_cfg,
                "n_models": len(model_index),
                "model_index": model_index,
                "query_dim": int(train_ds.query_dim),
                "profile_dim": int(train_ds.model_dim),
            },
            out / "model.pt",
        )
        (out / "run.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "git_sha": _git_sha(),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "config": nirt_cfg,
                    "n_models": len(model_index),
                    "model_index": model_index,
                    "best_epoch": best_epoch,
                    "history": history,
                    "val_metrics": val_metrics,
                    "baselines": baselines,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        result.path = out
        if verbose:
            print(f"[nirt] wrote {out}")

    return result


def _resolve_runs_dir(nirt_cfg: dict, phase0_cfg) -> Path:
    rel = nirt_cfg.get("runs_dir", _DEFAULT_RUNS_DIR)
    p = Path(rel)
    if p.is_absolute():
        return p
    if phase0_cfg is not None and hasattr(phase0_cfg, "resolve"):
        return phase0_cfg.resolve(rel)
    from ..config import REPO_ROOT

    return REPO_ROOT / rel


def load_run(name: str, runs_dir: Optional[os.PathLike] = None):
    """Return ``(model, config, model_index)`` for a saved run."""
    import torch

    base = Path(runs_dir) if runs_dir is not None else _resolve_runs_dir({}, None)
    blob = torch.load(base / name / "model.pt", map_location="cpu", weights_only=False)
    model = build_model(
        blob["config"].get("model", {}),
        n_models=blob["n_models"],
        query_dim=blob["query_dim"],
        profile_dim=blob["profile_dim"],
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob["config"], blob["model_index"]
