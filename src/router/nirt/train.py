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
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..determinism import seed_everything
from .metrics import marginal_baselines, prediction_metrics
from .model import build_model

_DEFAULT_RUNS_DIR = "data/processed/nirt_runs"

__all__ = ["fit", "load_run", "RunResult", "seed_everything"]


def _git_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover
        return None


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
    return {
        "q": np.ascontiguousarray(g["query_embedding"], dtype=np.float32),
        "m": np.ascontiguousarray(g["model_embedding"], dtype=np.float32),
        "y": np.ascontiguousarray(g["target"], dtype=np.float32),
        "midx": midx,
        "model_ids": model_ids,
    }


def _loss_fn(kind: str):
    import torch
    from torch.nn import functional as F

    if kind == "soft_bce":
        return lambda logit, y: F.binary_cross_entropy_with_logits(logit, y)
    if kind == "hard_bce":
        return lambda logit, y: F.binary_cross_entropy_with_logits(
            logit, (y >= 0.5).float()
        )
    if kind == "mse":
        return lambda logit, y: F.mse_loss(torch.sigmoid(logit), y)
    raise ValueError(f"unknown loss {kind!r}; use soft_bce | hard_bce | mse")


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
    loss_fn = _loss_fn(tcfg.get("loss", "soft_bce"))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    qt = torch.from_numpy(tr["q"])
    mt = torch.from_numpy(tr["m"])
    yt = torch.from_numpy(tr["y"])
    it = torch.from_numpy(tr["midx"])
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
        perm = torch.randperm(n, generator=gen)
        running = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            ref = mt[idx] if projected else it[idx]
            logit = model(qt[idx], ref)
            loss = loss_fn(logit, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(idx)
        train_loss = running / n

        val_prob = _predict_packed(model, va, batch_size=max(batch_size, 8192))
        vm = prediction_metrics(va["y"], val_prob)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in vm.items()}}
        history.append(row)
        if verbose:
            print(
                f"[nirt] epoch {epoch:3d}  train_loss {train_loss:.4f}  "
                f"val_bce {vm['bce']:.4f}  val_spearman {vm['spearman_r']:.3f}"
            )

        score = vm.get(val_key, vm["bce"])
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
