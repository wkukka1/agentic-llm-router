"""Phase 1 checkpoint format.

A run directory holds ``model.pt`` (state_dict + rebuild shapes), ``config.yaml``
(resolved run config), ``metrics.json``, ``training_history.json`` and
``provenance.json`` (seed, git sha, phase-0 artefact hashes, dataset stats).
``load_checkpoint`` rebuilds the model without the training data; ``load_run``
also returns the label-construction settings from ``config.yaml``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml

from router.provenance import git_sha

from .model import build_baseline_model

MODEL_PT = "model.pt"

__all__ = ["MODEL_PT", "git_sha", "save_checkpoint", "load_checkpoint", "load_run"]


def save_checkpoint(
    directory: str | Path, *, model, config: dict, model_index: dict,
    query_dim: int, profile_dim: int, relevance_dim: int,
    metrics: Optional[dict] = None, history: Optional[list] = None,
    provenance: Optional[dict] = None,
) -> Path:
    import torch

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "arch": getattr(model, "arch", "baseline"),
        "model_cfg": config.get("model", {}),
        "model_index": model_index,
        "query_dim": int(query_dim), "profile_dim": int(profile_dim),
        "relevance_dim": int(relevance_dim),
        "dim": int(model.dim), "model_params": model.model_params,
    }, d / MODEL_PT)
    (d / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for name, payload in (("metrics.json", metrics), ("training_history.json", history)):
        if payload is not None:
            (d / name).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    if provenance is not None:
        (d / "provenance.json").write_text(
            json.dumps({"git_sha": git_sha(), **provenance}, indent=2, default=float), encoding="utf-8")
    return d


def load_checkpoint(directory: str | Path, *, map_location: str = "cpu"):
    """Return ``(model, blob)`` -- ``model`` is a rebuilt, eval-mode BaselineNIRT."""
    import torch

    blob = torch.load(Path(directory) / MODEL_PT, map_location=map_location, weights_only=False)
    model = build_baseline_model(
        blob["model_cfg"], n_models=len(blob["model_index"]), query_dim=blob["query_dim"],
        profile_dim=blob["profile_dim"], relevance_dim=blob["relevance_dim"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def load_run(directory: str | Path):
    """``(model, blob, settings)`` -- ``settings`` = the label-construction knobs
    from ``config.yaml`` (``pathway, binary_threshold, score_kind``). Whether
    relevance / warm-up are active is read straight off ``model``."""
    model, blob = load_checkpoint(directory)
    p = Path(directory) / "config.yaml"
    full = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    resp = full.get("response", {}) or {}
    settings = {
        "pathway": (full.get("data", {}) or {}).get("pathway", "irt"),
        "binary_threshold": float(resp.get("binary_threshold", 0.5)),
        "score_kind": resp.get("score_kind", "effective"),
    }
    return model, blob, settings
