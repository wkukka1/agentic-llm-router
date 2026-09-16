"""Load a saved NIRT / IRT-Router run's checkpoint.

Split out of ``training.nirt.train`` (which still owns ``fit``) so that
serving code -- :meth:`router.routing.routers.NIRTRouter.from_run` -- can
reconstruct a fitted model without importing anything training-only.

``model.pt`` holds a ``format="nirt"`` tag distinguishing it from the unrelated
Phase 1/2 baseline ``model.pt`` written by ``training.nirt.baseline.train.fit``
(loaded by :func:`training.nirt.baseline.checkpoint.load_baseline_run`) -- the
two formats share a filename but not a key set, so pointing this loader at a
baseline run used to fail with an opaque ``KeyError`` instead of naming the
mismatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import os

from .model import build_model

__all__ = ["load_run"]

FORMAT = "nirt"


def load_run(name: str, runs_dir: Optional[os.PathLike] = None):
    """Return ``(model, config, model_index)`` for a saved run."""
    import torch

    if runs_dir is not None:
        base = Path(runs_dir)
    else:
        from router.config import DEFAULT_NIRT_RUNS_DIR, REPO_ROOT

        base = REPO_ROOT / DEFAULT_NIRT_RUNS_DIR
    path = base / name / "model.pt"
    # weights_only: the blob is tensors + plain dicts/ints, and a run directory
    # may be shared/downloaded -- never unpickle arbitrary objects on the serving path
    blob = torch.load(path, map_location="cpu", weights_only=True)
    fmt = blob.get("format")
    if fmt is not None and fmt != FORMAT:
        raise ValueError(
            f"{path}: format={fmt!r}, not a NIRT/IRT-Router checkpoint written by "
            "training.nirt.train.fit -- use "
            "training.nirt.baseline.checkpoint.load_baseline_run for a Phase 1/2 "
            "baseline run instead"
        )
    try:
        model = build_model(
            blob["config"].get("model", {}),
            n_models=blob["n_models"],
            query_dim=blob["query_dim"],
            profile_dim=blob["profile_dim"],
        )
    except KeyError as e:
        raise ValueError(
            f"{path}: missing key {e}; this does not look like a NIRT/IRT-Router "
            "checkpoint written by training.nirt.train.fit (maybe a Phase 1/2 "
            "baseline run? use "
            "training.nirt.baseline.checkpoint.load_baseline_run instead)"
        ) from e
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob["config"], blob["model_index"]
