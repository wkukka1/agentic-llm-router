"""Load a saved NIRT / IRT-Router run's checkpoint.

Split out of ``training.nirt.train`` (which still owns ``fit``) so that
serving code -- :meth:`router.routing.routers.NIRTRouter.from_run` -- can
reconstruct a fitted model without importing anything training-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import os

from .model import build_model

__all__ = ["load_run"]


def load_run(name: str, runs_dir: Optional[os.PathLike] = None):
    """Return ``(model, config, model_index)`` for a saved run."""
    import torch

    if runs_dir is not None:
        base = Path(runs_dir)
    else:
        from router.config import DEFAULT_NIRT_RUNS_DIR, REPO_ROOT

        base = REPO_ROOT / DEFAULT_NIRT_RUNS_DIR
    # weights_only: the blob is tensors + plain dicts/ints, and a run directory
    # may be shared/downloaded -- never unpickle arbitrary objects on the serving path
    blob = torch.load(base / name / "model.pt", map_location="cpu", weights_only=True)
    model = build_model(
        blob["config"].get("model", {}),
        n_models=blob["n_models"],
        query_dim=blob["query_dim"],
        profile_dim=blob["profile_dim"],
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob["config"], blob["model_index"]
