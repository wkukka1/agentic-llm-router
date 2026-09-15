"""One deterministic-seeding helper, shared by every subpackage.

Encoder inference and NIRT training both need the same "pin every RNG" call;
keeping it here avoids two copies that can drift apart.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_algorithms: bool = True) -> None:
    """Seed ``random`` / numpy / torch.

    ``deterministic_algorithms=True`` also calls
    ``torch.use_deterministic_algorithms`` -- a *process-wide* switch (slower
    CUDA kernels). Training entry points want it; serving-side encoder loads
    pass ``False`` so they don't flip it for a co-located training / GPU job.

    ``PYTHONHASHSEED`` is only set for *child* processes: hash randomisation of
    the current interpreter is fixed at start-up, and ``setdefault`` leaves an
    inherited value alone.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_algorithms:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover
        pass
