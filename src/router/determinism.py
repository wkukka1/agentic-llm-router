"""One deterministic-seeding helper, shared by every subpackage.

Encoder inference and NIRT training both need the same "pin every RNG" call;
keeping it here avoids two copies that can drift apart.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover
        pass
