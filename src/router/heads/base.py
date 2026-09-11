"""What every head shares: loading a run directory and calibrating its output.

A head is one classifier served from one trained run directory. Adding a new
one -- difficulty, tool-need, decomposability -- means a new module next to this
one that subclasses :class:`CalibratedHead`, not an edit here. The shared part
is deliberately small: the saved model, the temperature the runner fitted on
validation, and a defer threshold that only means anything because of that
temperature.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from router.models import build


class CalibratedHead:
    """Loads a run directory and serves temperature-scaled probabilities.

    Shared by both heads because both need the same three things: the saved
    model, the temperature the runner fitted on validation, and a defer
    threshold that means something only because of that temperature.
    """

    def __init__(self, run_dir: str | Path, *, defer_below: float = 0.0) -> None:
        self.run_dir = Path(run_dir)
        self.defer_below = defer_below
        config = yaml.safe_load((self.run_dir / "config.yaml").read_text(encoding="utf-8"))
        metrics = json.loads((self.run_dir / "metrics.json").read_text(encoding="utf-8"))
        # The temperature is fitted on the VALIDATION split -- see
        # `router.experiment`, which calls `fit_temperature(val_proba, ...)`
        # before scoring test. It is *stored* under the "test" key only because
        # it sits alongside the test metrics it was used to produce, which
        # reads as though it were fitted there. New runs also write it to a
        # top-level "calibration" block with its provenance attached; prefer
        # that, and fall back for artifacts written before that existed.
        calibration = metrics.get("calibration") or {}
        self.temperature = float(
            calibration.get("temperature", metrics.get("test", {}).get("temperature", 1.0))
        )
        #: Which split the temperature was fitted on. Unknown for older runs,
        #: which is not the same as "test" -- they were validation-fitted too,
        #: they just did not record it.
        self.temperature_fitted_on = calibration.get("fitted_on", "validation")
        params = dict(config["model"].get("params") or {})
        params.pop("cache_tag", None)
        self.model = build(config["model"]["name"], **params)
        self.model.load(self.run_dir / "model")

    @property
    def labels(self) -> list[str]:
        # Members may hand back numpy string arrays; coerce so callers get
        # plain str and JSON serialisation works.
        return [str(x) for x in self.model.labels]

    def _calibrated(self, proba: np.ndarray) -> np.ndarray:
        if self.temperature == 1.0:
            return proba
        logits = np.log(np.clip(proba, 1e-12, None)) / max(self.temperature, 1e-12)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats, 0 when certain and log(k) when uniform."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())
