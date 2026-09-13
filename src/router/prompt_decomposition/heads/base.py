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

from router.prompt_decomposition.calibration import apply_temperature
from router.prompt_decomposition.models import build


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
        # `training.prompt_decomposition.experiment`, which calls `fit_temperature(val_proba, ...)`
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
        """Temperature-scaled probabilities.

        One implementation, shared with the experiment runner that fitted the
        temperature: serving and evaluation must not be able to disagree about
        what it means. ``T == 1.0`` returns the input untouched rather than
        round-tripping it through log and exp.
        """
        if self.temperature == 1.0:
            return proba
        return apply_temperature(proba, self.temperature)

    def _calibrated_proba(self, prompts: list[str]) -> np.ndarray:
        """Calibrated probabilities for a batch, an empty one included.

        An empty batch is a legitimate serving call -- a filter upstream
        removed everything -- and sklearn raises on a zero-row matrix. Both
        heads carried that guard; it lives here instead.
        """
        prompts = list(prompts)
        if not prompts:
            return np.zeros((0, len(self.labels)))
        return self._calibrated(self.model.predict_proba(prompts))
