"""One encoder pass, eleven calibrated binary heads.

Each attribute gets its own logistic head over shared features. Independent
heads rather than one multi-label model because the attributes are not mutually
exclusive and their base rates span 0.075 to 0.819 -- a single model with one
decision rule would be tuned for the average of a set whose members have nothing
in common but their input.

Probabilities are the contract, not labels. A gate's threshold belongs to the
caller: the cost of a false "needs code execution" is not the same in every
deployment, and this module has no way to know it.

That contract only holds if the probabilities are calibrated, and these start
badly miscalibrated on purpose. `class_weight="balanced"` is what lifts an
attribute that is 7.5% of traffic, and it fits the head as though the classes
were even -- so its raw output is shifted towards the positive class by roughly
the log odds of the base rate. **Temperature scaling cannot fix that**: it
rescales the logit but cannot move it, and the error here is an offset. Platt
scaling fits both a slope and an intercept on validation, which is why it is
what runs here. Measured over the eleven heads it takes mean ECE from 0.115 to
0.013, leaving every ranking metric untouched because it is monotone.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prompt_decomposition.attribute_classifier.taxonomy import ATTRIBUTE_NAMES

_EPSILON = 1e-9


def _log_odds(proba: np.ndarray) -> np.ndarray:
    """The head's score on the logit scale, as a column, for the calibrator."""
    clipped = np.clip(proba, _EPSILON, 1 - _EPSILON)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


@dataclass(slots=True)
class AttributeModel:
    """A logistic head per attribute, each with its own temperature."""

    names: tuple[str, ...] = ATTRIBUTE_NAMES
    C: float = 1.0
    max_iter: int = 1000
    heads: dict[str, Any] = field(default_factory=dict)
    #: One-dimensional logistic fitted on validation log-odds, per head.
    calibrators: dict[str, Any] = field(default_factory=dict)
    #: Base rate in training, so a caller can see what "better than nothing"
    #: means for each head without going back to the corpus.
    base_rates: dict[str, float] = field(default_factory=dict)

    def fit(self, X: np.ndarray, y: dict[str, np.ndarray]) -> AttributeModel:
        """Fit one head per attribute, skipping rows the judge did not label."""
        for name in self.names:
            if name not in y:
                continue
            labels = np.asarray(y[name], dtype=float)
            known = ~np.isnan(labels)
            if known.sum() < 100 or len(np.unique(labels[known])) < 2:
                continue
            target = labels[known].astype(int)
            # Scaled, because the feature matrix mixes an L2-normalised
            # embedding with raw surface counts whose scales differ by orders of
            # magnitude. Unscaled, the solver does not converge in any
            # reasonable iteration budget and the head is quietly under-fitted.
            self.heads[name] = Pipeline([
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(C=self.C, max_iter=self.max_iter,
                                                class_weight="balanced")),
            ]).fit(X[known], target)
            self.base_rates[name] = float(target.mean())
        return self

    def calibrate(self, X: np.ndarray, y: dict[str, np.ndarray]) -> AttributeModel:
        """Fit each head's Platt calibrator on held-out rows.

        Slope *and* intercept, because the miscalibration here is an offset --
        see the module docstring. Fitted on validation, never on the rows the
        head was trained on, where its confidence is optimistic by
        construction.
        """
        for name, head in self.heads.items():
            labels = np.asarray(y[name], dtype=float)
            known = ~np.isnan(labels)
            target = labels[known].astype(int)
            if known.sum() < 100 or len(np.unique(target)) < 2:
                continue
            self.calibrators[name] = LogisticRegression(C=1e6, max_iter=1000).fit(
                _log_odds(head.predict_proba(X[known])[:, 1]), target)
        return self

    def predict_proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Calibrated ``P(attribute)`` per head, one array each."""
        out: dict[str, np.ndarray] = {}
        for name, head in self.heads.items():
            raw = head.predict_proba(X)[:, 1]
            calibrator = self.calibrators.get(name)
            out[name] = (raw if calibrator is None
                         else calibrator.predict_proba(_log_odds(raw))[:, 1])
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "attributes.pkl").open("wb") as fh:
            pickle.dump({"names": tuple(self.names), "C": self.C, "max_iter": self.max_iter,
                         "heads": self.heads, "calibrators": self.calibrators,
                         "base_rates": self.base_rates}, fh)

    @classmethod
    def load(cls, path: Path) -> AttributeModel:
        """Restore a saved run, refusing one this class cannot serve.

        An artifact written before the calibration change carries
        ``temperatures`` where this expects ``calibrators``, and unpacking it
        blind raises a bare TypeError naming a keyword argument -- true, and
        useless. Saved state that does not match is a stale artifact, and the
        fix is to retrain, so the error says that.
        """
        with (Path(path) / "attributes.pkl").open("rb") as fh:
            state = pickle.load(fh)
        expected = {f.name for f in fields(cls)}
        unknown = set(state) - expected
        if unknown:
            raise ValueError(
                f"{path} was written by an incompatible version of this class "
                f"(unexpected {sorted(unknown)}). Retrain it: "
                f"`prompt-decomposition attributes`."
            )
        return cls(**state)
