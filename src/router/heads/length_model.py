"""Predicting how long the answer will be, from the prompt alone.

A ridge regression on log tokens. Linear and boring on purpose: the question
this head answers is whether the *features* carry the signal, and a linear
probe over a frozen encoder answers that without the variance a fine-tune adds
(five seeds of one fine-tune config in this project spanned 4.3 points).

The prediction is a distribution, not a point. Residuals in log space are close
to Gaussian, so one number -- their standard deviation, fitted on validation --
turns the point estimate into ``P(tokens > T)`` for whatever threshold the
router cares about. That is cheaper and better calibrated than training a
second classifier per threshold, and the Gaussian assumption is checked rather
than assumed (see :func:`interval_coverage`).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class LengthModel:
    """Ridge over standardised features, plus the residual spread.

    ``alpha`` is chosen on validation, never on test -- with 1024 features over
    21k rows the difference between alpha 1 and alpha 1000 is visible, and
    picking it on test is how a held-out number stops being held out.
    """

    alpha: float = 10.0
    feature_set: str = "surface"
    scaler: Any = None
    ridge: Any = None
    #: Standard deviation of validation residuals, in log-token space.
    sigma: float = 0.0
    #: Quartile edges of the training target, for the bucket view.
    edges: list[float] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LengthModel:
        self.scaler = StandardScaler().fit(X)
        self.ridge = Ridge(alpha=self.alpha).fit(self.scaler.transform(X), y)
        self.edges = [float(q) for q in np.quantile(y, [0.25, 0.5, 0.75])]
        self.sigma = float(np.std(y - self.ridge.predict(self.scaler.transform(X))))
        return self

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray) -> LengthModel:
        """Refit the residual spread on held-out rows.

        Training residuals are optimistically small -- that is what fitting
        means -- so a ``P(tokens > T)`` built on them would be over-confident.
        """
        self.sigma = float(np.std(y_val - self.predict(X_val)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Expected log1p(tokens)."""
        return self.ridge.predict(self.scaler.transform(X))

    def expected_tokens(self, X: np.ndarray) -> np.ndarray:
        return np.expm1(self.predict(X))

    def probability_over(self, X: np.ndarray, tokens: float) -> np.ndarray:
        """``P(response is longer than `tokens`)`` under the fitted spread."""
        if self.sigma <= 0:
            return (self.predict(X) > np.log1p(tokens)).astype(float)
        return norm.sf(np.log1p(tokens), loc=self.predict(X), scale=self.sigma)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "length_model.pkl").open("wb") as fh:
            pickle.dump({"alpha": self.alpha, "feature_set": self.feature_set,
                         "scaler": self.scaler, "ridge": self.ridge,
                         "sigma": self.sigma, "edges": self.edges}, fh)

    @classmethod
    def load(cls, path: Path) -> LengthModel:
        with (Path(path) / "length_model.pkl").open("rb") as fh:
            state = pickle.load(fh)
        return cls(**state)


def interval_coverage(y_true: np.ndarray, y_pred: np.ndarray, sigma: float) -> dict[str, float]:
    """How often the true length falls inside the predicted interval.

    The claim being checked is that residuals are Gaussian enough for
    ``probability_over`` to mean something. If the 80% interval holds 80% of
    the rows, it does; if it holds 60%, the head is over-confident and the
    probabilities are decoration.
    """
    if sigma <= 0:
        return {}
    z = (y_true - y_pred) / sigma
    return {f"coverage@{int(level * 100)}": float(np.mean(np.abs(z) <= norm.ppf(0.5 + level / 2)))
            for level in (0.5, 0.8, 0.95)}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, *, sigma: float = 0.0,
             long_threshold: float | None = None) -> dict[str, float]:
    """Ranking, error and gating metrics for one split.

    Spearman leads because the router's decision is ordinal -- is this a bigger
    job than that one -- and because it is the number the earlier feature
    survey reported, so the two are comparable.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    # A constant prediction has no ranking to correlate. That is a real answer
    # about the constant baseline, not an error, so it is reported as NaN
    # rather than raised or warned about.
    ranked = float("nan") if np.ptp(y_pred) == 0 or np.ptp(y_true) == 0 else float(
        spearmanr(y_true, y_pred).statistic)
    out = {
        "n": int(len(y_true)),
        "spearman": ranked,
        "r2": float(r2_score(y_true, y_pred)),
        "mae_log": mae,
        # exp(MAE) in log space is the typical multiplicative error: 1.6 means
        # the true length is usually within a factor of 1.6 of the estimate.
        "typical_factor": float(np.exp(mae)),
    }
    if long_threshold is not None:
        is_long = (y_true >= long_threshold).astype(int)
        if 0 < is_long.sum() < len(is_long):
            out["auc_long"] = float(roc_auc_score(is_long, y_pred))
    out.update(interval_coverage(y_true, y_pred, sigma))
    return out


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, *, metric: str = "spearman",
                 n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Percentile CI over resampled rows -- the spread that matters is across
    prompts, not across folds."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(evaluate(y_true[idx], y_pred[idx])[metric])
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
