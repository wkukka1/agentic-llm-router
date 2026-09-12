"""Is the length head learning, memorising, or under-powered?

Three questions, three checks, and they answer different things:

* **Permutation** -- shuffle the targets and refit. Whatever score survives is
  what the pipeline produces from nothing. This is the check that catches a
  leak, and the one most often skipped.
* **Learning curve** -- fit on a growing share of the training rows. A test
  score still climbing at 100% means more data would help (under-fitting by
  starvation); a flat one means the features, not the volume, are the limit.
* **Regularisation sweep** -- alpha across decades, scored on train and
  validation. Small alpha with a wide train/val gap is over-fitting; large
  alpha with both scores low together is under-fitting. Seeing both ends makes
  the chosen alpha a position on a curve rather than a guess.

There is also a ceiling to measure. The two arena models, answering the *same*
prompt, agree on length at Spearman 0.592. No prompt-only feature can do better
than the target's own reproducibility, so a score is worth reading against that
number and not against 1.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from prompt_decomposition.length_estimator.model import LengthModel

log = logging.getLogger(__name__)


@dataclass(slots=True)
class LengthAudit:
    """One feature set's diagnosis."""

    name: str
    n_train: int
    n_features: int
    train_score: float
    test_score: float
    #: (n_rows, train score, test score) at each step of the learning curve.
    learning_curve: list[tuple[int, float, float]] = field(default_factory=list)
    #: (alpha, train score, val score) across the regularisation sweep.
    regularisation: list[tuple[float, float, float]] = field(default_factory=list)
    shuffled_scores: list[float] = field(default_factory=list)

    @property
    def gap(self) -> float:
        """Train minus test R-squared. Large and positive is memorisation."""
        return self.train_score - self.test_score

    @property
    def permutation_p(self) -> float:
        """Add-one empirical p: how often shuffled labels matched the real run."""
        beaten = sum(1 for s in self.shuffled_scores if s >= self.test_score)
        return (beaten + 1) / (len(self.shuffled_scores) + 1)

    @property
    def curve_slope(self) -> float:
        """Test-score change over the last doubling of training data.

        Near zero means more rows will not help; clearly positive means the
        head is data-starved and the cheapest improvement available is more of
        the corpus, which for this target costs nothing to collect.
        """
        if len(self.learning_curve) < 2:
            return 0.0
        return self.learning_curve[-1][2] - self.learning_curve[-2][2]

    @property
    def verdict(self) -> str:
        """Plain reading of the three checks together."""
        if self.permutation_p > 0.05:
            return "no signal (indistinguishable from shuffled labels)"
        if self.gap > 0.15:
            return "over-fitting (train far above test)"
        if self.curve_slope > 0.01:
            return "under-fed (still climbing on data)"
        return "fitted (gap small, curve flat, beats permutation)"

    def summary(self) -> str:
        lines = [
            f"{self.name}: train {self.train_score:.3f} / test {self.test_score:.3f} "
            f"(gap {self.gap:+.3f}, {self.n_features} features over {self.n_train} rows)",
            f"  permutation: real {self.test_score:.3f} vs shuffled "
            f"{np.mean(self.shuffled_scores):+.3f} (sd {np.std(self.shuffled_scores):.3f}), "
            f"p = {self.permutation_p:.3f}",
            "  learning curve: " + " -> ".join(
                f"{n}:{test:.3f}" for n, _, test in self.learning_curve),
            f"  last step {self.curve_slope:+.3f}  =>  {self.verdict}",
        ]
        return "\n".join(lines)


def _score(model: LengthModel, X: np.ndarray, y: np.ndarray) -> float:
    """R-squared, not the headline Spearman.

    Rank correlation is invariant to the sign and scale of the fit, so a
    one-feature model scores |0.26| under *shuffled* labels -- the coefficient
    collapses towards zero but the predicted ordering is still prompt length,
    up to a sign. A permutation test in rank space therefore cannot fail, which
    makes it useless. R-squared is sign- and scale-sensitive: a model fitted on
    noise scores at or below zero on real targets. Every number in this audit
    is R-squared for that reason; the Spearman figures are in the evaluation
    table alongside.
    """
    return float(r2_score(y, model.predict(X)))


def audit(X_train, y_train, X_val, y_val, X_test, y_test, *, name: str,
          alpha: float = 10.0, permutations: int = 30,
          alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
          seed: int = 0) -> LengthAudit:
    """Fit once, then run the three checks around it."""
    model = LengthModel(alpha=alpha).fit(X_train, y_train)
    result = LengthAudit(
        name=name, n_train=len(y_train), n_features=X_train.shape[1],
        train_score=_score(model, X_train, y_train),
        test_score=_score(model, X_test, y_test),
    )

    # Random subsets, not the first n rows: the corpus arrives in arena dump
    # order, so a prefix is a time slice and a curve over prefixes measures
    # drift as much as volume.
    order = np.random.default_rng(seed).permutation(len(y_train))
    for fraction in (0.1, 0.25, 0.5, 1.0):
        n = max(50, int(len(y_train) * fraction))
        rows = order[:n]
        step = LengthModel(alpha=alpha).fit(X_train[rows], y_train[rows])
        result.learning_curve.append(
            (n, _score(step, X_train[rows], y_train[rows]), _score(step, X_test, y_test))
        )

    for a in alphas:
        step = LengthModel(alpha=a).fit(X_train, y_train)
        result.regularisation.append(
            (a, _score(step, X_train, y_train), _score(step, X_val, y_val))
        )

    rng = np.random.default_rng(seed)
    for _ in range(permutations):
        shuffled = rng.permutation(y_train)
        step = LengthModel(alpha=alpha).fit(X_train, shuffled)
        result.shuffled_scores.append(_score(step, X_test, y_test))

    log.info("%s", result.summary())
    return result


def target_ceiling(tokens_a: np.ndarray, tokens_b: np.ndarray) -> float:
    """Spearman between the two models' lengths on the same prompts.

    The reproducibility of the target, and therefore the practical ceiling on
    predicting it from the prompt.
    """
    return float(spearmanr(np.log1p(tokens_a), np.log1p(tokens_b)).statistic)
