"""A Random Forest over domain, task type and the encoded query.

The question this answers is not "can a forest route" -- four measurements say
the routing decision is close to unpredictable from a prompt -- but **how much
each feature family contributes**, which needs a model that can report that.

Three families, deliberately kept separable:

* ``domain``     -- the domain head's calibrated distribution (8 or 10 columns)
* ``task``       -- the task head's calibrated distribution (6 or 7 columns)
* ``embedding``  -- the encoded query, the raw text pathway (384 or 1024)

Grown with ``criterion="entropy"``, so the impurity importances the forest
reports *are* information gain rather than a Gini proxy. That is the number
asked for; it is also the number most likely to mislead, because impurity
importance rewards a family for being wide and continuous. A 384-column
embedding gets 384 chances to be chosen at every split and the 8-column domain
block gets 8. :mod:`evaluation.prompt_decomposition.forest_report` therefore
reports permutation importance on held-out rows and family ablations alongside
it, and those are the numbers to trust when they disagree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier

log = logging.getLogger(__name__)

FEATURE_FAMILIES: tuple[str, ...] = ("domain", "task", "embedding")


@dataclass(slots=True)
class ForestFeatures:
    """A feature matrix that remembers which family each column came from."""

    X: np.ndarray
    names: list[str] = field(default_factory=list)
    #: Family per column, parallel to :attr:`names`. Kept beside the matrix
    #: rather than recomputed from name prefixes, which is one rename away from
    #: silently regrouping every importance number in the report.
    families: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (self.X.shape[1] == len(self.names) == len(self.families)):
            raise ValueError(
                f"width {self.X.shape[1]}, {len(self.names)} names, "
                f"{len(self.families)} families -- these must agree")

    def columns_for(self, family: str) -> np.ndarray:
        """Indices of one family's columns, for ablations and grouping."""
        return np.array([i for i, f in enumerate(self.families) if f == family], dtype=int)

    def without(self, family: str) -> ForestFeatures:
        """The same features minus one family."""
        keep = [i for i, f in enumerate(self.families) if f != family]
        return ForestFeatures(self.X[:, keep], [self.names[i] for i in keep],
                              [self.families[i] for i in keep])

    def only(self, family: str) -> ForestFeatures:
        keep = self.columns_for(family)
        return ForestFeatures(self.X[:, keep], [self.names[i] for i in keep],
                              [self.families[i] for i in keep])

    @property
    def present_families(self) -> list[str]:
        return [f for f in FEATURE_FAMILIES if f in set(self.families)]


def build_features(*, domain_proba: np.ndarray | None = None,
                   domain_labels: list[str] | None = None,
                   task_proba: np.ndarray | None = None,
                   task_labels: list[str] | None = None,
                   embeddings: np.ndarray | None = None) -> ForestFeatures:
    """Assemble the three families into one matrix, names and groups attached.

    Distributions rather than argmax labels: the shape carries what the label
    does not -- a prompt split 0.45/0.44 between two domains is a different
    routing problem from one at 0.89, and both have the same argmax.
    """
    blocks, names, families = [], [], []
    for proba, labels, family in ((domain_proba, domain_labels, "domain"),
                                  (task_proba, task_labels, "task")):
        if proba is None:
            continue
        if labels is None or len(labels) != proba.shape[1]:
            raise ValueError(f"{family}: need one label per column of the distribution")
        blocks.append(np.asarray(proba, dtype=float))
        names += [f"{family}.{label}" for label in labels]
        families += [family] * proba.shape[1]
    if embeddings is not None:
        embeddings = np.asarray(embeddings, dtype=float)
        blocks.append(embeddings)
        names += [f"embedding.{i}" for i in range(embeddings.shape[1])]
        families += ["embedding"] * embeddings.shape[1]
    if not blocks:
        raise ValueError("no feature families given")
    return ForestFeatures(np.hstack(blocks), names, families)


def fit_forest(features: ForestFeatures, y: np.ndarray, *, seed: int = 20260824,
               n_estimators: int = 400, min_samples_leaf: int = 5,
               max_features: str = "sqrt", n_jobs: int = -1) -> RandomForestClassifier:
    """Fit the forest.

    ``criterion="entropy"`` so the reported importances are information gain.
    ``min_samples_leaf`` is not the default 1: a forest grown to pure leaves on
    a noisy binary target memorises the training rows and reports importances
    for splits that are fitting noise, which is exactly the failure this
    experiment would otherwise hide behind a plausible-looking chart.
    """
    forest = RandomForestClassifier(
        n_estimators=n_estimators, criterion="entropy", min_samples_leaf=min_samples_leaf,
        max_features=max_features, class_weight="balanced_subsample",
        random_state=seed, n_jobs=n_jobs,
    )
    forest.fit(features.X, y)
    log.info("forest: %d trees over %d features, %d rows",
             n_estimators, features.X.shape[1], len(y))
    return forest
