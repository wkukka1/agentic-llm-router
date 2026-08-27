"""Decorrelation and dimensionality reduction for dense features.

Embedding dimensions are heavily correlated -- a 384-d encoder does not carry
384 independent facts about a prompt -- so a linear head spends capacity on
redundancy. Two tools address that, and the order they are applied in matters:

* **VIF pruning** removes features that other features already explain.
  ``VIF_i = 1 / (1 - R²_i)`` where ``R²_i`` is from regressing feature *i* on the
  rest. High VIF means redundant.
* **PCA** rotates into an orthogonal basis and keeps the top components.

**Applying VIF after PCA is a no-op.** Principal components are orthogonal by
construction, so every ``R²`` is 0 and every VIF is exactly 1.0 -- nothing is
ever dropped. Verified empirically in ``tests/test_reduction.py``. The two
useful orders are therefore:

* ``vif_then_pca`` -- prune redundant raw dimensions, then rotate. VIF does real
  work here, and PCA runs on a cleaner matrix.
* ``pca_then_vif`` -- only meaningful when non-PCA features (token counts,
  hardness scores) are concatenated onto the components afterwards, since those
  *can* correlate with them. Supported, and it reports how many it dropped so
  the no-op case is visible rather than silent.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def variance_inflation_factors(X: np.ndarray) -> np.ndarray:
    """VIF per column, via the inverse correlation matrix.

    ``diag(R⁻¹)`` equals the VIF vector, which is far cheaper than fitting one
    regression per feature. A pseudo-inverse keeps perfectly collinear columns
    from raising.
    """
    if X.shape[1] < 2:
        return np.ones(X.shape[1])
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return np.abs(np.diag(np.linalg.pinv(corr)))


class VIFPruner(BaseEstimator, TransformerMixin):
    """Iteratively drop the single highest-VIF column until all are below cut.

    Dropping one at a time (rather than everything above the threshold at once)
    matters: VIF is computed *jointly*, so removing one redundant feature often
    drops its partners' VIF below the cut on its own. Batch-dropping discards
    information the one-at-a-time loop keeps.
    """

    def __init__(self, threshold: float = 10.0, max_drop: int | None = None) -> None:
        self.threshold = threshold
        self.max_drop = max_drop

    def fit(self, X: np.ndarray, y=None) -> VIFPruner:
        X = np.asarray(X, dtype=np.float64)
        keep = list(range(X.shape[1]))
        dropped: list[tuple[int, float]] = []
        budget = self.max_drop if self.max_drop is not None else X.shape[1] - 1

        while len(keep) > 1 and len(dropped) < budget:
            vifs = variance_inflation_factors(X[:, keep])
            worst = int(np.argmax(vifs))
            if vifs[worst] <= self.threshold:
                break
            dropped.append((keep[worst], float(vifs[worst])))
            keep.pop(worst)

        self.keep_indices_ = np.array(keep, dtype=int)
        self.dropped_ = dropped
        self.n_features_in_ = X.shape[1]
        log.info(
            "VIF pruning: kept %d/%d columns (threshold %.1f, max VIF dropped %.1f)",
            len(keep), X.shape[1], self.threshold,
            max((v for _, v in dropped), default=float("nan")),
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[:, self.keep_indices_]


class DenseReducer(BaseEstimator, TransformerMixin):
    """Standardise, then decorrelate via PCA and/or VIF in a chosen order."""

    def __init__(
        self,
        *,
        order: str = "vif_then_pca",
        n_components: int | float | None = 0.95,
        vif_threshold: float = 10.0,
        max_vif_drop: int | None = 200,
        standardize: bool = True,
    ) -> None:
        self.order = order
        self.n_components = n_components
        self.vif_threshold = vif_threshold
        self.max_vif_drop = max_vif_drop
        self.standardize = standardize

    def _make_pca(self) -> PCA | None:
        return None if self.n_components is None else PCA(n_components=self.n_components, random_state=0)

    def fit(self, X: np.ndarray, y=None) -> DenseReducer:
        if self.order not in ("vif_then_pca", "pca_then_vif", "pca_only", "vif_only"):
            raise ValueError(f"unknown order {self.order!r}")

        X = np.asarray(X, dtype=np.float64)
        self.scaler_ = StandardScaler().fit(X) if self.standardize else None
        Z = self.scaler_.transform(X) if self.scaler_ is not None else X

        self.pruner_ = None
        self.pca_ = None

        if self.order in ("vif_then_pca", "vif_only"):
            self.pruner_ = VIFPruner(self.vif_threshold, self.max_vif_drop).fit(Z)
            Z = self.pruner_.transform(Z)
        if self.order in ("vif_then_pca", "pca_then_vif", "pca_only"):
            self.pca_ = self._make_pca()
            if self.pca_ is not None:
                Z = self.pca_.fit_transform(Z)
        if self.order == "pca_then_vif":
            # Expected to drop nothing on pure PCA output; kept so the no-op is
            # observable rather than assumed.
            self.pruner_ = VIFPruner(self.vif_threshold, self.max_vif_drop).fit(Z)
            Z = self.pruner_.transform(Z)

        self.n_features_out_ = Z.shape[1]
        log.info("DenseReducer[%s]: %d -> %d features", self.order, X.shape[1], self.n_features_out_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = np.asarray(X, dtype=np.float64)
        if self.scaler_ is not None:
            Z = self.scaler_.transform(Z)
        if self.order in ("vif_then_pca", "vif_only") and self.pruner_ is not None:
            Z = self.pruner_.transform(Z)
        if self.pca_ is not None:
            Z = self.pca_.transform(Z)
        if self.order == "pca_then_vif" and self.pruner_ is not None:
            Z = self.pruner_.transform(Z)
        return Z

    def diagnostics(self) -> dict:
        """What the reduction actually did -- for reporting, not for control flow."""
        info: dict = {"order": self.order, "n_features_out": getattr(self, "n_features_out_", None)}
        if self.pruner_ is not None:
            info["vif_dropped"] = len(self.pruner_.dropped_)
            info["vif_max_dropped"] = max((v for _, v in self.pruner_.dropped_), default=None)
        if self.pca_ is not None:
            info["pca_components"] = int(self.pca_.n_components_)
            info["explained_variance"] = float(self.pca_.explained_variance_ratio_.sum())
        return info


def correlation_report(X: np.ndarray, *, vif_threshold: float = 10.0, top_n: int = 15) -> str:
    """Describe the redundancy structure of a feature matrix.

    This is the diagnostic use of VIF and PCA -- understanding *which*
    dimensions carry independent signal -- as opposed to using them as a
    preprocessing step, which on L2-normalised embeddings measurably hurts.
    """
    X = np.asarray(X, dtype=np.float64)
    n_rows, n_cols = X.shape

    vifs = variance_inflation_factors(X)
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    off_diagonal = corr[~np.eye(n_cols, dtype=bool)]

    pca_full = PCA().fit(X)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    thresholds = {p: int(np.searchsorted(cumulative, p) + 1) for p in (0.80, 0.90, 0.95, 0.99)}

    worst = np.argsort(-vifs)[:top_n]
    rows = "\n".join(f"| {int(i)} | {vifs[i]:.1f} |" for i in worst)

    # Effective rank: how many dimensions the data actually occupies, via the
    # entropy of the normalised eigenvalue spectrum.
    spectrum = pca_full.explained_variance_ratio_
    spectrum = spectrum[spectrum > 0]
    effective_rank = float(np.exp(-(spectrum * np.log(spectrum)).sum()))

    return f"""# Feature correlation diagnostics

{n_rows} rows x {n_cols} dimensions.

## Redundancy

| measure | value |
|---|---|
| dimensions with VIF > {vif_threshold:g} | **{int((vifs > vif_threshold).sum())} / {n_cols}** |
| median VIF | {np.median(vifs):.2f} |
| max VIF | {vifs.max():.1f} |
| mean abs pairwise correlation | {np.abs(off_diagonal).mean():.3f} |
| max abs pairwise correlation | {np.abs(off_diagonal).max():.3f} |
| effective rank (spectral entropy) | **{effective_rank:.1f}** of {n_cols} |

Effective rank is the honest count of independent directions: {effective_rank:.0f}
of {n_cols} nominal dimensions.

## Components needed for a variance target

| variance kept | components |
|---|---|
""" + "\n".join(f"| {int(p * 100)}% | {k} |" for p, k in thresholds.items()) + f"""

## Highest-VIF dimensions

These are the most redundant -- each is well predicted by the others.

| dimension | VIF |
|---|---|
{rows}

## How to read this

VIF and pairwise correlation disagree here, and the disagreement is the point.
Max pairwise correlation is low, which by itself suggests "no redundancy" --
but nearly every dimension has a very high VIF. That combination means the
redundancy is **multivariate**: no two dimensions duplicate each other, yet
each one is almost perfectly predicted by a *linear combination* of the rest.

Pairwise correlation cannot see this; VIF can. It is also why "drop the most
correlated column" is the wrong instinct on embeddings -- there is no single
correlated partner to drop. Effective rank is the number that actually matters:
it says how many independent directions the data occupies.

## Caveat

Dropping these does **not** improve a downstream classifier here. Measured on
this data, VIF pruning and PCA both cost ~7 points of accuracy, and the cause
was `StandardScaler` rather than the reduction itself: the encoder emits
L2-normalised vectors, and rescaling each dimension to unit variance destroys
that geometry. PCA *without* standardisation is accuracy-neutral (0.763 vs
0.762 baseline) at 286 of 384 dimensions.

Treat this report as a description of the feature space, not as a
recommendation to prune it.
"""
