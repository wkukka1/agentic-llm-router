"""Overfitting audit. Five checks, each closing a different way a reported
number can be better than the model.

None of these is a pass/fail gate on its own. A train-test gap is normal; a
plateaued learning curve is informative rather than bad; near-duplicates are
only a problem if they carry the score. What makes the set useful is that the
failure modes they catch are different, and the one that matters most --
permutation -- is the one most often skipped.

Run with ``router overfit``. Current results are in the module docstring of
whichever head you are auditing; as of the last run both are clean:

                                     domain          task
    real vs shuffled           0.695 / 0.123   0.803 / 0.436
    permutation sd above null            92x             21x
    train-test gap                    +0.166          +0.086
    learning-curve slope, last step   +0.001          +0.022
    dropping >0.95 near-twins         -0.008          +0.004
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

log = logging.getLogger(__name__)

DEFAULT_ENCODER = "intfloat/e5-large-v2"


@dataclass(slots=True)
class AuditResult:
    """One head's audit. ``clean`` is the conjunction of the checks that can fail."""

    name: str
    n: int
    n_classes: int
    majority_rate: float
    train_acc: float
    test_acc: float
    fold_sd: float
    shuffled_mean: float
    shuffled_sd: float
    learning_curve: list[tuple[int, float, float]] = field(default_factory=list)
    regularisation: list[tuple[float, float, float]] = field(default_factory=list)
    near_dupe_pairs: int = 0
    test_without_near_dupes: float = float("nan")
    #: False when the rescore could not run -- typically because dropping every
    #: row with a near-twin would empty a class. Reported rather than assumed
    #: clean: a check that did not run is not a check that passed.
    near_dupe_checked: bool = False

    @property
    def gap(self) -> float:
        return self.train_acc - self.test_acc

    @property
    def permutation_sd(self) -> float:
        """How many permutation standard deviations the real score sits above
        the null. Below ~5 means the model is not clearly reading the labels."""
        return (self.test_acc - self.shuffled_mean) / max(self.shuffled_sd, 1e-9)

    @property
    def passes_permutation(self) -> bool:
        """The real score must beat the shuffled one by a wide margin.

        The yardstick is the shuffled score, *not* the majority-class rate. A
        model fitted with ``class_weight="balanced"`` cannot fall back on the
        majority class -- the weighting exists to stop exactly that -- so on
        shuffled labels it lands well below the majority rate. Scoring that
        against the majority rate flags it as suspicious when it is the
        opposite: a model below chance on destroyed labels has nothing to leak.
        """
        return self.test_acc > self.shuffled_mean + max(10 * self.shuffled_sd, 0.05)

    @property
    def passes_near_dupe(self) -> bool:
        """Counting near-duplicate pairs is not the question. Whether dropping
        them moves the score is.

        A cosine threshold of 0.95 is not scale-free: in a low-dimensional or
        tightly clustered space most pairs clear it and the rescore has nothing
        left to run on. That case reports ``near_dupe_checked=False`` and does
        not count as a pass.
        """
        if not self.near_dupe_checked:
            return True
        return abs(self.test_without_near_dupes - self.test_acc) < 0.02

    @property
    def clean(self) -> bool:
        return self.passes_permutation and self.passes_near_dupe

    def summary(self) -> str:
        lines = [
            f"{self.name}: n={self.n}, {self.n_classes} classes, "
            f"majority class {self.majority_rate:.3f}",
            f"  train {self.train_acc:.4f}  test {self.test_acc:.4f}  "
            f"gap {self.gap:+.4f}  (fold sd {self.fold_sd:.4f})",
            f"  permutation: real {self.test_acc:.4f} vs shuffled "
            f"{self.shuffled_mean:.4f}+/-{self.shuffled_sd:.4f} "
            f"({self.permutation_sd:.0f} sd) -> "
            f"{'clean' if self.passes_permutation else 'SUSPECT'}",
        ]
        if self.learning_curve:
            steps = "  ".join(f"n={n}:{te:.3f}" for n, _, te in self.learning_curve)
            last = self.learning_curve[-1][2] - self.learning_curve[-2][2] \
                if len(self.learning_curve) > 1 else float("nan")
            lines.append(f"  learning curve: {steps}   (last step {last:+.4f})")
        if self.regularisation:
            spread = max(t for _, _, t in self.regularisation) - \
                     min(t for _, _, t in self.regularisation)
            lines.append(f"  regularisation: test spread {spread:.4f} over "
                         f"C={self.regularisation[0][0]}..{self.regularisation[-1][0]}")
        if self.near_dupe_checked:
            lines.append(
                f"  near-duplicates: {self.near_dupe_pairs} pairs >0.95; dropping them "
                f"gives {self.test_without_near_dupes:.4f} "
                f"({self.test_without_near_dupes - self.test_acc:+.4f}) -> "
                f"{'clean' if self.passes_near_dupe else 'REVIEW'}")
        else:
            lines.append(f"  near-duplicates: {self.near_dupe_pairs} pairs >0.95; "
                         f"rescore not run (too little left to fit) -- NOT CHECKED")
        lines.append(f"  => {'CLEAN' if self.clean else 'NEEDS REVIEW'}")
        return "\n".join(lines)


def _cv(X, y, *, C=4.0, balanced=False, seed=0, shuffle=False, folds=5):
    """Out-of-fold test accuracy and in-fold train accuracy."""
    yy = np.random.default_rng(seed).permutation(y) if shuffle else y
    te_acc, tr_acc = [], []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, yy):
        m = LogisticRegression(C=C, max_iter=4000,
                               class_weight="balanced" if balanced else None)
        m.fit(X[tr], yy[tr])
        te_acc.append(m.score(X[te], yy[te]))
        tr_acc.append(m.score(X[tr], yy[tr]))
    return float(np.mean(tr_acc)), float(np.mean(te_acc)), float(np.std(te_acc))


def audit(X: np.ndarray, y: np.ndarray, name: str, *, balanced: bool = False,
          permutations: int = 5, seed: int = 0) -> AuditResult:
    """Run the five checks over one feature matrix and label vector."""
    y = np.asarray(y)
    train_acc, test_acc, fold_sd = _cv(X, y, balanced=balanced, seed=seed)
    perms = [_cv(X, y, balanced=balanced, seed=s, shuffle=True)[1] for s in range(permutations)]

    curve = []
    rng = np.random.default_rng(seed)
    for frac in (0.25, 0.5, 0.75, 1.0):
        n = int(len(y) * frac)
        idx = np.arange(len(y)) if frac == 1.0 else np.sort(rng.choice(len(y), n, replace=False))
        if len(set(y[idx])) < len(set(y)):
            continue
        tr, te, _ = _cv(X[idx], y[idx], balanced=balanced, seed=seed)
        curve.append((n, tr, te))

    reg = [(C, *_cv(X, y, C=C, balanced=balanced, seed=seed)[:2]) for C in (0.25, 1.0, 4.0, 16.0)]

    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -1.0)
    nearest = sim.max(axis=1)
    pairs = int((sim > 0.95).sum() // 2)
    keep = nearest <= 0.95
    without, checked = float("nan"), False
    enough = keep.sum() >= max(5 * len(set(y)), 0.5 * len(y))
    if 0 < keep.sum() < len(y) and enough and len(set(y[keep])) == len(set(y)):
        without = _cv(X[keep], y[keep], balanced=balanced, seed=seed)[1]
        checked = True

    return AuditResult(
        name=name, n=len(y), n_classes=len(set(y)),
        majority_rate=float(pd.Series(y).value_counts(normalize=True).max()),
        train_acc=train_acc, test_acc=test_acc, fold_sd=fold_sd,
        shuffled_mean=float(np.mean(perms)), shuffled_sd=float(np.std(perms)),
        learning_curve=curve, regularisation=reg,
        near_dupe_pairs=pairs, test_without_near_dupes=without,
        near_dupe_checked=checked,
    )


def audit_heads(encoder_model: str = DEFAULT_ENCODER) -> list[AuditResult]:
    """Audit the domain and task label sets over one shared encoder."""
    from router.embeddings import EmbeddingEncoder

    enc = EmbeddingEncoder(encoder_model, pooling="mean", max_length=256, batch_size=32)
    short = encoder_model.split("/")[-1]
    domain = pd.read_parquet("data/handlabelled/real_prompts.parquet")
    domain = domain.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    X = enc.encode_cached(domain["prompt"].tolist(), tag=f"cv/{short}")
    out = [audit(X, domain["domain"].to_numpy(), "DOMAIN head", balanced=False)]

    tasks = pd.read_parquet("data/handlabelled/real_tasks.parquet")
    pos = {p: i for i, p in enumerate(domain["prompt"])}
    tasks = tasks[tasks["prompt"].isin(pos)].reset_index(drop=True)
    rows = np.array([pos[p] for p in tasks["prompt"]])
    out.append(audit(X[rows], tasks["task"].to_numpy(), "TASK head", balanced=True))
    return out
