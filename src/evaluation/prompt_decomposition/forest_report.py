"""What the forest learned, and how much of it is real.

Three views of the same fit, because the cheapest one is the least reliable:

* **Information gain** (impurity importance) -- what was asked for, and what
  the forest reports for free. Biased towards wide continuous families: a
  384-column embedding is offered 384 candidate splits where the domain block
  is offered 8, so it accumulates gain by width as well as by usefulness.
* **Permutation importance** -- shuffle one family's columns in the *held-out*
  rows and measure what the score loses. Costs a refit-free re-scoring per
  family and answers the question actually being asked: how much does this
  family contribute to performance on data the forest has not seen.
* **Ablation** -- fit on each family alone, and on everything-but-each-family.
  The most expensive and the most honest: it is the only one that cannot be
  fooled by correlated families sharing credit.

Read them together. When impurity says "embedding dominates" and ablation says
"dropping the embedding costs nothing", the ablation is right and the impurity
number was counting columns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from training.prompt_decomposition.heads.forest import ForestFeatures, fit_forest

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ForestReport:
    """One experiment: the fit, the three importance views, and the baseline."""

    target: str
    n_train: int
    n_test: int
    base_rate: float
    auc: float
    accuracy: float
    #: family -> summed impurity importance (information gain)
    information_gain: dict[str, float] = field(default_factory=dict)
    #: family -> AUC lost when that family is shuffled on held-out rows
    permutation_drop: dict[str, float] = field(default_factory=dict)
    #: family -> AUC using that family alone
    alone: dict[str, float] = field(default_factory=dict)
    #: family -> AUC using everything except that family
    without: dict[str, float] = field(default_factory=dict)
    #: family -> information gain divided by its column count. The cheap
    #: correction for the width bias: it asks what the *average column* of a
    #: family is worth, which is the comparison the raw gain pretends to be.
    gain_per_column: dict[str, float] = field(default_factory=dict)
    #: the widest individual columns, for the long tail view
    top_features: list[tuple[str, float]] = field(default_factory=list)
    columns_per_family: dict[str, int] = field(default_factory=dict)

    @property
    def lift_over_chance(self) -> float:
        return self.auc - 0.5

    def summary(self) -> str:
        lines = [
            f"target `{self.target}`: {self.n_train} train / {self.n_test} test, "
            f"base rate {self.base_rate:.3f}",
            f"  AUC {self.auc:.4f}  accuracy {self.accuracy:.4f}  "
            f"(lift over chance {self.lift_over_chance:+.4f})",
            f"  {'family':12} {'cols':>5} {'info gain':>10} {'per col':>9} "
            f"{'perm drop':>10} {'alone':>8} {'without':>8}",
        ]
        for family in self.information_gain:
            lines.append(
                f"  {family:12} {self.columns_per_family.get(family, 0):>5} "
                f"{self.information_gain[family]:>10.3f} "
                f"{self.gain_per_column.get(family, float('nan')):>9.5f} "
                f"{self.permutation_drop.get(family, float('nan')):>10.4f} "
                f"{self.alone.get(family, float('nan')):>8.4f} "
                f"{self.without.get(family, float('nan')):>8.4f}")
        lines.append("  info gain is impurity-based and rewards wide families; "
                     "read `alone` and `without` when they disagree")
        return "\n".join(lines)


def _auc(forest, features: ForestFeatures, y: np.ndarray) -> float:
    return float(roc_auc_score(y, forest.predict_proba(features.X)[:, 1]))


def build_report(train: ForestFeatures, y_train: np.ndarray,
                 test: ForestFeatures, y_test: np.ndarray, *,
                 target: str = "target", seed: int = 20260824,
                 permutation_repeats: int = 5, top_n: int = 15) -> ForestReport:
    """Fit once, then run all three importance views around it."""
    forest = fit_forest(train, y_train, seed=seed)
    proba = forest.predict_proba(test.X)[:, 1]
    report = ForestReport(
        target=target, n_train=len(y_train), n_test=len(y_test),
        base_rate=float(np.mean(y_test)),
        auc=float(roc_auc_score(y_test, proba)),
        accuracy=float(accuracy_score(y_test, (proba >= 0.5).astype(int))),
        columns_per_family={f: int(len(train.columns_for(f))) for f in train.present_families},
    )

    gains = forest.feature_importances_
    for family in train.present_families:
        cols = train.columns_for(family)
        report.information_gain[family] = float(gains[cols].sum())
        report.gain_per_column[family] = float(gains[cols].mean())

    # Permutation over whole families, not single columns: shuffling one of 384
    # correlated embedding dimensions moves nothing and would report the
    # embedding as worthless.
    for family in train.present_families:
        rng = np.random.default_rng(seed)
        drops = []
        for _ in range(permutation_repeats):
            shuffled = test.X.copy()
            cols = test.columns_for(family)
            shuffled[:, cols] = shuffled[rng.permutation(len(shuffled))][:, cols]
            drops.append(report.auc - roc_auc_score(y_test, forest.predict_proba(shuffled)[:, 1]))
        report.permutation_drop[family] = float(np.mean(drops))

    for family in train.present_families:
        report.alone[family] = _auc(
            fit_forest(train.only(family), y_train, seed=seed), test.only(family), y_test)
        if len(train.present_families) > 1:
            report.without[family] = _auc(
                fit_forest(train.without(family), y_train, seed=seed),
                test.without(family), y_test)

    order = np.argsort(-gains)[:top_n]
    report.top_features = [(train.names[i], float(gains[i])) for i in order]
    log.info("\n%s", report.summary())
    return report


def plot_report(report: ForestReport, path: str | Path, *, title: str | None = None) -> Path:
    """Four panels: the three importance views, and the long tail of columns.

    Reusable on any :class:`ForestReport`, whatever the target or families --
    nothing here is specific to one experiment. Written to `path` and the path
    returned, so a caller can drop it straight into a report.
    """
    import matplotlib
    matplotlib.use("Agg")                      # no display on a training box
    import matplotlib.pyplot as plt

    families = list(report.information_gain)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title or f"Random forest on `{report.target}` "
                          f"(AUC {report.auc:.3f}, base rate {report.base_rate:.3f})",
                 fontsize=13)

    ax = axes[0][0]
    gains = [report.information_gain[f] for f in families]
    ax.barh(families, gains, color="#4C72B0")
    for i, f in enumerate(families):
        ax.text(gains[i], i, f"  {report.columns_per_family.get(f, 0)} cols",
                va="center", fontsize=8, color="#555")
    ax.set_title("Information gain (impurity)\nrewards wide families — see panel 3", fontsize=10)
    ax.set_xlabel("summed gain")

    ax2 = ax.twiny()
    per_col = [report.gain_per_column.get(f, 0.0) for f in families]
    ax2.plot(per_col, range(len(families)), "o", color="#C44E52", markersize=7)
    ax2.set_xlabel("gain per column (red)", color="#C44E52", fontsize=8)
    ax2.tick_params(axis="x", labelcolor="#C44E52", labelsize=7)
    ax2.set_xlim(0, max(per_col) * 1.6 if max(per_col) else 1)

    ax = axes[0][1]
    drops = [report.permutation_drop.get(f, 0.0) for f in families]
    ax.barh(families, drops, color="#DD8452")
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_title("Permutation importance (held-out)\nAUC lost when the family is shuffled",
                 fontsize=10)
    ax.set_xlabel("AUC drop")

    ax = axes[1][0]
    y = np.arange(len(families))
    ax.barh(y - 0.2, [report.alone.get(f, np.nan) for f in families], height=0.4,
            label="family alone", color="#55A868")
    ax.barh(y + 0.2, [report.without.get(f, np.nan) for f in families], height=0.4,
            label="everything without it", color="#C44E52")
    ax.axvline(0.5, color="#333", linestyle="--", linewidth=0.8, label="chance")
    ax.axvline(report.auc, color="#4C72B0", linestyle=":", linewidth=1.2, label="all families")
    ax.set_yticks(y, families)
    ax.set_xlabel("AUC")
    ax.set_title("Ablation — the view that cannot be fooled\nby a family being wide", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1][1]
    names = [n for n, _ in report.top_features][::-1]
    values = [v for _, v in report.top_features][::-1]
    ax.barh(names, values, color="#8172B3")
    ax.set_title(f"Top {len(names)} individual columns by gain", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def run_forest_experiment(*, domain_run: str, task_run: str, encoder_model: str,
                          max_rows: int = 6000, seed: int = 20260824,
                          out_dir: str | Path = "artifacts/forest") -> ForestReport:
    """The experiment as asked: domain + task type + encoded query -> the routing decision.

    The target is `strong_needed`: did the stronger model actually win the
    battle this prompt came from. That is the decision the router makes, and
    the only label in this corpus that is about a decision rather than a
    description.

    Capped at ``max_rows`` because the domain head is a six-member ensemble and
    the cap is what keeps this a minutes-long experiment rather than an
    hours-long one. The cap is a random subsample of the held-out corpus, not a
    prefix.
    """
    import pandas as pd

    from decompose.classifiers.prompt_decomposition import DomainHead, TaskHead
    from decompose.classifiers.prompt_decomposition.encoder import EmbeddingEncoder
    from evaluation.prompt_decomposition.routing_value import load_routing_labels
    from training.prompt_decomposition.data.arena_corpus import load_arena_splits
    from training.prompt_decomposition.heads.forest import build_features

    labels = load_routing_labels()
    frames = []
    for split, frame in load_arena_splits().items():
        joined = frame.merge(labels, on="arena_id", how="inner")
        joined["split"] = split
        frames.append(joined)
    rows = pd.concat(frames, ignore_index=True)
    rng = np.random.default_rng(seed)
    if max_rows < len(rows):
        rows = rows.iloc[sorted(rng.choice(len(rows), max_rows, replace=False))].reset_index(drop=True)
    y = (rows["routing_label"] == "strong_needed").to_numpy().astype(int)
    prompts = rows["prompt"].tolist()
    log.info("forest corpus: %d prompts, base rate %.3f", len(rows), y.mean())

    domain, task = DomainHead(domain_run, merge_domains=True), TaskHead(task_run)
    domain_preds = domain.predict_batch(prompts)
    task_preds = task.predict_batch(prompts)
    domain_labels = sorted(domain_preds[0].distribution)
    task_labels = sorted(task_preds[0].distribution)
    encoder = EmbeddingEncoder(encoder_model, max_length=256, batch_size=128)

    features = build_features(
        domain_proba=np.array([[p.distribution[k] for k in domain_labels] for p in domain_preds]),
        domain_labels=domain_labels,
        task_proba=np.array([[p.distribution[k] for k in task_labels] for p in task_preds]),
        task_labels=task_labels,
        embeddings=encoder.encode_cached(prompts, tag=f"forest/{encoder_model.split('/')[-1]}"),
    )
    is_test = (rows["split"] == "test").to_numpy()
    train = ForestFeatures(features.X[~is_test], features.names, features.families)
    test = ForestFeatures(features.X[is_test], features.names, features.families)

    report = build_report(train, y[~is_test], test, y[is_test],
                          target="strong_needed", seed=seed)
    out_dir = Path(out_dir)
    plot_report(report, out_dir / "forest_importance.png")
    (out_dir / "forest_report.txt").write_text(report.summary() + "\n", encoding="utf-8")
    return report
