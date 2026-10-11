"""Per-model correctness by category: does knowing the domain change the best model?

Every earlier test of the prompt signals asked a binary question -- was the
stronger model needed -- against a label that is one human vote per battle, and
every one of them landed near chance. That label conflates two things: whether
the prompt was hard, and whether *these two* models differ on it.

This module asks the question routing actually turns on, against a target that
is not a vote. A model pool scores every one of its models on every query, so
per-model correctness is observed rather than inferred. Pools are plugged in by
name (`training.prompt_decomposition.data.pools`), because a routing conclusion
drawn on a 2023 pool may be a fact about that pool rather than about routing.
The question is:

    If each category is sent to the model that is best *for that category*,
    how much of the gap between "one model for everything" and "the best
    model for every single prompt" does that close, and at what cost?

Four routers answer it, all fitted on a training split and scored on held-out
prompts:

* ``single_best``   -- one model for everything. The floor any router must beat.
* ``by_<grouping>`` -- per category, the model with the best training accuracy
  (or best accuracy net of cost, with ``cost_weight``).
* ``oracle``        -- the best model for each prompt, chosen with the answer.
  Unreachable; it is the ceiling that defines the gap.

The headline is **gap closed**: ``(router - single_best) / (oracle - single_best)``.
A category signal that closes 2% of the gap is a label, not a routing signal,
however accurately it is predicted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from training.prompt_decomposition.data.pools import CorrectnessTable

log = logging.getLogger(__name__)

@dataclass(slots=True)
class RouterResult:
    name: str
    accuracy: float
    cost: float                    # mean dollars per prompt
    gap_closed: float = float("nan")
    #: category -> chosen model, for routers that choose per category
    choices: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PerModelReport:
    n_train: int
    n_test: int
    models: list[str]
    #: False when the pool prices nothing; cost figures are then meaningless
    #: and are printed as `--` rather than as zeros.
    priced: bool = True
    #: model -> held-out accuracy overall
    model_accuracy: dict[str, float] = field(default_factory=dict)
    model_cost: dict[str, float] = field(default_factory=dict)
    #: grouping -> (category x model) held-out accuracy table
    by_group: dict[str, pd.DataFrame] = field(default_factory=dict)
    routers: list[RouterResult] = field(default_factory=list)

    def router(self, name: str) -> RouterResult:
        return next(r for r in self.routers if r.name == name)

    def summary(self) -> str:
        lines = [f"{self.n_train} train / {self.n_test} test prompts, {len(self.models)} models",
                 f"  {'router':28} {'accuracy':>9} {'$/prompt':>10} {'gap closed':>11}"]
        for r in self.routers:
            gap = "" if np.isnan(r.gap_closed) else f"{r.gap_closed:>10.1%}"
            cost = f"{r.cost:>10.6f}" if self.priced else f"{'--':>10}"
            lines.append(f"  {r.name:28} {r.accuracy:>9.4f} {cost} {gap:>11}")
        return "\n".join(lines)


def spendable(table: CorrectnessTable) -> tuple[np.ndarray, bool]:
    """The cost matrix to optimise against, and whether it is real.

    An unpriced pool gets zeros rather than NaN: NaN poisons every argmax, and
    silently substituting 1.0 -- as a caller did once -- prints a cost column
    of ones that reads like dollars. Zeros make cost-weighted choice collapse
    to accuracy, which is the honest behaviour when nothing is priced, and the
    flag travels so reports can print `--` instead of a number.
    """
    return (table.cost, True) if table.has_cost else (np.zeros_like(table.cost), False)


def _pick(correct: np.ndarray, cost: np.ndarray, cost_weight: float) -> int:
    """Index of the model with the best mean accuracy, net of weighted cost."""
    return int(np.argmax(correct.mean(axis=0) - cost_weight * cost.mean(axis=0)))


def category_router(train: CorrectnessTable, test: CorrectnessTable, grouping: str, *,
                    cost_weight: float = 0.0, min_train: int = 30,
                    name: str | None = None) -> RouterResult:
    """Send each category to its best model on the training split.

    Categories with fewer than `min_train` training prompts fall back to the
    single best model: a per-category choice from a dozen prompts is a coin
    toss that would be scored as if it were a decision.
    """
    fallback = _pick(train.correct, train.cost, cost_weight)
    choice: dict[str, int] = {}
    for category, idx in train.prompts.groupby(grouping).indices.items():
        choice[str(category)] = (_pick(train.correct[idx], train.cost[idx], cost_weight)
                                 if len(idx) >= min_train else fallback)
    picks = np.array([choice.get(str(c), fallback) for c in test.prompts[grouping]])
    rows = np.arange(len(picks))
    return RouterResult(
        name=name or f"by_{grouping}",
        accuracy=float(test.correct[rows, picks].mean()),
        cost=float(test.cost[rows, picks].mean()),
        choices={c: train.models[i] for c, i in choice.items()},
    )


def build_report(table: CorrectnessTable, groupings: list[str], *,
                 cost_weight: float = 0.0) -> PerModelReport:
    """Fit every router on train, score on test, and fill in gap closed."""
    costs, priced = spendable(table)
    table = CorrectnessTable(table.prompts, table.correct, costs, table.models, table.name)
    train = table.subset((table.prompts["split"] == "train").to_numpy())
    test = table.subset((table.prompts["split"] == "test").to_numpy())
    report = PerModelReport(n_train=len(train.prompts), n_test=len(test.prompts),
                            models=list(table.models), priced=priced)
    report.model_accuracy = dict(zip(table.models, test.correct.mean(axis=0).tolist(),
                                     strict=True))
    report.model_cost = dict(zip(table.models, test.cost.mean(axis=0).tolist(), strict=True))

    best = _pick(train.correct, train.cost, cost_weight)
    single = RouterResult(f"single_best ({table.models[best]})",
                          float(test.correct[:, best].mean()), float(test.cost[:, best].mean()))
    # The oracle takes the best score per prompt; among ties, the cheapest.
    top = test.correct.max(axis=1, keepdims=True)
    cheapest_top = np.where(test.correct == top, test.cost, np.inf).argmin(axis=1)
    rows = np.arange(len(cheapest_top))
    oracle = RouterResult("oracle (per prompt)", float(top.mean()),
                          float(test.cost[rows, cheapest_top].mean()))

    report.routers.append(single)
    for grouping in groupings:
        report.routers.append(category_router(train, test, grouping, cost_weight=cost_weight))
        report.by_group[grouping] = pd.DataFrame(
            {m: pd.Series(test.correct[:, j]).groupby(test.prompts[grouping].to_numpy()).mean()
             for j, m in enumerate(table.models)})
    report.routers.append(oracle)

    span = oracle.accuracy - single.accuracy
    for r in report.routers:
        r.gap_closed = (r.accuracy - single.accuracy) / span if span > 0 else float("nan")
    log.info("\n%s", report.summary())
    return report


def short_model(name: str) -> str:
    return name.split("/")[-1].replace("-chat", "").replace("-instruct", "")


def plot_report(report: PerModelReport, grouping: str, path: str | Path, *,
                title: str | None = None, max_categories: int = 14) -> Path:
    """Two panels: per-model accuracy by category, and the routers side by side.

    The heatmap is the per-model correctness itself -- read across a row to see
    whether a category has a different best model than the rest. The bar panel
    is what that is worth once it has to be chosen from training data.
    Reusable on any :class:`PerModelReport` and any grouping it holds.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = report.by_group[grouping]
    if len(table) > max_categories:
        table = table.loc[table.mean(axis=1).sort_values().index[-max_categories:]]
    order = sorted(report.models, key=lambda m: -report.model_accuracy[m])
    table = table[order]

    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 0.42 * len(table) + 3.2),
                                      gridspec_kw={"width_ratios": [3, 2]})
    im = left.imshow(table.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    left.set_xticks(range(len(order)), [short_model(m) for m in order], rotation=45,
                    ha="right", fontsize=8)
    left.set_yticks(range(len(table)), table.index, fontsize=8)
    best_col = table.to_numpy().argmax(axis=1)
    for i, j in enumerate(best_col):
        left.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                     edgecolor="white", linewidth=2))
    left.set_title(f"Held-out accuracy by `{grouping}` — boxed: best model per row", fontsize=10)
    fig.colorbar(im, ax=left, fraction=0.03)

    names = [r.name for r in report.routers]
    acc = [r.accuracy for r in report.routers]
    colours = ["#999999"] + ["#4C72B0"] * (len(names) - 2) + ["#55A868"]
    right.barh(names, acc, color=colours)
    lo = min(acc) - 0.03
    right.set_xlim(lo, max(acc) + 0.03)
    for i, r in enumerate(report.routers):
        label = f" {r.accuracy:.3f}"
        if not np.isnan(r.gap_closed) and 0 < i < len(names) - 1:
            label += f"  ({r.gap_closed:.0%} of gap)"
        right.text(r.accuracy, i, label, va="center", fontsize=8)
    right.invert_yaxis()
    right.set_title("Router accuracy on held-out prompts\n"
                    "grey = one model for all, green = oracle", fontsize=10)

    fig.suptitle(title or f"Per-model correctness — {report.n_test} held-out prompts, "
                          f"{len(report.models)} models", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def gap_decomposition(table: CorrectnessTable, grouping: str) -> dict[str, float]:
    """Split the oracle's advantage into the part a category can see and the rest.

    ``between`` is what choosing the right model *per category* buys over one
    model for everything; ``within`` is what choosing per *prompt* buys on top.
    Measured on the full table rather than a split -- this is a property of the
    data, not of a fitted router -- so it is the ceiling for any router keyed
    on `grouping`, however well the category is predicted.
    """
    overall = table.correct.mean(axis=0).max()
    oracle = table.correct.max(axis=1).mean()
    best_per_category = sum(
        table.correct[idx].mean(axis=0).max() * len(idx)
        for idx in table.prompts.groupby(grouping).indices.values()) / len(table.prompts)
    gap = oracle - overall
    return {"single_best": float(overall), "category_best": float(best_per_category),
            "oracle": float(oracle),
            "between": float((best_per_category - overall) / gap) if gap > 0 else float("nan"),
            "within": float((oracle - best_per_category) / gap) if gap > 0 else float("nan")}


def cost_frontier(table: CorrectnessTable, groupings: list[str], *,
                  weights: np.ndarray | None = None) -> pd.DataFrame:
    """Accuracy against cost as the price of a dollar rises, per router.

    Accuracy alone flatters one strong model; a cost-aware router is judged on
    its frontier. At each `cost_weight` every router picks, on train, the model
    maximising accuracy minus weight x cost -- per category or globally -- and
    is scored on test. A category router is worth something exactly where its
    curve sits above the single-model curve.
    """
    weights = np.concatenate([[0.0], np.geomspace(1, 1e4, 25)]) if weights is None else weights
    costs, priced = spendable(table)
    if not priced:
        raise ValueError(f"pool {table.name!r} has no prices; a cost frontier over "
                         f"unpriced models would be a curve through made-up numbers")
    table = CorrectnessTable(table.prompts, table.correct, costs, table.models, table.name)
    train = table.subset((table.prompts["split"] == "train").to_numpy())
    test = table.subset((table.prompts["split"] == "test").to_numpy())
    rows = []
    for w in weights:
        best = _pick(train.correct, train.cost, w)
        rows.append({"cost_weight": w, "router": "single_best",
                     "accuracy": test.correct[:, best].mean(), "cost": test.cost[:, best].mean()})
        for grouping in groupings:
            r = category_router(train, test, grouping, cost_weight=w)
            rows.append({"cost_weight": w, "router": f"by_{grouping}",
                         "accuracy": r.accuracy, "cost": r.cost})
    return pd.DataFrame(rows)


def plot_frontier(frontier: pd.DataFrame, path: str | Path, *,
                  models: dict[str, tuple[float, float]] | None = None,
                  title: str = "Accuracy against cost") -> Path:
    """Each router's accuracy/cost curve; individual models as reference points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = ["#999999", "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for colour, (name, part) in zip(palette, frontier.groupby("router", sort=False),
                                    strict=False):
        part = part.sort_values("cost")
        ax.plot(part["cost"], part["accuracy"], "o-", color=colour, label=name,
                markersize=4, linewidth=2 if name != "single_best" else 1.5)
    for name, (cost, acc) in (models or {}).items():
        ax.scatter(cost, acc, color="black", s=12, zorder=5)
        ax.annotate(short_model(name), (cost, acc), fontsize=7, xytext=(4, -3),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("mean $ per prompt (log)")
    ax.set_ylabel("held-out accuracy")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --- diagnostics from "The Routing Plateau" (arXiv 2606.07587) ---------------
#
# The paper evaluates 21 routers on five benchmarks and finds they all converge
# to a narrow band far below the oracle -- a *routing plateau* caused by a
# correctness-prediction bottleneck: routers learn global model-capability
# averages rather than instance-level correctness. Its sharpest diagnostic is
# the hard/easy split below, which localises the whole oracle gap.


def stratify_hard_easy(table: CorrectnessTable, *, threshold: float = 0.5) -> pd.Series:
    """Label each prompt `hard`, `easy` or `unsolvable`, per the paper's Sec 5.1.

    A solvable prompt is **hard** when either

    * exactly one model in the pool gets it right -- the router must find that
      one model, or
    * few do (at most ``max(8, ceil(K/2))``) *and the globally best model is
      wrong on it* -- so the router has to look past the strongest model.

    Every other solvable prompt is **easy**: the best-on-average model already
    handles it, and any router that defaults to that model collects it. A
    prompt no model solves is `unsolvable` and belongs to neither -- it cannot
    be routed well or badly.

    ``threshold`` binarises graded scores; RouterBench and SPROUT both carry
    fractional judge scores for some items.
    """
    correct = table.correct >= threshold
    n_correct = correct.sum(axis=1)
    k = len(table.models)
    top_model = int(table.correct.mean(axis=0).argmax())
    few = n_correct <= max(8, -(-k // 2))
    hard = (n_correct == 1) | (few & ~correct[:, top_model] & (n_correct > 1))
    labels = np.where(n_correct == 0, "unsolvable", np.where(hard, "hard", "easy"))
    return pd.Series(labels, index=table.prompts.index, name="difficulty")


@dataclass(slots=True)
class PlateauReport:
    """Where the oracle gap lives: on the hard prompts, or spread evenly."""

    pool: str
    n: int
    f_hard: float
    f_unsolvable: float
    gap_hard: float
    gap_easy: float
    #: gap_hard / gap_easy -- the paper reports 10.7x to 38.5x
    ratio: float
    #: share of the *total* oracle gap contributed by hard prompts (69.9-90.8%)
    explained_by_hard: float
    best_model: str = ""

    def summary(self) -> str:
        return (f"{self.pool}: {self.n} prompts, {self.f_hard:.1%} hard, "
                f"{self.f_unsolvable:.1%} unsolvable\n"
                f"  oracle gap on hard {self.gap_hard:.3f} vs easy {self.gap_easy:.3f} "
                f"({self.ratio:.1f}x), hard prompts hold {self.explained_by_hard:.1%} "
                f"of the total gap\n"
                f"  best single model: {self.best_model}")


def plateau_report(table: CorrectnessTable, *, threshold: float = 0.5) -> PlateauReport:
    """The oracle gap, split over the hard and easy prompts.

    ``gap`` here is oracle minus the single best model, measured within each
    subset -- the paper measures against the all-router mean, which needs a
    fleet of routers; the best single model is the same reference our routers
    are scored against and makes the two analyses comparable.
    """
    difficulty = stratify_hard_easy(table, threshold=threshold).to_numpy()
    best = int(table.correct.mean(axis=0).argmax())
    oracle = table.correct.max(axis=1)
    baseline = table.correct[:, best]

    def gap(mask: np.ndarray) -> float:
        return float((oracle[mask] - baseline[mask]).mean()) if mask.any() else 0.0

    hard, easy = difficulty == "hard", difficulty == "easy"
    total = float((oracle - baseline).sum())
    return PlateauReport(
        pool=table.name, n=len(table.prompts),
        f_hard=float(hard.mean()), f_unsolvable=float((difficulty == "unsolvable").mean()),
        gap_hard=gap(hard), gap_easy=gap(easy),
        ratio=gap(hard) / gap(easy) if gap(easy) > 0 else float("inf"),
        explained_by_hard=float((oracle[hard] - baseline[hard]).sum() / total) if total else 0.0,
        best_model=table.models[best],
    )


def selection_concentration(table: CorrectnessTable, picks: np.ndarray, *,
                            threshold: float = 0.5) -> pd.DataFrame:
    """How a router spreads its choices, on hard prompts against all prompts.

    The paper's Fig. 4: routers pick the same two or three models whatever the
    difficulty, while the oracle's correct-model mass on hard prompts is spread
    right across the pool. A router whose hard-prompt distribution matches its
    overall one is not routing, it is defaulting.
    """
    difficulty = stratify_hard_easy(table, threshold=threshold).to_numpy()
    correct = table.correct >= threshold
    share = lambda mask: pd.Series(picks[mask]).value_counts(normalize=True)  # noqa: E731
    oracle_mass = pd.Series(
        (correct[difficulty == "hard"].sum(axis=0) /
         max(correct[difficulty == "hard"].sum(), 1)), index=range(len(table.models)))
    frame = pd.DataFrame({
        "router_all": share(np.ones(len(picks), bool)),
        "router_hard": share(difficulty == "hard"),
        "oracle_hard_mass": oracle_mass,
    }).fillna(0.0)
    frame.index = [short_model(table.models[i]) for i in frame.index]
    return frame.sort_values("oracle_hard_mass", ascending=False)


def knn_router(train: CorrectnessTable, test: CorrectnessTable, train_vectors: np.ndarray,
               test_vectors: np.ndarray, *, k: int = 16, cost_weight: float = 0.0,
               name: str = "knn") -> RouterResult:
    """Pick the model with the best score among the k nearest training prompts.

    The paper's headline baseline: plain similarity search over frozen
    embeddings ranks top-2 on all five benchmarks, matching routers with far
    more machinery. Anything we build should be measured against it, because if
    it cannot beat kNN the extra machinery is decoration.
    """
    train_unit = train_vectors / np.clip(np.linalg.norm(train_vectors, axis=1, keepdims=True),
                                         1e-12, None)
    test_unit = test_vectors / np.clip(np.linalg.norm(test_vectors, axis=1, keepdims=True),
                                       1e-12, None)
    picks = np.empty(len(test_unit), dtype=int)
    for start in range(0, len(test_unit), 512):          # blocked: the full matrix is n x m
        block = test_unit[start:start + 512] @ train_unit.T
        for i, row in enumerate(block):
            neighbours = np.argpartition(-row, min(k, len(row) - 1))[:k]
            utility = (train.correct[neighbours].mean(axis=0)
                       - cost_weight * train.cost[neighbours].mean(axis=0))
            picks[start + i] = int(np.nanargmax(utility))
    rows = np.arange(len(picks))
    return RouterResult(name=name, accuracy=float(test.correct[rows, picks].mean()),
                        cost=float(test.cost[rows, picks].mean()))
