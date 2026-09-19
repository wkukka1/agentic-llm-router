"""Per-model correctness by category: does knowing the domain change the best model?

Every earlier test of the prompt signals asked a binary question -- was the
stronger model needed -- against a label that is one human vote per battle, and
every one of them landed near chance. That label conflates two things: whether
the prompt was hard, and whether *these two* models differ on it.

This module asks the question routing actually turns on, against a target that
is not a vote. RouterBench scores every one of its models on every query, so
per-model correctness is observed rather than inferred. The question is:

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

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROUTERBENCH_META = ("sample_id", "prompt", "eval_name", "oracle_model_to_route_to")


@dataclass(slots=True)
class CorrectnessTable:
    """Prompts x models: correctness in [0, 1] and the cost of each call."""

    prompts: pd.DataFrame          # one row per prompt; at least `prompt`, `split`
    correct: np.ndarray            # (n_prompts, n_models)
    cost: np.ndarray               # (n_prompts, n_models), dollars per call
    models: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.correct.shape != self.cost.shape or self.correct.shape != (
                len(self.prompts), len(self.models)):
            raise ValueError(
                f"correct {self.correct.shape}, cost {self.cost.shape}, "
                f"{len(self.prompts)} prompts x {len(self.models)} models -- must agree")

    def subset(self, mask: np.ndarray) -> CorrectnessTable:
        return CorrectnessTable(self.prompts[mask].reset_index(drop=True),
                                self.correct[mask], self.cost[mask], list(self.models))


def benchmark_family(eval_name: str) -> str:
    """`mmlu-professional-law` -> `mmlu`; the ~30 tiny Chinese-language sets ->
    `chinese`; everything else keeps its name.

    Not a split on the first hyphen, which turns `grade-school-math` into
    `grade` and `arc-challenge` into `arc`.
    """
    name = eval_name.split(".")[0]
    for prefix in ("mmlu", "mtbench"):
        if name.startswith(prefix):
            return prefix
    if name.lower().startswith("chinese"):
        return "chinese"
    return name


def _split_of(key: str) -> str:
    """Hashed, so a rebuild never moves a prompt between train and test."""
    return "train" if int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.7 \
        else "test"


def load_routerbench(path: str | Path = "routerbench/routerbench_0shot.pkl") -> CorrectnessTable:
    """RouterBench's wide table as a :class:`CorrectnessTable`.

    Rows missing any model's score are dropped rather than imputed: an imputed
    correctness would be a guess about exactly the quantity being measured.
    """
    raw = pd.read_pickle(path)
    models = [c for c in raw.columns if "|" not in c and c not in ROUTERBENCH_META]
    cost_cols = [f"{m}|total_cost" for m in models]
    complete = raw[models + cost_cols].notna().all(axis=1)
    raw = raw[complete].reset_index(drop=True)
    prompts = pd.DataFrame({
        "sample_id": raw["sample_id"].astype(str),
        "prompt": raw["prompt"].astype(str),
        "eval_name": raw["eval_name"].astype(str),
        # The benchmark family: the *true* category, known without any
        # classifier, and so the bound on what category routing can do.
        "benchmark": raw["eval_name"].map(benchmark_family),
    })
    prompts["split"] = [_split_of(k) for k in prompts["sample_id"]]
    log.info("routerbench: %d prompts x %d models (%d incomplete rows dropped)",
             len(prompts), len(models), int((~complete).sum()))
    return CorrectnessTable(prompts, raw[models].to_numpy(float),
                            raw[cost_cols].to_numpy(float), models)


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
            lines.append(f"  {r.name:28} {r.accuracy:>9.4f} {r.cost:>10.6f} {gap:>11}")
        return "\n".join(lines)


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
    train = table.subset((table.prompts["split"] == "train").to_numpy())
    test = table.subset((table.prompts["split"] == "test").to_numpy())
    report = PerModelReport(n_train=len(train.prompts), n_test=len(test.prompts),
                            models=list(table.models))
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
