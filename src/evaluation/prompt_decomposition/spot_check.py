"""Are the classifier predictions the forest was fed actually right?

The forest scored AUC 0.566 on the routing decision, and domain and task added
almost nothing to it. That has two very different explanations, and this module
exists to tell them apart:

* **The features are fine and the target is hard.** Domain and task are
  predicted correctly; they just do not bear on which model wins.
* **The features are wrong on this data.** The heads were trained on 2,441
  hand-labelled prompts with a balanced label mix -- no domain above 15%. The
  forest's rows are a different distribution: English-only (the routing labels
  come from the English preference battles), and on measurement 2.5x heavier in
  `software_tech`. If the predictions are wrong or collapse into a couple of
  domains, the forest was fed noise and its verdict on "domain" says nothing
  about domain.

Three checks, cheapest first:

* :func:`distribution_report` -- where the predictions land. If two domains
  hold most of the traffic, a domain feature has little variance to route on
  whatever its accuracy.
* :func:`review_sheet` -- a random subset written out for a person to read and
  mark. There are no gold labels for arena prompts, so reading them is the
  only way to measure accuracy on this distribution.
* :func:`score_review` -- accuracy from a marked sheet, with Wilson intervals,
  because a spot check is small and a point estimate from sixty rows means
  little without its interval.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Columns a reviewer fills in. Blank on the written sheet; `score_review`
#: reads them back.
REVIEW_COLUMNS = ("domain_correct", "domain_in_shortlist", "task_correct",
                  "true_domain", "true_task", "note")


def predict_corpus(rows: pd.DataFrame, domain_head, task_head,
                   batch_size: int = 256) -> pd.DataFrame:
    """Domain and task predictions for every row, in a frame worth keeping.

    Batched, because the domain head is a six-member ensemble and one giant
    batch holds every member's activations at once. The result carries
    `arena_id` so it joins back to anything else keyed on the battle -- the
    per-model correctness analysis reads it.
    """
    out = []
    prompts = rows["prompt"].tolist()
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        for d, t in zip(domain_head.predict_batch(chunk), task_head.predict_batch(chunk),
                        strict=True):
            ranked = sorted(d.distribution.items(), key=lambda kv: -kv[1])
            out.append({
                "domain": d.domain, "domain_conf": d.confidence,
                "domain_2nd": ranked[1][0] if len(ranked) > 1 else None,
                "domain_2nd_conf": ranked[1][1] if len(ranked) > 1 else 0.0,
                "domain_shortlist": "|".join(d.shortlist),
                "task": t.task, "task_conf": t.confidence,
            })
        log.info("predicted %d / %d", min(start + batch_size, len(prompts)), len(prompts))
    preds = pd.DataFrame(out)
    keep = [c for c in ("arena_id", "prompt", "language", "split", "routing_label") if c in rows]
    return pd.concat([rows[keep].reset_index(drop=True), preds], axis=1)


@dataclass(slots=True)
class DistributionReport:
    """Where the predictions land, against where the training labels were."""

    n: int
    domain_share: dict[str, float] = field(default_factory=dict)
    task_share: dict[str, float] = field(default_factory=dict)
    training_domain_share: dict[str, float] = field(default_factory=dict)
    #: exp(entropy) of the domain shares: 8 for a perfectly even spread over 8,
    #: 2 if the traffic behaves like two equally sized domains.
    effective_domains: float = 0.0
    top2_share: float = 0.0
    mean_domain_conf: float = 0.0
    low_conf_share: float = 0.0
    #: domain share by language group: a head trained on English prompts may
    #: be sending the non-English half somewhere systematic.
    by_language: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"{self.n} prompts. Effective number of domains {self.effective_domains:.2f} "
                 f"(of {len(self.domain_share)}); top two hold {self.top2_share:.1%}.",
                 f"Mean domain confidence {self.mean_domain_conf:.3f}; "
                 f"{self.low_conf_share:.1%} below 0.5.",
                 f"  {'domain':22} {'arena':>7} {'training':>9}"]
        for name, share in sorted(self.domain_share.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:22} {share:>7.1%} "
                         f"{self.training_domain_share.get(name, float('nan')):>9.1%}")
        lines.append(f"  {'task':22} {'arena':>7}")
        for name, share in sorted(self.task_share.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:22} {share:>7.1%}")
        return "\n".join(lines)


def distribution_report(preds: pd.DataFrame, *,
                        training_labels: pd.Series | None = None) -> DistributionReport:
    """How concentrated the predicted domains are, and against what baseline.

    `training_labels` are the labels the head learned from, mapped into the
    same space as the predictions. Without them there is no way to say whether
    a concentration is the traffic or the head.
    """
    shares = preds["domain"].value_counts(normalize=True)
    p = shares.to_numpy()
    report = DistributionReport(
        n=len(preds),
        domain_share={k: float(v) for k, v in shares.items()},
        task_share={k: float(v) for k, v in preds["task"].value_counts(normalize=True).items()},
        effective_domains=float(np.exp(-(p * np.log(p)).sum())),
        top2_share=float(p[:2].sum()),
        mean_domain_conf=float(preds["domain_conf"].mean()),
        low_conf_share=float((preds["domain_conf"] < 0.5).mean()),
    )
    if training_labels is not None:
        report.training_domain_share = {
            k: float(v) for k, v in training_labels.value_counts(normalize=True).items()}
    if "language" in preds:
        group = np.where(preds["language"] == "en", "en", "non-en")
        for name, part in preds.groupby(group):
            report.by_language[str(name)] = {
                k: float(v) for k, v in part["domain"].value_counts(normalize=True).items()}
    return report


def review_sheet(preds: pd.DataFrame, path: str | Path, *, n: int = 60,
                 seed: int = 20260824) -> Path:
    """A random subset, written out for a person to read and mark.

    Random, not the least confident: a sheet of hard cases measures the hard
    cases, and the question is what accuracy looks like on the traffic the
    forest actually saw. Prompts are truncated for reading, not for scoring --
    the reviewer should mark on what the prompt asks, which is almost always in
    the first few hundred characters.
    """
    sample = preds.sample(n=min(n, len(preds)), random_state=seed).reset_index(drop=True)
    sheet = sample[["arena_id", "prompt", "domain", "domain_conf", "domain_shortlist",
                    "task", "task_conf"]].copy()
    if "language" in sample:
        sheet.insert(2, "language", sample["language"])
    sheet["prompt"] = sheet["prompt"].str.replace(r"\s+", " ", regex=True).str.slice(0, 400)
    for col in REVIEW_COLUMNS:
        sheet[col] = ""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(path, index=False)
    log.info("wrote %d-row review sheet to %s", len(sheet), path)
    return path


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion that stays sane at small n and near 0 or 1."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    # Clamped: at 0 or n successes the bounds are exactly 0 or 1 in theory and
    # come out a hair outside it in floating point, which a report would print
    # as a negative accuracy.
    return (max(0.0, centre - half), min(1.0, centre + half))


def score_review(sheet: str | Path | pd.DataFrame) -> dict[str, dict]:
    """Accuracy per marked column, with the interval, from a filled sheet.

    Marks are read leniently -- y/yes/1/true count as correct -- and blank rows
    are left out of that column's denominator rather than counted as wrong,
    so a half-finished review still scores what was reviewed.
    """
    frame = sheet if isinstance(sheet, pd.DataFrame) else pd.read_csv(sheet)
    out = {}
    for col in ("domain_correct", "domain_in_shortlist", "task_correct"):
        if col not in frame:
            continue
        marks = frame[col].astype(str).str.strip().str.lower()
        marked = marks[marks.isin({"y", "yes", "1", "true", "n", "no", "0", "false"})]
        hits = int(marked.isin({"y", "yes", "1", "true"}).sum())
        low, high = wilson_interval(hits, len(marked))
        out[col] = {"n": len(marked), "correct": hits,
                    "accuracy": hits / len(marked) if len(marked) else float("nan"),
                    "ci_low": low, "ci_high": high}
    return out


def plot_distribution(report: DistributionReport, path: str | Path) -> Path:
    """Predicted domain share on arena traffic, beside the training label share.

    Reusable on any :class:`DistributionReport`. The comparison is the point: a
    tall bar means little until it sits next to what the head was taught.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(report.domain_share, key=lambda k: -report.domain_share[k])
    y = np.arange(len(names))
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5),
                                      gridspec_kw={"width_ratios": [3, 2]})
    left.barh(y - 0.2, [report.domain_share[n] for n in names], height=0.4,
              color="#4C72B0", label="predicted on arena traffic")
    if report.training_domain_share:
        left.barh(y + 0.2, [report.training_domain_share.get(n, 0.0) for n in names],
                  height=0.4, color="#BBBBBB", label="training labels")
    left.set_yticks(y, names)
    left.invert_yaxis()
    left.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    left.set_title(f"Domain share — effective domains {report.effective_domains:.1f} "
                   f"of {len(names)}; top two hold {report.top2_share:.0%}", fontsize=10)
    left.legend(fontsize=8, loc="lower right")

    tasks = sorted(report.task_share, key=lambda k: -report.task_share[k])
    right.barh(tasks, [report.task_share[t] for t in tasks], color="#DD8452")
    right.invert_yaxis()
    right.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    right.set_title("Task share (predicted)", fontsize=10)

    fig.suptitle(f"Where the classifier predictions land — {report.n} arena prompts", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


#: RouterBench subsets whose domain is unambiguous under the taxonomy's own
#: definitions (``domain_taxonomy.DOMAIN_DESCRIPTIONS``), in the merged
#: 8-domain space. Anything that could defensibly go two ways is left out --
#: `moral-scenarios`, `professional-psychology`, `electrical-engineering`, the
#: Chinese-literature sets, HellaSwag -- so every scored row is scored against a
#: label the taxonomy itself dictates, not one chosen by the person scoring.
BENCHMARK_DOMAIN: dict[str, str] = {
    **dict.fromkeys([
        "abstract-algebra", "astronomy", "college-biology", "college-chemistry",
        "college-mathematics", "college-physics", "conceptual-physics",
        "elementary-mathematics", "high-school-biology", "high-school-chemistry",
        "high-school-mathematics", "high-school-physics", "high-school-statistics",
        "formal-logic", "grade-school-math", "arc-challenge"], "science_math"),
    **dict.fromkeys([
        "college-computer-science", "high-school-computer-science", "computer-security",
        "machine-learning", "mbpp"], "software_tech"),
    **dict.fromkeys([
        "anatomy", "clinical-knowledge", "college-medicine", "professional-medicine",
        "nutrition", "virology", "human-aging"], "medicine_health"),
    **dict.fromkeys([
        "econometrics", "high-school-macroeconomics", "high-school-microeconomics",
        "management", "marketing", "professional-accounting", "public-relations",
        "professional-law", "jurisprudence", "international-law",
        "high-school-government-and-politics", "us-foreign-policy",
        "security-studies"], "business_law"),
    **dict.fromkeys([
        "high-school-european-history", "high-school-us-history",
        "high-school-world-history", "prehistory", "high-school-geography",
        "philosophy", "world-religions", "sociology", "moral-disputes"], "culture"),
}


def expected_domain(eval_name: str) -> str | None:
    """The domain a RouterBench subset belongs to, or None if it is ambiguous."""
    name = eval_name.split(".")[0]
    if name.startswith("mmlu-"):
        name = name[len("mmlu-"):]
    return BENCHMARK_DOMAIN.get(name)


def score_against_benchmark(preds: pd.DataFrame) -> dict:
    """Domain accuracy where the benchmark fixes the answer.

    The objective half of the spot check: no reader's judgement involved. It
    covers only the unambiguous subsets, so it measures accuracy on clean,
    well-formed exam prompts -- an upper bound on what open traffic will see,
    not an estimate of it.
    """
    frame = preds.assign(expected=preds["eval_name"].map(expected_domain)).dropna(
        subset=["expected"])
    hit = frame["domain"] == frame["expected"]
    in_short = [e in s.split("|") for e, s in zip(frame["expected"], frame["domain_shortlist"],
                                                   strict=True)]
    low, high = wilson_interval(int(hit.sum()), len(frame))
    return {
        "n": len(frame), "accuracy": float(hit.mean()), "ci": (low, high),
        "in_shortlist": float(np.mean(in_short)),
        "per_domain": {k: float(v) for k, v in hit.groupby(frame["expected"]).mean().items()},
        "confusion": pd.crosstab(frame["expected"], frame["domain"]),
    }
