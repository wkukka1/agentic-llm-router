"""Scoring against externally-labelled prompt sets.

Everything else in this repo is measured against labels I wrote myself. These
sets were labelled by someone else, which makes them the only check on whether
the taxonomy means the same thing to two people.

They are held to a different standard than the frozen eval: they are *clean*,
balanced 25-per-class prompt sets, not a sample of traffic. So they measure
"does the classifier work on well-formed prompts" -- which it does, at
0.86-0.95 -- and not "what will production accuracy be", which the frozen
real-traffic eval answers at 0.773. Both numbers are real; quoting the
flattering one alone would repeat this project's original mistake.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

log = logging.getLogger(__name__)

EXTERNAL_DIR = Path("data/external")


def score(head, frame: pd.DataFrame) -> dict:
    """Score a :class:`~router.inference.DomainHead` against labelled prompts."""
    preds = head.predict_batch(frame["prompt"].tolist())
    out = frame.copy()
    out["pred"] = [p.domain for p in preds]
    out["confidence"] = [p.confidence for p in preds]
    out["in_top2"] = [row.domain in p.shortlist for row, p in zip(frame.itertuples(), preds, strict=True)]
    out["ok"] = out["domain"] == out["pred"]
    return {"frame": out, "top1": float(out["ok"].mean()), "top2": float(out["in_top2"].mean())}


def render(result: dict, name: str) -> str:
    """Markdown report for one external set."""
    f = result["frame"]
    labels = sorted(f["domain"].unique())
    precision, recall, f1, support = precision_recall_fscore_support(
        f["domain"], f["pred"], labels=labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "n": support}, index=labels
    ).sort_values("f1", ascending=False)
    cm = pd.DataFrame(
        confusion_matrix(f["domain"], f["pred"], labels=labels),
        index=[x[:14] for x in labels], columns=[x[:14] for x in labels],
    )
    errors = f[~f["ok"]]
    lines = [
        f"# External eval: `{name}`",
        "",
        f"- {len(f)} prompts, median {int(f['prompt'].str.len().median())} chars",
        f"- **top-1 {result['top1']:.4f}  |  top-2 {result['top2']:.4f}**",
        "",
        "## Per class", "", per_class.round(3).to_markdown(), "",
        "## Confusion (rows = external label)", "", cm.to_markdown(), "",
        f"## Errors ({len(errors)})", "",
    ]
    lines += [
        f"- `{r.domain}` → `{r.pred}` @{r.confidence:.2f} — {r.prompt[:90]}"
        for r in errors.itertuples()
    ]
    return "\n".join(lines) + "\n"
