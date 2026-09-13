"""Labels the arena dump gives away, and whether a prompt predicts them.

The length head established the pattern: the cheapest supervision is the kind
nobody has to write. The same dump carries eleven more labels on 100% of rows
-- `is_code`, a maths flag, a creative-writing flag, an instruction-constraint
flag and score, and seven difficulty criteria -- which retires the "needs 200
hand labels" blocker for several signals that were parked behind it.

Two caveats travel with every one of them:

* **They are LLM-judged, not human ground truth.** A head trained here learns
  to reproduce an automatic judge. For gating signals (does this contain code,
  does it need maths) that is close enough to the real question. For the
  difficulty criteria it is a judgement about a judgement.
* **Learnable is not useful.** The hardness rubric composed from those seven
  criteria scores AUC 0.487 on whether the stronger model was actually needed
  -- chance. See :mod:`evaluation.prompt_decomposition.routing_value`.
  Predicting a label well says nothing about whether the label moves a
  decision, and this project has now been caught by that twice.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ARENA_GLOB = ("~/.cache/huggingface/hub/datasets--lmarena-ai--arena-human-preference-140k"
              "/snapshots/*/data/*.parquet")

#: Readable name -> column in the flattened dump. Ordered by what they serve:
#: gating first, then the difficulty criteria.
FREE_LABELS: dict[str, str] = {
    "code": "is_code",
    "math": "math_v0.1.math",
    "creative_writing": "creative_writing_v0.1.creative_writing",
    "constrained_instruction": "if_v0.1.if",
    "non_english": "non_english",
    "complexity": "criteria_v0.1.complexity",
    "creativity": "criteria_v0.1.creativity",
    "domain_knowledge": "criteria_v0.1.domain_knowledge",
    "problem_solving": "criteria_v0.1.problem_solving",
    "real_world": "criteria_v0.1.real_world",
    "specificity": "criteria_v0.1.specificity",
    "technical_accuracy": "criteria_v0.1.technical_accuracy",
}


def load_arena_labels(pattern: str = ARENA_GLOB) -> pd.DataFrame:
    """Every free label, indexed by ``arena_id``.

    Joins to the length corpus on that id, never on prompt text: the two
    pipelines extract the text differently and a text join silently returns
    nothing.
    """
    paths = sorted(glob.glob(pattern.replace("~", str(Path.home()))))
    if not paths:
        raise FileNotFoundError(f"no arena parquet files under {pattern}")
    raw = pd.concat([pd.read_parquet(p, columns=["id", "category_tag", "is_code", "language"])
                     for p in paths], ignore_index=True)
    frame = pd.json_normalize(raw["category_tag"])
    frame["is_code"] = raw["is_code"].to_numpy()
    frame["non_english"] = (raw["language"] != "en").to_numpy()
    frame.index = pd.Index(raw["id"].astype("object").tolist(), name="arena_id")
    frame = frame[~frame.index.duplicated()]
    return frame[[c for c in FREE_LABELS.values() if c in frame.columns]]


def label_for(labels: pd.DataFrame, arena_ids, name: str) -> np.ndarray:
    """One named label aligned to a corpus split, NaN where the row is missing."""
    if name not in FREE_LABELS:
        raise KeyError(f"unknown label {name!r}; have {sorted(FREE_LABELS)}")
    return labels[FREE_LABELS[name]].reindex(list(arena_ids)).to_numpy(dtype=float)
