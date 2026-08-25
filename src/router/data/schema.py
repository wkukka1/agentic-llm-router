"""The canonical record every data source is normalised into.

Every loader in :mod:`router.data.sources` emits ``Example`` rows, so the
feature/model layers never learn anything about where the data came from.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

#: Columns of the canonical parquet, in order.
COLUMNS: tuple[str, ...] = (
    "uid",
    "prompt",
    "source",
    "subset",
    "domain",
    "category",
    "task_type",
    "difficulty",
    "routing_label",
    "hardness_score",
    "n_chars",
    "meta",
)

_WHITESPACE = re.compile(r"\s+")


@dataclass(slots=True)
class Example:
    """One routable prompt plus whatever supervision the source provides.

    Attributes:
        uid: Stable id derived from ``source`` and the normalised prompt.
        prompt: The text a user would actually send to the router.
        source: Dataset the row came from (e.g. ``routerarena``).
        subset: Provenance inside the source (e.g. ``LiveCodeBench``).
        domain: Topic label, a :class:`~router.data.taxonomy.Domain` value.
        category: Finer-grained topic label, kept as free text.
        task_type: Capability label, a :class:`~router.data.taxonomy.TaskType`.
        difficulty: Ordinal label, a :class:`~router.data.taxonomy.Difficulty`.
        routing_label: The routing decision itself, when a source supervises it
            directly -- ``strong_needed`` or ``weak_sufficient``. Sources with
            only a topic label leave this ``None``.
        hardness_score: Continuous difficulty target in ``[0, 1]`` when the
            source exposes one (LMArena's 7 hardness criteria, normalised).
        meta: Source-specific extras that no downstream stage may depend on.
    """

    prompt: str
    source: str
    subset: str | None = None
    domain: str | None = None
    category: str | None = None
    task_type: str | None = None
    difficulty: str | None = None
    routing_label: str | None = None
    hardness_score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        digest = hashlib.sha1(
            f"{self.source}\x00{normalize_prompt(self.prompt)}".encode()
        )
        return digest.hexdigest()[:16]

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["uid"] = self.uid
        row["n_chars"] = len(self.prompt)
        row["meta"] = json.dumps(row["meta"], default=str)
        return {col: row[col] for col in COLUMNS}


def normalize_prompt(text: str) -> str:
    """Collapse whitespace and case for dedup/leakage checks only."""
    return _WHITESPACE.sub(" ", (text or "")).strip().lower()


def to_frame(examples: list[Example]) -> pd.DataFrame:
    """Materialise examples as a dataframe with the canonical column order."""
    if not examples:
        return pd.DataFrame(columns=list(COLUMNS))
    return pd.DataFrame([e.to_row() for e in examples], columns=list(COLUMNS))
