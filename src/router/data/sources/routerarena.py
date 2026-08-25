"""RouterArena -> canonical examples.

RouterArena is the only source in the pool with clean, human-curated topic and
difficulty labels on a routing-shaped prompt distribution, so it is the primary
supervision for the domain classifier.
"""

from __future__ import annotations

import logging

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from router.data.schema import Example
from router.data.taxonomy import (
    Difficulty,
    domain_from_routerarena,
    task_type_from_dataset_name,
)

log = logging.getLogger(__name__)

REPO_ID = "RouteWorks/RouterArena"
SOURCE = "routerarena"

#: RouterArena ships three cuts; ``full`` is the 8.4k labelled set.
SPLIT_FILES = {
    "full": "data/full-00000-of-00001.parquet",
    "sub_10": "data/sub_10-00000-of-00001.parquet",
    "robustness": "data/robustness-00000-of-00001.parquet",
}

_OPTION_LETTERS = "ABCDEFGHIJKLMNOP"


def render_prompt(
    question: str,
    context: str | None,
    options: list[str] | None,
    *,
    include_context: bool = True,
    include_options: bool = True,
) -> str:
    """Reassemble the row into the prompt a user would send.

    Options and context are toggleable because they are a double-edged sword:
    they are part of the real request, but an option block is also a strong
    format artifact that a classifier can latch onto instead of the topic. The
    experiment configs sweep both settings to measure that gap.
    """
    parts: list[str] = []
    if include_context and (context or "").strip():
        parts.append(context.strip())
    parts.append((question or "").strip())
    if include_options and options is not None and len(options) > 0:
        rendered = "\n".join(
            f"{_OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options[: len(_OPTION_LETTERS)])
        )
        parts.append(rendered)
    return "\n\n".join(p for p in parts if p)


def load(
    split: str = "full",
    *,
    include_context: bool = True,
    include_options: bool = True,
) -> list[Example]:
    """Download (cached) and normalise a RouterArena split."""
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown RouterArena split {split!r}; expected one of {list(SPLIT_FILES)}")

    path = hf_hub_download(REPO_ID, SPLIT_FILES[split], repo_type="dataset")
    table = pq.read_table(path)
    rows = table.to_pylist()

    examples: list[Example] = []
    dropped_no_domain = 0
    dropped_empty = 0

    for row in rows:
        domain = domain_from_routerarena(row.get("Domain") or "")
        if domain is None:
            dropped_no_domain += 1
            continue

        options = row.get("Options")
        prompt = render_prompt(
            row.get("Question") or "",
            row.get("Context"),
            list(options) if options is not None else None,
            include_context=include_context,
            include_options=include_options,
        )
        if not prompt.strip():
            dropped_empty += 1
            continue

        subset = (row.get("Dataset name") or "").strip() or None
        difficulty = (row.get("Difficulty") or "").strip().lower()

        examples.append(
            Example(
                prompt=prompt,
                source=SOURCE,
                subset=subset,
                domain=domain.value,
                category=(row.get("Category") or "").strip() or None,
                task_type=task_type_from_dataset_name(subset or "").value,
                difficulty=difficulty if difficulty in set(Difficulty) else None,
                hardness_score=None,
                meta={
                    "global_index": row.get("Global Index"),
                    "answer": row.get("Answer"),
                    "keywords": row.get("Keywords"),
                    "n_options": len(options) if options is not None else 0,
                },
            )
        )

    log.info(
        "routerarena[%s]: kept %d, dropped %d (no domain) + %d (empty prompt)",
        split,
        len(examples),
        dropped_no_domain,
        dropped_empty,
    )
    return examples
