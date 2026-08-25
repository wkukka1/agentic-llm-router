"""LMArena human-preference battles -> canonical examples.

This source carries no topic label. What it does carry is (a) a realistic
open-ended prompt distribution, (b) weak capability flags (``is_code``,
``math_v0.1``, ``creative_writing_v0.1``), and (c) the seven ``criteria_v0.1``
hardness flags, which are the cleanest available supervision for the difficulty
regressor in the router design.

Use it for the task-type and difficulty heads, and as unlabelled/weakly-labelled
data for the domain head -- never as a domain ground truth.
"""

from __future__ import annotations

import logging

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from router.data.schema import Example
from router.data.taxonomy import task_type_from_arena_tags

log = logging.getLogger(__name__)

REPO_ID = "lmarena-ai/arena-human-preference-140k"
SOURCE = "lmarena_140k"

#: The seven binary "hard prompt" criteria LMArena annotates each battle with.
HARDNESS_CRITERIA = (
    "complexity",
    "creativity",
    "domain_knowledge",
    "problem_solving",
    "real_world",
    "specificity",
    "technical_accuracy",
)

#: Only these columns are pulled off disk; the rest of the row is ~10x larger.
_COLUMNS = [
    "id",
    "conversation_a",
    "conv_metadata",
    "category_tag",
    "language",
    "is_code",
    "model_a",
    "model_b",
    "winner",
]


def _first_user_text(conversation: list[dict] | None) -> str:
    """Extract the opening user turn as plain text, skipping image parts."""
    if not conversation:
        return ""
    for turn in conversation:
        if turn.get("role") != "user":
            continue
        parts = turn.get("content") or []
        texts = [p.get("text") or "" for p in parts if (p.get("type") or "text") == "text"]
        joined = "\n".join(t for t in texts if t).strip()
        if joined:
            return joined
    return ""


def _hardness_score(category_tag: dict | None) -> float | None:
    """Fraction of the seven hardness criteria that fired, in ``[0, 1]``."""
    criteria = (category_tag or {}).get("criteria_v0.1")
    if not criteria:
        return None
    fired = sum(1 for key in HARDNESS_CRITERIA if criteria.get(key))
    return fired / len(HARDNESS_CRITERIA)


def load(
    *,
    max_shards: int | None = None,
    language: str | None = "en",
    max_prompt_chars: int = 20_000,
) -> list[Example]:
    """Download (cached) and normalise LMArena battles.

    Args:
        max_shards: Stop after this many parquet shards. The full set is ~1.6 GB;
            one shard is ~19k battles, which is plenty for the weak-label heads.
        language: Keep only this language tag, or ``None`` for all.
        max_prompt_chars: Truncate pathological prompts rather than dropping them.
    """
    shards = sorted(f for f in list_repo_files(REPO_ID, repo_type="dataset") if f.endswith(".parquet"))
    if max_shards is not None:
        shards = shards[:max_shards]

    examples: list[Example] = []
    seen_ids: set[str] = set()

    for shard in shards:
        path = hf_hub_download(REPO_ID, shard, repo_type="dataset")
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512, columns=_COLUMNS):
            for row in batch.to_pylist():
                if language is not None and row.get("language") != language:
                    continue
                prompt = _first_user_text(row.get("conversation_a"))
                if not prompt:
                    continue

                tag = row.get("category_tag") or {}
                is_math = bool((tag.get("math_v0.1") or {}).get("math"))
                is_creative = bool(
                    (tag.get("creative_writing_v0.1") or {}).get("creative_writing")
                )
                is_code = bool(row.get("is_code"))

                meta = row.get("conv_metadata") or {}
                examples.append(
                    Example(
                        prompt=prompt[:max_prompt_chars],
                        source=SOURCE,
                        subset=shard.rsplit("/", 1)[-1],
                        domain=None,
                        category=None,
                        task_type=task_type_from_arena_tags(
                            is_code=is_code,
                            is_math=is_math,
                            is_creative_writing=is_creative,
                        ).value,
                        difficulty=None,
                        hardness_score=_hardness_score(tag),
                        meta={
                            "arena_id": row.get("id"),
                            "is_code": is_code,
                            "is_math": is_math,
                            "is_creative_writing": is_creative,
                            "if_score": (tag.get("if_v0.1") or {}).get("score"),
                            "sum_user_tokens": meta.get("sum_user_tokens"),
                            "turns": meta.get("turns"),
                            "model_a": row.get("model_a"),
                            "model_b": row.get("model_b"),
                            "winner": row.get("winner"),
                        },
                    )
                )
                seen_ids.add(row.get("id"))

        log.info("lmarena[%s]: running total %d prompts", shard, len(examples))

    return examples
