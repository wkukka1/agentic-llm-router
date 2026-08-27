"""Dataset construction: canonical rows, the RouterArena source, and splits.

One module rather than a ``schema`` / ``sources`` / ``build`` package, because
there is exactly one supervised source. That structure was overhead without
benefit; if a second source is added, split it then.

Flow: ``load()`` -> ``Example`` rows -> ``to_frame()`` -> dedupe -> stratified
split -> leakage assertion -> parquet. Splitting happens once, so every
experiment scores identical rows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from sklearn.model_selection import train_test_split

from router.taxonomy import (
    Difficulty,
    capability_from_arena_flags,
    capability_from_dataset_name,
)

log = logging.getLogger(__name__)


#: Columns of the canonical parquet, in order.
COLUMNS: tuple[str, ...] = (
    "uid",
    "prompt",
    "source",
    "subset",
    "capability",
    "category",
    "difficulty",
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
        capability: What the answering model must do; a
            :class:`~router.taxonomy.Capability` value.
        category: Finer-grained topic label, kept as free text.
        task_type: Capability label, a :class:`~router.taxonomy.TaskType`.
        difficulty: Ordinal label, a :class:`~router.taxonomy.Difficulty`.
        hardness_score: Continuous difficulty target in ``[0, 1]`` when the
            source exposes one (LMArena's 7 hardness criteria, normalised).
        meta: Source-specific extras that no downstream stage may depend on.
    """

    prompt: str
    source: str
    subset: str | None = None
    capability: str | None = None
    category: str | None = None
    difficulty: str | None = None
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
    dropped_empty = 0

    for row in rows:
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
        capability = capability_from_dataset_name(subset or "")
        difficulty = (row.get("Difficulty") or "").strip().lower()

        examples.append(
            Example(
                prompt=prompt,
                source=SOURCE,
                subset=subset,
                capability=capability.value,
                category=(row.get("Category") or "").strip() or None,
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
        "routerarena[%s]: kept %d, dropped %d (empty prompt)", split, len(examples), dropped_empty
    )
    return examples


PROCESSED_DIR = Path("data/processed")
SPLIT_NAMES = ("train", "val", "test")


def dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop exact prompt duplicates (after whitespace/case normalisation).

    Duplicates that straddle a split boundary are pure leakage, and RouterArena
    does contain repeated stems across its constituent benchmarks.
    """
    before = len(frame)
    key = frame["prompt"].map(normalize_prompt)
    frame = frame.loc[~key.duplicated()].reset_index(drop=True)
    if before != len(frame):
        log.info("dedupe: dropped %d duplicate prompts (%d -> %d)", before - len(frame), before, len(frame))
    return frame


def _stratify_key(frame: pd.DataFrame, columns: list[str], min_count: int = 3) -> pd.Series:
    """Composite stratification key, backing off when a cell is too small.

    Stratifying on domain x difficulty keeps difficulty balanced inside each
    domain, but rare cells would make the split fail; those fall back to the
    first column alone.
    """
    key = frame[columns].astype(str).agg("|".join, axis=1)
    counts = key.value_counts()
    rare = counts[counts < min_count].index
    return key.where(~key.isin(rare), frame[columns[0]].astype(str))


def split_frame(
    frame: pd.DataFrame,
    *,
    stratify_on: list[str],
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 20260824,
) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split."""
    key = _stratify_key(frame, stratify_on)

    train_idx, holdout_idx = train_test_split(
        frame.index,
        test_size=val_size + test_size,
        stratify=key,
        random_state=seed,
    )
    holdout_key = key.loc[holdout_idx]
    relative_test = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=relative_test,
        stratify=holdout_key,
        random_state=seed,
    )

    return {
        "train": frame.loc[train_idx].reset_index(drop=True),
        "val": frame.loc[val_idx].reset_index(drop=True),
        "test": frame.loc[test_idx].reset_index(drop=True),
    }


def assert_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Fail loudly if a normalised prompt appears in more than one split."""
    seen: dict[str, str] = {}
    for name, frame in splits.items():
        for prompt in frame["prompt"]:
            norm = normalize_prompt(prompt)
            prior = seen.get(norm)
            if prior is not None and prior != name:
                raise AssertionError(f"prompt leaks between {prior!r} and {name!r}: {prompt[:80]!r}")
            seen[norm] = name


def build_domain_dataset(
    *,
    include_context: bool = True,
    include_options: bool = True,
    out_dir: Path = PROCESSED_DIR,
    variant: str = "full_prompt",
    seed: int = 20260824,
) -> dict[str, pd.DataFrame]:
    """Build the labelled domain-classification splits from RouterArena.

    Args:
        include_context / include_options: Prompt-rendering switches. These
            define the dataset *variant*, which is why the variant name is part
            of the output path -- two renderings are two different datasets.
        variant: Sub-directory name under ``out_dir``.
    """
    examples = load(
        "full", include_context=include_context, include_options=include_options
    )
    frame = to_frame(examples)
    frame = dedupe(frame)

    splits = split_frame(frame, stratify_on=["capability", "difficulty"], seed=seed)
    assert_no_leakage(splits)

    target = out_dir / variant
    target.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(target / f"{name}.parquet", index=False)
        log.info("wrote %s: %d rows", target / f"{name}.parquet", len(part))

    return splits


def load_splits(variant: str = "full_prompt", out_dir: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Read previously built splits, with a clear error if they are missing."""
    target = out_dir / variant
    missing = [n for n in SPLIT_NAMES if not (target / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"missing splits {missing} under {target}; run `python -m router.cli build-data --variant {variant}`"
        )
    return {name: pd.read_parquet(target / f"{name}.parquet") for name in SPLIT_NAMES}


LMARENA_REPO = "lmarena-ai/arena-human-preference-140k"
LMARENA_SOURCE = "lmarena"

_LMARENA_COLUMNS = ["id", "conversation_a", "conv_metadata", "category_tag", "language", "is_code"]


def _first_user_text(conversation: list[dict] | None) -> str:
    """The opening user turn as plain text, skipping image parts."""
    for turn in conversation or []:
        if turn.get("role") != "user":
            continue
        parts = turn.get("content") or []
        texts = [p.get("text") or "" for p in parts if (p.get("type") or "text") == "text"]
        joined = "\n".join(t for t in texts if t).strip()
        if joined:
            return joined
    return ""


def load_lmarena(
    *,
    max_shards: int | None = None,
    language: str | None = "en",
    min_chars: int = 10,
    max_prompt_chars: int = 8_000,
) -> list[Example]:
    """Real user prompts, labelled by LMArena's own capability annotations.

    This is the half of the training set that RouterArena cannot provide.
    Benchmark questions are formatted -- option blocks, context headers -- and a
    model trained only on them learns the formatting: the Dewey-era classifier
    scored 91% on benchmarks and 47% on prompts like these.

    An absent flag is a real negative, not missing data, so unflagged rows
    become ``OTHER`` rather than being dropped. That matters: roughly half of
    real traffic is ``OTHER``, and a model that never sees it will not predict
    it.
    """
    shards = sorted(f for f in list_repo_files(LMARENA_REPO, repo_type="dataset")
                    if f.endswith(".parquet"))
    if max_shards is not None:
        shards = shards[:max_shards]

    examples: list[Example] = []
    for shard in shards:
        path = hf_hub_download(LMARENA_REPO, shard, repo_type="dataset")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=_LMARENA_COLUMNS):
            for row in batch.to_pylist():
                if language is not None and row.get("language") != language:
                    continue
                prompt = _first_user_text(row.get("conversation_a"))
                if len(prompt) < min_chars:
                    continue

                tag = row.get("category_tag") or {}
                capability = capability_from_arena_flags(
                    is_code=bool(row.get("is_code")),
                    is_math=bool((tag.get("math_v0.1") or {}).get("math")),
                    is_creative_writing=bool(
                        (tag.get("creative_writing_v0.1") or {}).get("creative_writing")
                    ),
                )
                meta = row.get("conv_metadata") or {}
                examples.append(
                    Example(
                        prompt=prompt[:max_prompt_chars],
                        source=LMARENA_SOURCE,
                        subset=shard.rsplit("/", 1)[-1],
                        capability=capability.value,
                        hardness_score=_hardness_score(tag),
                        meta={"arena_id": row.get("id"), "turns": meta.get("turns")},
                    )
                )
        log.info("lmarena[%s]: running total %d", shard, len(examples))
    return examples


#: The seven binary "hard prompt" criteria LMArena annotates each battle with.
HARDNESS_CRITERIA = (
    "complexity", "creativity", "domain_knowledge", "problem_solving",
    "real_world", "specificity", "technical_accuracy",
)


def _hardness_score(category_tag: dict | None) -> float | None:
    """Fraction of the seven hardness criteria that fired, in ``[0, 1]``."""
    criteria = (category_tag or {}).get("criteria_v0.1")
    if not criteria:
        return None
    return sum(1 for k in HARDNESS_CRITERIA if criteria.get(k)) / len(HARDNESS_CRITERIA)


def build_capability_dataset(
    *,
    lmarena_shards: int | None = 3,
    include_routerarena: bool = True,
    out_dir: Path = PROCESSED_DIR,
    variant: str = "capability",
    seed: int = 20260826,
) -> dict[str, pd.DataFrame]:
    """Build capability splits from real prompts plus (optionally) benchmarks.

    Mixing the two distributions is the point. RouterArena supplies clean
    task-type signal but exam formatting; LMArena supplies the messy
    conversational prompts the router will actually see. Training on the mix is
    what stops the model keying on format.

    The split is stratified by ``capability`` **and** ``source``, so every split
    holds both distributions and per-source accuracy is measurable separately --
    which is the number that actually matters.
    """
    examples = load_lmarena(max_shards=lmarena_shards)
    if include_routerarena:
        examples += load("full", include_context=True, include_options=True)

    frame = to_frame(examples)
    frame = dedupe(frame)

    splits = split_frame(frame, stratify_on=["capability", "source"], seed=seed)
    assert_no_leakage(splits)

    target = out_dir / variant
    target.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(target / f"{name}.parquet", index=False)
        log.info("wrote %s: %d rows", target / f"{name}.parquet", len(part))
    return splits
