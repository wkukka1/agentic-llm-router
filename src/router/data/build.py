"""Turn raw sources into deduplicated, stratified, leakage-checked splits.

Everything downstream reads the parquet files this module writes, so splitting
happens exactly once and every experiment is scored on identical rows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from router.data.schema import normalize_prompt, to_frame
from router.data.sources import routerarena

log = logging.getLogger(__name__)

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
    examples = routerarena.load(
        "full", include_context=include_context, include_options=include_options
    )
    frame = to_frame(examples)
    frame = dedupe(frame)

    splits = split_frame(frame, stratify_on=["domain", "difficulty"], seed=seed)
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
