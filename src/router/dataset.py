"""Canonical rows, splitting, and dataset assembly.

Loaders live in :mod:`router.sources`. This module owns the row type every
loader emits, and the once-only split that every experiment scores against.

Two invariants are enforced here rather than trusted:

* **No duplicate prompts.** Sources overlap; a prompt in both train and test
  turns memorisation into apparent skill.
* **No leakage between splits**, asserted after splitting rather than assumed.

Splits are stratified by ``domain`` *and* ``source``, so every split holds every
distribution and per-source accuracy is measurable. That last part matters more
than it sounds: a classifier that scores 91% on exam questions and 47% on real
prompts looks fine in aggregate.
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
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
#: Never regenerated. See build_dataset for why.
FROZEN_EVAL_PATH = Path("data/handlabelled/eval_frozen.parquet")
SPLIT_NAMES = ("train", "val", "test")

COLUMNS: tuple[str, ...] = (
    "uid", "prompt", "source", "subset", "domain", "difficulty", "n_chars", "meta",
)

_WHITESPACE = re.compile(r"\s+")


@dataclass(slots=True)
class Example:
    """One labelled prompt.

    ``capability`` is accepted as an alias for ``domain`` so the loaders read
    naturally either way; both populate the same field.
    """

    prompt: str
    source: str
    subset: str | None = None
    capability: str | None = None
    difficulty: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def domain(self) -> str | None:
        return self.capability

    @property
    def uid(self) -> str:
        digest = hashlib.sha1(normalize_prompt(self.prompt).encode())
        return digest.hexdigest()[:16]

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["uid"] = self.uid
        row["domain"] = self.capability
        row["n_chars"] = len(self.prompt)
        row["meta"] = json.dumps(row.get("meta") or {}, default=str)
        return {c: row.get(c) for c in COLUMNS}


def normalize_prompt(text: str) -> str:
    """Collapse whitespace and case, for dedup and leakage checks only."""
    return _WHITESPACE.sub(" ", (text or "")).strip().lower()


def to_frame(examples: list[Example]) -> pd.DataFrame:
    if not examples:
        return pd.DataFrame(columns=list(COLUMNS))
    return pd.DataFrame([e.to_row() for e in examples], columns=list(COLUMNS))


def dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop exact prompt duplicates, keeping the first occurrence.

    Sources are ordered most-trusted-first by the builder, so the survivor of a
    collision is the row with the better label.
    """
    before = len(frame)
    key = frame["prompt"].map(normalize_prompt)
    frame = frame.loc[~key.duplicated()].reset_index(drop=True)
    if before != len(frame):
        log.info("dedupe: %d -> %d (%d duplicates)", before, len(frame), before - len(frame))
    return frame


def _stratify_key(frame: pd.DataFrame, columns: list[str], min_count: int = 3) -> pd.Series:
    """Composite stratification key that degrades safely.

    Two back-offs, because stratifying needs at least two members per group:
    a cell thinner than ``min_count`` falls back to the first column alone, and
    anything *still* alone is pooled into one bucket. Without the second pass a
    single rare (domain, source) pair aborts the whole split.
    """
    key = frame[columns].astype(str).agg("|".join, axis=1)
    counts = key.value_counts()
    key = key.where(~key.isin(counts[counts < min_count].index), frame[columns[0]].astype(str))
    counts = key.value_counts()
    # Anything still alone joins the largest group. Pooling singletons into
    # their own bucket does not help -- a bucket of one cannot be stratified
    # either -- and folding them into a big group costs nothing.
    return key.where(~key.isin(counts[counts < 2].index), counts.idxmax())


def split_frame(frame: pd.DataFrame, *, stratify_on: list[str], val_size: float = 0.15,
                test_size: float = 0.15, seed: int = 20260826) -> dict[str, pd.DataFrame]:
    key = _stratify_key(frame, stratify_on)
    train_idx, holdout_idx = train_test_split(
        frame.index, test_size=val_size + test_size, stratify=key, random_state=seed)

    # Re-derive the key on the holdout alone: a cell with three members overall
    # can land here with one, which stratification cannot split.
    holdout_key = _stratify_key(frame.loc[holdout_idx], stratify_on)
    val_idx, test_idx = train_test_split(
        holdout_idx, test_size=test_size / (val_size + test_size),
        stratify=holdout_key, random_state=seed)
    return {"train": frame.loc[train_idx].reset_index(drop=True),
            "val": frame.loc[val_idx].reset_index(drop=True),
            "test": frame.loc[test_idx].reset_index(drop=True)}


def assert_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    seen: dict[str, str] = {}
    for name, frame in splits.items():
        for prompt in frame["prompt"]:
            norm = normalize_prompt(prompt)
            prior = seen.get(norm)
            if prior is not None and prior != name:
                raise AssertionError(f"prompt leaks between {prior!r} and {name!r}: {prompt[:80]!r}")
            seen[norm] = name


def load_splits(variant: str = "domain_v3", out_dir: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    target = out_dir / variant
    missing = [n for n in SPLIT_NAMES if not (target / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"missing splits {missing} under {target}; run `router build-data`")
    return {name: pd.read_parquet(target / f"{name}.parquet") for name in SPLIT_NAMES}


def build_dataset(
    *,
    arena_shards: int = 3,
    use_mmlu_pro: bool = True,
    use_routerarena: bool = True,
    use_bigbench: bool = True,
    synthetic_domains: list[str] | None = None,
    real_eval_size: int = 250,
    hand_oversample: int = 4,
    cap_per_source_domain: int | None = 1200,
    out_dir: Path = PROCESSED_DIR,
    variant: str = "domain_v3",
    seed: int = 20260826,
) -> dict[str, pd.DataFrame]:
    """Assemble every source, then split.

    ``real_eval_size`` hand-labelled real prompts are held out *before* anything
    else and placed directly into the test split. They are the honest measure:
    genuine user traffic, labelled by hand, never seen in training.
    """
    from router import sources

    # Dedupe hand labels *before* carving off the eval set: a prompt labelled
    # twice would otherwise put one copy in eval and one in train.
    hand = dedupe(to_frame(sources.load_handlabelled()))

    # The eval set is FROZEN on disk and matched by prompt text, not resampled
    # per run. Earlier versions drew a fresh sample each build, which silently
    # made results from different runs incomparable -- including a learning
    # curve that was fitted across two different test sets. Never resample this.
    frozen = pd.read_parquet(FROZEN_EVAL_PATH)
    eval_keys = set(frozen["prompt"].map(normalize_prompt))
    is_eval = hand["prompt"].map(normalize_prompt).isin(eval_keys)
    hand_eval = hand[is_eval].assign(source="handlabelled_eval").reset_index(drop=True)
    hand_train = hand[~is_eval].reset_index(drop=True)
    log.info("frozen eval set: %d rows; %d hand rows remain for training",
             len(hand_eval), len(hand_train))

    # Hand labels are the only rows pairing real traffic with all 12 domains,
    # and there are ~750 of them against tens of thousands of benchmark rows.
    # Repeating them raises their weight in the loss without discarding the
    # benchmark signal; duplicates are added after dedupe so they survive it.
    hand_repeated = pd.concat([hand_train] * max(hand_oversample, 1), ignore_index=True)

    # Ordered most-trusted-first: dedupe keeps the first occurrence.
    parts = [hand_train, to_frame(sources.load_arena_flagged(max_shards=arena_shards))]
    if use_mmlu_pro:
        parts.append(to_frame(sources.load_mmlu_pro()))
    if use_routerarena:
        parts.append(to_frame(sources.load_routerarena_bycategory()))
    if use_bigbench:
        parts.append(to_frame(sources.load_bigbench()))
    if synthetic_domains:
        parts.append(to_frame(sources.load_synthetic(synthetic_domains)))

    frame = dedupe(pd.concat(parts, ignore_index=True))

    # The eval prompts are carved out before dedupe runs, so a benchmark row
    # with identical text would survive here and leak into training. Drop any
    # such row: the held-out real prompts must appear nowhere else.
    eval_keys = set(hand_eval["prompt"].map(normalize_prompt))
    before = len(frame)
    frame = frame[~frame["prompt"].map(normalize_prompt).isin(eval_keys)].reset_index(drop=True)
    if before != len(frame):
        log.info("dropped %d row(s) colliding with the held-out eval set", before - len(frame))

    # Cap any one (source, domain) cell. Without this, arena's `is_code` flag
    # contributes ~7k software_tech rows and the model learns that domain at the
    # expense of the eleven others.
    if cap_per_source_domain:
        capped = []
        for _, group in frame.groupby(["source", "domain"], observed=True):
            capped.append(group.sample(min(len(group), cap_per_source_domain),
                                       random_state=seed))
        frame = pd.concat(capped, ignore_index=True)
        log.info("after per-(source,domain) cap: %d rows", len(frame))

    splits = split_frame(frame, stratify_on=["domain", "source"], seed=seed)

    # Oversampling applies to training only -- never to val or test.
    extra = hand_repeated[~hand_repeated["uid"].isin(
        set(splits["val"]["uid"]) | set(splits["test"]["uid"]))]
    splits["train"] = pd.concat([splits["train"], extra], ignore_index=True)

    # Held-out real prompts join the test split only.
    splits["test"] = pd.concat([splits["test"], hand_eval], ignore_index=True)
    assert_no_leakage(splits)

    target = out_dir / variant
    target.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(target / f"{name}.parquet", index=False)
        log.info("wrote %s: %d rows", target / f"{name}.parquet", len(part))
    return splits


def build_real_only_dataset(
    *,
    merge_domains: bool = False,
    out_dir: Path = PROCESSED_DIR,
    variant: str = "real_only",
    seed: int = 20260827,
) -> dict[str, pd.DataFrame]:
    """Splits containing only hand-labelled real prompts.

    This is what the shipped model trains on. Benchmark rows are excluded
    entirely: they are exam-formatted, and a model trained on them scored 0.91
    on benchmarks and 0.47 on real traffic. They remain useful for pretraining
    a transformer (see ``build_dataset``), but the frozen-encoder ensemble does
    not need them and is measurably better without.

    The evaluation rows are the frozen set, matched by prompt text, so results
    stay comparable across runs and across changes to the label pool.
    """
    from router import sources

    hand = dedupe(to_frame(sources.load_handlabelled()))
    if merge_domains:
        # Applied here rather than in the stored labels: merging is lossy and
        # one-way, so the 10-class labels stay the source of truth.
        from router.taxonomy import apply_domain_merges

        hand["domain"] = hand["domain"].map(apply_domain_merges)
        log.info("merged domains -> %d classes", hand["domain"].nunique())
    frozen = pd.read_parquet(FROZEN_EVAL_PATH)
    is_eval = hand["prompt"].map(normalize_prompt).isin(
        set(frozen["prompt"].map(normalize_prompt))
    )
    hand_eval = hand[is_eval].assign(source="handlabelled_eval").reset_index(drop=True)
    pool = hand[~is_eval].reset_index(drop=True)
    log.info("real_only: %d train pool, %d frozen eval", len(pool), len(hand_eval))

    # Small val/test off the pool; the frozen set is appended to test and is
    # the split every reported number comes from.
    splits = split_frame(pool, stratify_on=["domain"], val_size=0.15, test_size=0.05, seed=seed)
    splits["test"] = pd.concat([splits["test"], hand_eval], ignore_index=True)
    assert_no_leakage(splits)

    target = out_dir / variant
    target.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(target / f"{name}.parquet", index=False)
        log.info("wrote %s: %d rows", target / f"{name}.parquet", len(part))
    return splits
