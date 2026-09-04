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
#: Never regenerated: it was once resampled per build, which silently made
#: runs incomparable and produced a learning curve fitted across two
#: different test sets.
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


def build_task_dataset(
    *,
    include_synthetic: bool = True,
    include_mined: bool = False,
    out_dir: Path = PROCESSED_DIR,
    variant: str = "task",
    seed: int = 20260902,
) -> dict[str, pd.DataFrame]:
    """Splits for the task-type head: real prompts, hand-labelled by task.

    Dolly-15k is deliberately not here. A head trained on its 14,776 rows scores
    0.700 on real prompts, below the 0.729 of always predicting `answer`, and
    mixing it in costs accuracy at every ratio tried. See :mod:`router.tasktype`.

    ``include_synthetic`` adds 780 non-real prompts for the four weakest
    classes: 377 hand-written and 403 agent-generated. Like the mined rows they
    are pinned to **train**; the evaluation set stays real throughout. Measured
    against the random 1,000:

        real only                  0.802 top-1 / 0.933 top-2 / macro-F1 0.524
        + 377 hand-written         0.787        / 0.944       / macro-F1 0.573
        + 403 agent-generated too  0.793        / 0.950       / macro-F1 0.581

        vs real only, hand + generated:
          top-1     -0.010  [-0.031, +0.012]   not significant
          macro-F1  +0.055  [-0.019, +0.128]   not significant

    On by default despite neither difference being significant, for a reason
    the intervals do not carry: `extract` goes from never being predicted at
    all (F1 0.000) to being predicted (0.250), and `classify` from 0.476 to
    0.552. That is a capability appearing, not a metric drifting, and top-2 --
    what the router actually consumes -- improves.

    The wide intervals are the eval set, not the effect: 1,000 random real
    prompts contain 3 `extract` and 11 `summarize` rows, so macro-F1 cannot be
    measured tightly on them however good the model gets. Fixing that needs
    more *labelled real* prompts of those classes, not more training data.

    ``include_mined`` adds 240 prompts found by scoring the unlabelled pool for
    the rare classes and hand-labelling the top candidates. Those rows are a
    biased sample -- chosen because the model already leaned that way -- so they
    are pinned to **train** and never appear in val or test. Even so they are off
    by default: measured against the random sample they move macro-F1 +0.025 and
    top-1 -0.013, neither of which clears a paired bootstrap at n=1,000.
    """
    from router import sources

    # `to_frame` writes the label into `domain` (the canonical field on
    # Example); copy it to `task` so a split file carries an unambiguous name
    # and the experiment runner can select on it.
    frame = dedupe(to_frame(sources.load_real_tasks()))
    frame["task"] = frame["domain"]
    splits = split_frame(frame, stratify_on=["task"], val_size=0.15, test_size=0.20, seed=seed)
    if include_synthetic:
        extra = to_frame(sources.load_synthetic_tasks() + sources.load_generated_tasks())
        extra["task"] = extra["domain"]
        splits["train"] = pd.concat([splits["train"], extra], ignore_index=True)
        log.info("task: +%d synthetic rows into train", len(extra))
    if include_mined:
        mined = to_frame(sources.load_mined_tasks())
        mined["task"] = mined["domain"]
        held = set(splits["val"]["prompt"]) | set(splits["test"]["prompt"])
        mined = mined[~mined["prompt"].isin(held)]
        splits["train"] = pd.concat([splits["train"], mined], ignore_index=True)
        log.info("task: +%d mined rows into train", len(mined))
    assert_no_leakage(splits)

    target = out_dir / variant
    target.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(target / f"{name}.parquet", index=False)
    log.info("task splits: %s", {k: len(v) for k, v in splits.items()})
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
    a transformer, but fine-tuning was rejected for seed instability and the
    loaders went with it -- see EXPERIMENTS.md.

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
