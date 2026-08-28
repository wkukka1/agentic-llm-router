"""Canonical rows, dedup, splitting, and the leakage guard."""

import numpy as np
import pandas as pd
import pytest

from router.dataset import (
    Example,
    assert_no_leakage,
    dedupe,
    normalize_prompt,
    split_frame,
    to_frame,
)


def _frame(n=240, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "prompt": [f"prompt number {i} about things" for i in range(n)],
        "domain": rng.choice(["science_math", "humanities", "language"], n),
        "source": rng.choice(["handlabelled", "mmlu_pro"], n),
    })


def test_normalize_prompt_collapses_whitespace_and_case():
    assert normalize_prompt("  Hello\n\tWorld  ") == "hello world"


def test_uid_ignores_whitespace_and_case():
    """Dedup keys on uid, so it must not be defeated by formatting."""
    assert Example(prompt="What is 2+2?", source="a").uid == \
           Example(prompt="what is   2+2?", source="b").uid


def test_to_row_exposes_capability_as_domain():
    """Loaders set `capability`; every consumer reads `domain`."""
    row = Example(prompt="x", source="s", capability="science_math").to_row()
    assert row["domain"] == "science_math"
    assert row["n_chars"] == 1


def test_to_frame_empty_keeps_schema():
    assert list(to_frame([]).columns)[:3] == ["uid", "prompt", "source"]


def test_dedupe_drops_case_and_whitespace_variants():
    frame = pd.DataFrame({"prompt": ["Same thing", "same   thing", "different"]})
    assert len(dedupe(frame)) == 2


def test_dedupe_keeps_the_first_occurrence():
    """The builder orders sources most-trusted-first, so first wins matters."""
    frame = pd.DataFrame({"prompt": ["dup", "dup"], "source": ["handlabelled", "mmlu_pro"]})
    assert dedupe(frame).iloc[0]["source"] == "handlabelled"


def test_split_is_disjoint_and_leakage_free():
    splits = split_frame(_frame(), stratify_on=["domain", "source"])
    assert sum(len(v) for v in splits.values()) == 240
    assert_no_leakage(splits)


def test_split_preserves_label_proportions():
    frame = _frame(600)
    splits = split_frame(frame, stratify_on=["domain", "source"])
    overall = frame["domain"].value_counts(normalize=True)
    for part in splits.values():
        share = part["domain"].value_counts(normalize=True)
        assert np.allclose(share[overall.index], overall, atol=0.06)


def test_split_survives_a_singleton_stratum():
    """A (domain, source) cell of one used to abort the whole split."""
    frame = _frame(200)
    frame.loc[0, "domain"] = "unique_domain"
    splits = split_frame(frame, stratify_on=["domain", "source"])
    assert sum(len(v) for v in splits.values()) == 200


def test_assert_no_leakage_detects_a_shared_prompt():
    splits = {"train": pd.DataFrame({"prompt": ["shared prompt"]}),
              "test": pd.DataFrame({"prompt": ["Shared   Prompt"]})}
    with pytest.raises(AssertionError, match="leaks between"):
        assert_no_leakage(splits)


def test_handlabelled_dataset_is_present_and_consistent():
    """The hand-labelled set is the project's key asset; guard its shape."""
    from pathlib import Path

    from router.taxonomy import DOMAIN_LABELS

    path = Path("data/handlabelled/real_prompts.parquet")
    if not path.exists():
        pytest.skip("hand-labelled set not present")
    frame = pd.read_parquet(path)
    assert len(frame) >= 1000
    assert set(frame["domain"]) <= set(DOMAIN_LABELS)
    assert not frame["prompt"].duplicated().any()
    # No class should be so rare it cannot be split three ways.
    assert frame["domain"].value_counts().min() >= 20


def test_frozen_eval_set_is_present_and_stable():
    """The eval set must never be resampled.

    Regression: earlier builds drew a fresh eval sample each run, which made
    results from different runs silently incomparable -- a learning curve was
    fitted across two *different* test sets before this was caught.
    """
    from pathlib import Path

    from router.dataset import FROZEN_EVAL_PATH

    if not Path(FROZEN_EVAL_PATH).exists():
        pytest.skip("frozen eval set not present")
    frame = pd.read_parquet(FROZEN_EVAL_PATH)
    assert len(frame) == 400
    assert not frame["prompt"].duplicated().any()
    assert set(frame.columns) == {"prompt", "domain"}


def test_frozen_eval_prompts_are_all_hand_labelled():
    """Every eval prompt must exist in the labelled pool with a matching label."""
    from pathlib import Path

    from router.dataset import FROZEN_EVAL_PATH

    if not Path(FROZEN_EVAL_PATH).exists():
        pytest.skip("frozen eval set not present")
    frozen = pd.read_parquet(FROZEN_EVAL_PATH)
    hand = pd.read_parquet("data/handlabelled/real_prompts.parquet")
    merged = frozen.merge(hand, on="prompt", suffixes=("_frozen", "_hand"))
    assert len(merged) == len(frozen)
    assert (merged["domain_frozen"] == merged["domain_hand"]).all()
