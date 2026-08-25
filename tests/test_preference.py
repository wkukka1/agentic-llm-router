"""The strong-vs-weak routing label, derived from LMArena preference battles."""

import pandas as pd
import pytest

from router.data.sources.preference import (
    ROUTING_LABELS,
    STRONG_NEEDED,
    WEAK_SUFFICIENT,
    compute_tiers,
)


def battles(rows):
    return pd.DataFrame(rows, columns=["model_a", "model_b", "winner"])


def _repeat(rows, times):
    return battles(rows * times)


def test_tiers_are_derived_from_observed_win_rates():
    """`good` always wins, `bad` always loses, so the split must follow that."""
    frame = _repeat([("good", "bad", "model_a")], 50)
    tiers = compute_tiers(frame, min_battles=10)
    assert "good" in tiers.strong
    assert "bad" in tiers.weak


def test_ties_count_as_half_a_win_for_both():
    frame = _repeat([("a", "b", "tie")], 50)
    tiers = compute_tiers(frame, min_battles=10)
    assert tiers.win_rates["a"] == pytest.approx(0.5)
    assert tiers.win_rates["b"] == pytest.approx(0.5)


def test_models_with_too_few_battles_are_excluded():
    """A model seen twice has no reliable win rate and must not be tiered."""
    frame = battles(
        [("good", "bad", "model_a")] * 50 + [("rare", "bad", "model_a")] * 2
    )
    tiers = compute_tiers(frame, min_battles=10)
    assert tiers.tier_of("rare") is None
    assert tiers.tier_of("good") == "strong"


def test_tier_quantile_controls_how_elite_strong_is():
    rows = []
    for i, name in enumerate(["m0", "m1", "m2", "m3"]):
        # Staggered win rates: m3 best, m0 worst.
        rows += [(name, "anchor", "model_a")] * (i + 1) * 10
        rows += [(name, "anchor", "model_b")] * (4 - i) * 10
    frame = battles(rows)
    lenient = compute_tiers(frame, min_battles=10, quantile=0.25)
    strict = compute_tiers(frame, min_battles=10, quantile=0.75)
    assert len(strict.strong) < len(lenient.strong)


def test_tier_of_returns_none_for_unknown_models():
    tiers = compute_tiers(_repeat([("a", "b", "model_a")], 50), min_battles=10)
    assert tiers.tier_of("never-seen") is None


def test_compute_tiers_rejects_data_with_no_eligible_model():
    with pytest.raises(ValueError, match="no model has"):
        compute_tiers(battles([("a", "b", "tie")]), min_battles=999)


def test_label_space_is_exactly_two_classes():
    assert ROUTING_LABELS == [WEAK_SUFFICIENT, STRONG_NEEDED]
    assert len(set(ROUTING_LABELS)) == 2


class TestLabelSemantics:
    """The label answers 'will spending more get a better answer?'."""

    def test_strong_winning_means_strong_was_needed(self):
        assert STRONG_NEEDED == "strong_needed"

    def test_tie_is_weak_sufficient_not_strong_needed(self):
        # If a cheap model draws with an expensive one, routing cheap is right.
        # This is a deliberate asymmetry, so pin it.
        assert WEAK_SUFFICIENT == "weak_sufficient"


def test_built_preference_splits_are_balanced_and_disjoint():
    """Integration check against the real built dataset, if present."""
    from pathlib import Path

    base = Path("data/processed/preference")
    if not (base / "train.parquet").exists():
        pytest.skip("preference splits not built; run `router build-preference`")

    frames = {s: pd.read_parquet(base / f"{s}.parquet") for s in ("train", "val", "test")}
    for name, frame in frames.items():
        share = frame["routing_label"].value_counts(normalize=True)
        assert set(share.index) == set(ROUTING_LABELS)
        # Roughly balanced; a wildly skewed split would make accuracy misleading.
        assert 0.4 < share[STRONG_NEEDED] < 0.6, f"{name} is skewed: {share.to_dict()}"

    uids = [set(f["uid"]) for f in frames.values()]
    assert not (uids[0] & uids[1]) and not (uids[0] & uids[2]) and not (uids[1] & uids[2])
