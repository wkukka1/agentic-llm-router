"""Pluggable model pools, and the plateau diagnostics from arXiv 2606.07587.

The hard/easy split is the paper's Sec 5.1 definition and the whole analysis
rests on it, so the fixtures here are hand-built cases where the label is not a
matter of opinion: a prompt only one model solves, a prompt the best model
misses but several others get, a prompt everyone solves, a prompt nobody does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.prompt_decomposition.per_model import (
    knn_router,
    plateau_report,
    selection_concentration,
    spendable,
    stratify_hard_easy,
)
from training.prompt_decomposition.data.pools import (
    POOLS,
    CorrectnessTable,
    benchmark_family,
    hashed_split,
    load_pool,
    load_pool_from_frame,
    register_pool,
)


def _table(correct, cost=None, categories=None, split=None, name="t"):
    correct = np.asarray(correct, dtype=float)
    n, m = correct.shape
    prompts = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)], "prompt": [f"p{i}" for i in range(n)],
        "benchmark": categories or ["b"] * n,
        "split": split or [("train", "test")[i % 2] for i in range(n)]})
    return CorrectnessTable(prompts, correct,
                            np.asarray(cost if cost is not None else np.full((n, m), np.nan)),
                            [f"m{j}" for j in range(m)], name)


class TestPoolPlumbing:
    def test_a_new_pool_is_one_registration(self):
        register_pool("_tiny", lambda: _table([[1, 0], [0, 1]]))
        try:
            assert load_pool("_tiny").correct.shape == (2, 2)
        finally:
            POOLS.pop("_tiny")

    def test_registering_a_name_twice_is_refused(self):
        register_pool("_dupe", lambda: _table([[1, 0]]))
        try:
            with pytest.raises(ValueError, match="already registered"):
                register_pool("_dupe", lambda: _table([[0, 1]]))
        finally:
            POOLS.pop("_dupe")

    def test_an_unknown_pool_names_the_registered_ones(self):
        with pytest.raises(KeyError, match="unknown pool"):
            load_pool("nope")

    def test_a_wide_frame_becomes_a_pool_without_writing_a_loader(self):
        frame = pd.DataFrame({"prompt": ["a", "b"], "benchmark": ["x", "y"],
                              "gpt": [1.0, 0.0], "llama": [0.0, 1.0]})
        table = load_pool_from_frame(frame, ["gpt", "llama"])
        assert table.models == ["gpt", "llama"]
        assert table.correct.shape == (2, 2)
        assert not table.has_cost                      # no cost columns given

    def test_a_frame_missing_a_model_column_says_which(self):
        frame = pd.DataFrame({"prompt": ["a"], "benchmark": ["x"], "gpt": [1.0]})
        with pytest.raises(KeyError, match="llama"):
            load_pool_from_frame(frame, ["gpt", "llama"])

    def test_the_split_is_stable_under_a_rebuild(self):
        assert hashed_split(["a", "b", "c"]) == hashed_split(["a", "b", "c"])

    @pytest.mark.parametrize("name, family", [
        ("mmlu-professional-law", "mmlu"), ("grade-school-math", "grade-school-math"),
        ("chinese_zodiac", "chinese"), ("hellaswag.dev", "hellaswag")])
    def test_benchmark_families_survive_their_hyphens(self, name, family):
        assert benchmark_family(name) == family


class TestUnpricedPools:
    def test_missing_prices_become_zeros_not_ones(self):
        """A caller once substituted 1.0 and the report printed a column of
        dollars that were not dollars."""
        costs, priced = spendable(_table([[1, 0], [0, 1]]))
        assert not priced
        assert (costs == 0).all()

    def test_a_priced_pool_is_left_alone(self):
        costs, priced = spendable(_table([[1, 0]], cost=[[0.5, 0.1]]))
        assert priced and costs[0][0] == 0.5


class TestHardEasySplit:
    """The paper's definition, case by case (Sec 5.1)."""

    def _pool(self):
        # model 0 is best overall; 12 models so the `few` bound is max(8, 6) = 8
        rows = [
            [1] + [0] * 11,          # only one model solves it, even though it is
                                     # the best one: HARD (paper: n_correct == 1)
            [0, 1] + [0] * 10,       # exactly one model, and not the best: HARD
            [0] + [1] * 5 + [0] * 6,  # few correct, best model wrong: HARD
            [1] * 12,                # everybody: easy
            [0] * 12,                # nobody: unsolvable
            [0] + [1] * 11,          # many correct, best wrong -> not `few`: easy
        ]
        # ... plus a block only m0 solves, so m0 is unambiguously the best
        # model overall. Without it m1 had the higher column mean and "the best
        # model is wrong here" was false for the third case.
        return _table(rows * 4 + [[1] + [0] * 11] * 12)

    def test_each_case_lands_where_the_definition_says(self):
        labels = list(stratify_hard_easy(self._pool())[:6])
        assert labels == ["hard", "hard", "hard", "easy", "unsolvable", "easy"]

    def test_a_single_solver_is_hard_even_when_it_is_the_best_model(self):
        """`n_correct == 1` is hard in the paper regardless of which model it
        is -- the router still has to find that one model."""
        table = _table([[1, 0, 0]] * 10 + [[1, 1, 1]] * 10)
        assert (stratify_hard_easy(table)[:10] == "hard").all()

    def test_graded_scores_are_binarised_at_the_threshold(self):
        table = _table([[0.6, 0.4, 0.4]] * 6)
        assert (stratify_hard_easy(table, threshold=0.5) == "hard").all()
        assert (stratify_hard_easy(table, threshold=0.7) == "unsolvable").all()


class TestPlateauReport:
    def test_the_gap_concentrates_on_the_hard_prompts(self):
        """The paper's headline: a small hard subset holds most of the gap."""
        easy = [[1, 1, 0]] * 80          # best model already right
        hard = [[0, 0, 1]] * 20          # only model 2, and the best model is wrong
        r = plateau_report(_table(easy + hard))
        assert r.f_hard == pytest.approx(0.2)
        assert r.gap_easy == pytest.approx(0.0)
        assert r.gap_hard == pytest.approx(1.0)
        assert r.explained_by_hard == pytest.approx(1.0)

    def test_unsolvable_prompts_are_excluded_from_both_subsets(self):
        r = plateau_report(_table([[1, 1]] * 10 + [[0, 0]] * 10))
        assert r.f_unsolvable == pytest.approx(0.5)
        assert r.f_hard == pytest.approx(0.0)

    def test_the_summary_names_the_baseline_model(self):
        assert "m0" in plateau_report(_table([[1, 0]] * 10)).summary()


class TestSelectionConcentration:
    def test_a_router_that_always_picks_one_model_shows_it(self):
        """The Fig. 4 pathology: identical distribution on hard and on all
        queries means the router is defaulting, not routing."""
        table = _table([[1, 1, 0]] * 40 + [[0, 1, 0]] * 10)
        frame = selection_concentration(table, np.zeros(50, dtype=int))
        assert frame.loc["m0", "router_all"] == pytest.approx(1.0)
        assert frame.loc["m0", "router_hard"] == pytest.approx(1.0)
        assert frame.loc["m1", "oracle_hard_mass"] > 0     # where the answers actually are


class TestKnnRouter:
    def test_it_copies_the_best_model_from_similar_training_prompts(self):
        """Two clusters, each with its own specialist; kNN should learn that
        from neighbours alone."""
        rng = np.random.default_rng(0)
        left = rng.normal([-5, 0], 0.1, (60, 2))
        right = rng.normal([5, 0], 0.1, (60, 2))
        vectors = np.vstack([left, right])
        correct = np.array([[1, 0]] * 60 + [[0, 1]] * 60, dtype=float)
        table = _table(correct, cost=np.zeros((120, 2)),
                       split=["train"] * 50 + ["test"] * 10 + ["train"] * 50 + ["test"] * 10)
        is_train = (table.prompts.split == "train").to_numpy()
        r = knn_router(table.subset(is_train), table.subset(~is_train),
                       vectors[is_train], vectors[~is_train], k=5)
        assert r.accuracy == pytest.approx(1.0)
