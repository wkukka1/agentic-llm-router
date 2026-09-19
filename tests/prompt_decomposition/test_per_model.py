"""Per-model correctness: does knowing the category change the best model?

The fixtures are small correctness tables built so the right answer is known:
one where a category genuinely changes the best model, and one where the same
model wins everywhere. The module has to tell those apart, and has to credit
the gains to the right place -- between categories or within them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.prompt_decomposition.per_model import (
    CorrectnessTable,
    benchmark_family,
    build_report,
    category_router,
    cost_frontier,
    gap_decomposition,
    plot_frontier,
    plot_report,
)


def _table(correct, categories, cost=None, split=None):
    correct = np.asarray(correct, dtype=float)
    n, m = correct.shape
    cost = np.asarray(cost if cost is not None else np.ones((n, m)), dtype=float)
    prompts = pd.DataFrame({"prompt": [f"p{i}" for i in range(n)], "category": categories,
                            "split": split or ["train", "test"] * (n // 2)})
    return CorrectnessTable(prompts, correct, cost, [f"m{j}" for j in range(m)])


@pytest.fixture
def specialists():
    """Model 0 is right on every `code` prompt, model 1 on every `prose` one.
    One model for everything gets half; routing by category gets all of it.
    80 per category so each has 40 training rows, clear of `min_train=30` --
    below it the router correctly refuses to trust a per-category choice."""
    rows = [[1, 0]] * 80 + [[0, 1]] * 80
    return _table(rows, ["code"] * 80 + ["prose"] * 80)


@pytest.fixture
def one_dominant_model():
    """Model 0 is best in every category; the misses are scattered per prompt,
    so the only way to beat it is to pick per prompt."""
    rng = np.random.default_rng(0)
    m0 = (rng.uniform(size=200) < 0.8).astype(float)
    m1 = np.where(m0 == 0, (rng.uniform(size=200) < 0.6), 0).astype(float)
    return _table(np.column_stack([m0, m1]), ["a", "b"] * 100)


class TestTable:
    def test_shapes_must_agree(self):
        with pytest.raises(ValueError, match="must agree"):
            CorrectnessTable(pd.DataFrame({"prompt": ["a"]}), np.zeros((1, 2)),
                             np.zeros((1, 3)), ["m0", "m1"])


class TestRouters:
    def test_a_category_that_changes_the_best_model_is_worth_everything(self, specialists):
        report = build_report(specialists, ["category"])
        assert report.router("by_category").accuracy == pytest.approx(1.0)
        assert report.router("by_category").gap_closed == pytest.approx(1.0)
        assert report.routers[0].accuracy == pytest.approx(0.5)

    def test_a_category_that_does_not_change_it_is_worth_nothing(self, one_dominant_model):
        """The RouterBench shape: one model wins everywhere, so the category
        router picks it everywhere and closes none of the gap."""
        report = build_report(one_dominant_model, ["category"])
        assert report.router("by_category").gap_closed == pytest.approx(0.0, abs=1e-9)
        assert report.routers[-1].accuracy > report.routers[0].accuracy   # oracle still wins

    def test_a_thin_category_falls_back_rather_than_guessing(self):
        """A choice fitted on three prompts is a coin toss scored as a decision."""
        table = _table([[1, 0]] * 60 + [[0, 1]] * 6, ["big"] * 60 + ["tiny"] * 6)
        r = category_router(table.subset((table.prompts.split == "train").to_numpy()),
                            table.subset((table.prompts.split == "test").to_numpy()),
                            "category", min_train=30)
        assert r.choices["tiny"] == "m0"                                   # the global best

    def test_cost_weight_trades_accuracy_for_price(self):
        """Model 0 is right slightly more often and costs ten times as much."""
        rows = [[1, 1]] * 70 + [[1, 0]] * 10
        cost = [[10.0, 1.0]] * 80
        table = _table(rows, ["x"] * 80, cost=cost)
        free = build_report(table, ["category"], cost_weight=0.0).router("by_category")
        priced = build_report(table, ["category"], cost_weight=1.0).router("by_category")
        assert free.choices["x"] == "m0" and priced.choices["x"] == "m1"
        assert priced.cost < free.cost


class TestGapDecomposition:
    def test_specialists_put_the_whole_gap_between_categories(self, specialists):
        d = gap_decomposition(specialists, "category")
        assert d["between"] == pytest.approx(1.0)
        assert d["within"] == pytest.approx(0.0)

    def test_a_dominant_model_puts_it_all_within(self, one_dominant_model):
        """The finding on RouterBench, in miniature: the opportunity is per
        prompt, so no category label can reach it."""
        d = gap_decomposition(one_dominant_model, "category")
        assert d["within"] > 0.9

    def test_the_two_parts_account_for_the_whole_gap(self, one_dominant_model):
        d = gap_decomposition(one_dominant_model, "category")
        assert d["between"] + d["within"] == pytest.approx(1.0)


class TestFamilies:
    @pytest.mark.parametrize("name, family", [
        ("mmlu-professional-law", "mmlu"),
        ("grade-school-math", "grade-school-math"),      # not `grade`
        ("arc-challenge", "arc-challenge"),              # not `arc`
        ("chinese_zodiac", "chinese"),
        ("Chinese_character_riddles", "chinese"),
        ("mtbench-math", "mtbench"),
        ("hellaswag.dev.v0", "hellaswag"),
    ])
    def test_family_names_survive_their_hyphens(self, name, family):
        assert benchmark_family(name) == family


def test_the_frontier_has_a_point_per_router_per_weight(specialists):
    f = cost_frontier(specialists, ["category"], weights=np.array([0.0, 1.0, 10.0]))
    assert set(f["router"]) == {"single_best", "by_category"}
    assert len(f) == 6


def test_the_figures_are_written(specialists, tmp_path):
    report = build_report(specialists, ["category"])
    assert plot_report(report, "category", tmp_path / "r.png").stat().st_size > 5_000
    f = cost_frontier(specialists, ["category"], weights=np.array([0.0, 1.0]))
    assert plot_frontier(f, tmp_path / "f.png").stat().st_size > 5_000
