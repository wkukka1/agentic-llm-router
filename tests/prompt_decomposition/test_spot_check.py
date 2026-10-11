"""The spot check: where predictions land, and scoring a hand-marked sample.

No model in these tests -- the heads are stubbed -- because what is under test
is the bookkeeping: that concentration is measured the way it is described,
that a half-marked sheet scores only what was marked, and that the interval is
honest at the small sample sizes a spot check always has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from evaluation.prompt_decomposition.spot_check import (
    REVIEW_COLUMNS,
    distribution_report,
    plot_distribution,
    predict_corpus,
    review_sheet,
    score_review,
    wilson_interval,
)


@dataclass
class _Domain:
    domain: str
    confidence: float
    distribution: dict = field(default_factory=dict)
    shortlist: list = field(default_factory=list)


@dataclass
class _Task:
    task: str
    confidence: float


class _Heads:
    """Stands in for both heads: code prompts go to software, the rest to other."""

    def predict_batch(self, prompts):
        if not hasattr(self, "task_mode"):
            return [_Domain("software_tech" if "code" in p else "meta_other", 0.8,
                            {"software_tech": 0.8, "meta_other": 0.2}, ["software_tech"])
                    for p in prompts]
        return [_Task("answer", 0.7) for _ in prompts]


def _preds(domains, confs=None, languages=None):
    n = len(domains)
    return pd.DataFrame({
        "arena_id": [f"id{i}" for i in range(n)], "prompt": [f"prompt {i}" for i in range(n)],
        "language": languages or ["en"] * n, "domain": domains,
        "domain_conf": confs or [0.9] * n, "domain_shortlist": domains,
        "task": ["answer"] * n, "task_conf": [0.8] * n,
    })


class TestPredictCorpus:
    def test_batches_are_stitched_back_in_row_order(self):
        rows = pd.DataFrame({"arena_id": list("abcde"),
                             "prompt": ["code a", "hi", "code b", "yo", "code c"]})
        task = _Heads()
        task.task_mode = True
        out = predict_corpus(rows, _Heads(), task, batch_size=2)
        assert list(out["arena_id"]) == list("abcde")
        assert list(out["domain"]) == ["software_tech", "meta_other", "software_tech",
                                       "meta_other", "software_tech"]
        assert out.loc[0, "domain_2nd"] == "meta_other"


class TestDistribution:
    def test_even_traffic_counts_every_domain(self):
        r = distribution_report(_preds(["a", "b", "c", "d"] * 25))
        assert r.effective_domains == pytest.approx(4.0)
        assert r.top2_share == pytest.approx(0.5)

    def test_traffic_in_two_domains_behaves_like_two(self):
        """Will's hypothesis made measurable: eight labels in use, but if two
        hold nearly everything the feature has almost nothing to vary on."""
        domains = ["a"] * 48 + ["b"] * 48 + list("cdefgh") * 1
        r = distribution_report(_preds(domains[:100]))
        assert r.effective_domains < 2.5
        assert r.top2_share > 0.9

    def test_low_confidence_share_uses_the_documented_cut(self):
        r = distribution_report(_preds(["a"] * 4, confs=[0.2, 0.4, 0.6, 0.9]))
        assert r.low_conf_share == pytest.approx(0.5)

    def test_language_groups_are_reported_separately(self):
        r = distribution_report(_preds(["a", "a", "b", "b"],
                                       languages=["en", "en", "zh", "pl"]))
        assert r.by_language["en"] == {"a": 1.0}
        assert r.by_language["non-en"] == {"b": 1.0}

    def test_the_training_baseline_travels_with_the_report(self):
        r = distribution_report(_preds(["a", "b"]), training_labels=pd.Series(["a", "a", "b"]))
        assert r.training_domain_share["a"] == pytest.approx(2 / 3)


class TestReviewSheet:
    def test_it_is_a_random_sample_with_blank_marking_columns(self, tmp_path):
        path = review_sheet(_preds(list("abcdefghij") * 10), tmp_path / "s.csv", n=12)
        sheet = pd.read_csv(path, keep_default_na=False)
        assert len(sheet) == 12
        for col in REVIEW_COLUMNS:
            assert col in sheet and (sheet[col] == "").all()

    def test_the_same_seed_draws_the_same_rows(self, tmp_path):
        """A sheet two people mark independently must be the same sheet."""
        preds = _preds(list("abcdefghij") * 10)
        a = pd.read_csv(review_sheet(preds, tmp_path / "a.csv", n=10))
        b = pd.read_csv(review_sheet(preds, tmp_path / "b.csv", n=10))
        assert list(a["arena_id"]) == list(b["arena_id"])


class TestScoring:
    def test_unmarked_rows_are_left_out_not_counted_wrong(self):
        sheet = pd.DataFrame({"domain_correct": ["y", "n", "", "yes", np.nan]})
        s = score_review(sheet)["domain_correct"]
        assert s["n"] == 3 and s["correct"] == 2

    def test_marks_are_read_leniently(self):
        sheet = pd.DataFrame({"task_correct": ["Y", " yes ", "1", "TRUE", "No", "0"]})
        assert score_review(sheet)["task_correct"]["correct"] == 4

    def test_the_interval_is_wide_at_spot_check_sizes(self):
        """The reason it is reported: 50 of 60 is not '83%', it is 71-90%."""
        low, high = wilson_interval(50, 60)
        assert low < 0.75 and high > 0.88
        assert low < 50 / 60 < high

    def test_the_interval_stays_inside_zero_and_one(self):
        assert wilson_interval(0, 10)[0] >= 0.0
        assert wilson_interval(10, 10)[1] <= 1.0


def test_the_distribution_plot_writes_a_figure(tmp_path):
    r = distribution_report(_preds(list("abcd") * 5), training_labels=pd.Series(list("abcd")))
    path = plot_distribution(r, tmp_path / "d.png")
    assert path.exists() and path.stat().st_size > 5_000


class TestBenchmarkGold:
    """The objective half: score only where the benchmark fixes the answer."""

    @pytest.mark.parametrize("name, domain", [
        ("mmlu-professional-law.val.1", "business_law"),
        ("grade-school-math", "science_math"),
        ("mbpp", "software_tech"),
        ("mmlu-high-school-us-history", "culture"),
        ("mmlu-formal-logic", "science_math"),          # "logic" is in the definition
    ])
    def test_unambiguous_subsets_map_by_the_taxonomys_own_definitions(self, name, domain):
        from evaluation.prompt_decomposition.spot_check import expected_domain
        assert expected_domain(name) == domain

    @pytest.mark.parametrize("name", ["mmlu-moral-scenarios", "hellaswag",
                                      "mmlu-professional-psychology", "chinese_idioms"])
    def test_ambiguous_subsets_are_left_out_not_guessed(self, name):
        from evaluation.prompt_decomposition.spot_check import expected_domain
        assert expected_domain(name) is None

    def test_only_scorable_rows_enter_the_score(self):
        from evaluation.prompt_decomposition.spot_check import score_against_benchmark
        preds = pd.DataFrame({
            "eval_name": ["mbpp", "mbpp", "hellaswag", "grade-school-math"],
            "domain": ["software_tech", "science_math", "culture", "science_math"],
            "domain_shortlist": ["software_tech", "science_math|software_tech",
                                 "culture", "science_math"],
        })
        s = score_against_benchmark(preds)
        assert s["n"] == 3                                   # hellaswag excluded
        assert s["accuracy"] == pytest.approx(2 / 3)
        assert s["in_shortlist"] == pytest.approx(1.0)       # the miss had it second
