"""Surface signals: cheap features read straight off the prompt string.

These have no training data and no accuracy number, so the tests are the only
thing standing between a broken regex and a silently-zero feature column. Each
detector gets a prompt it must fire on and one it must not -- the second half
matters more, since a feature that fires on everything is worse than no feature.
"""

from __future__ import annotations

import numpy as np
import pytest

from prompt_decomposition.signals import SURFACE_FEATURES, extract, extract_many
from prompt_decomposition.signals.surface import SurfaceSignals


def flag(prompt: str, name: str) -> float:
    return extract(prompt)[name]


class TestContract:
    def test_names_and_values_agree_on_length(self):
        """The pair that must never drift: a mislabelled column is worse than a
        missing one, because nothing errors."""
        assert len(extract("hello").values) == len(SURFACE_FEATURES)

    def test_as_dict_round_trips_every_feature(self):
        d = extract("write me a poem")
        assert set(d.as_dict()) == set(SURFACE_FEATURES)
        assert d["log_words"] == pytest.approx(np.log1p(4))

    def test_wrong_width_is_rejected_rather_than_stored(self):
        with pytest.raises(ValueError, match="expected"):
            SurfaceSignals(np.zeros(3))

    def test_extract_many_returns_matrix_and_names_together(self):
        X, names = extract_many(["a b c", "d e f"])
        assert X.shape == (2, len(names)) == (2, len(SURFACE_FEATURES))

    def test_empty_batch_keeps_its_width(self):
        X, names = extract_many([])
        assert X.shape == (0, len(names))

    @pytest.mark.parametrize("prompt", ["", "   ", "\n\n"])
    def test_degenerate_prompts_do_not_divide_by_zero(self, prompt):
        v = extract(prompt).values
        assert np.all(np.isfinite(v))


class TestCodeDetection:
    def test_fenced_block_fires(self):
        assert flag("fix this:\n```\nx = 1\n```", "has_code_fence") == 1.0

    def test_call_syntax_fires_without_a_fence(self):
        """Real prompts paste code without fencing it far more often than with."""
        assert flag("why does foo(bar) return None", "has_code_shape") == 1.0

    def test_prose_about_software_does_not_fire(self):
        """The trap: `software_tech` prompts are mostly ordinary English and
        must not all look like code."""
        p = "what is the difference between a compiler and an interpreter"
        assert flag(p, "has_code_shape") == 0.0
        assert flag(p, "has_code_fence") == 0.0


class TestOutputShapeSignals:
    """What the user asked the *answer* to look like -- the part a cost model
    can read before generating anything."""

    @pytest.mark.parametrize("prompt", [
        "list the top 5 cities as json",
        "answer in a table",
        "one word answer please",
        "give me bullet points",
    ])
    def test_format_requests_fire(self, prompt):
        assert flag(prompt, "asks_format") == 1.0

    @pytest.mark.parametrize("prompt", [
        "in 50 words, explain entropy",
        "keep it brief",
        "summarise in at most 3 sentences",
        "tl;dr please",
    ])
    def test_length_limits_fire(self, prompt):
        assert flag(prompt, "asks_length_limit") == 1.0

    def test_an_open_request_asks_for_neither(self):
        p = "tell me about the history of the roman empire"
        assert flag(p, "asks_format") == 0.0
        assert flag(p, "asks_length_limit") == 0.0


class TestGatingSignals:
    """Signals that decide whether the cheap path can serve the request at all."""

    def test_recency_words_fire(self):
        assert flag("what happened in the news today", "needs_recency") == 1.0

    def test_timeless_question_does_not(self):
        assert flag("who wrote the iliad", "needs_recency") == 0.0

    def test_urls_are_counted_not_just_flagged(self):
        two = extract("compare https://a.com and https://b.com")
        assert two["has_url"] == 1.0
        assert two["n_urls"] == pytest.approx(np.log1p(2))

    def test_maths_notation_fires_on_latex_and_on_bare_arithmetic(self):
        assert flag(r"solve $$x^2 + 1 = 0$$", "has_math") == 1.0
        assert flag("what is 12 * 14", "has_math") == 1.0


class TestStructureSignals:
    def test_enumerated_subrequests_are_counted(self):
        p = "do these:\n1. summarise it\n2. translate it\n3. check the grammar"
        s = extract(p)
        assert s["has_enumeration"] == 1.0
        assert s["n_enumerated_items"] == pytest.approx(np.log1p(3))

    def test_a_single_request_has_no_enumeration(self):
        assert flag("summarise this article for me", "has_enumeration") == 0.0

    def test_imperative_and_interrogative_openings_are_distinguished(self):
        """'write me X' and 'what is X' route differently -- one produces an
        artifact, the other retrieves a fact."""
        assert flag("write me an email to my landlord", "starts_imperative") == 1.0
        assert flag("what is the capital of peru", "starts_imperative") == 0.0
        assert flag("what is the capital of peru?", "ends_with_question") == 1.0

    def test_question_count_separates_one_from_many(self):
        one = extract("what is x?")["n_questions"]
        three = extract("what is x? why does it matter? how do I use it?")["n_questions"]
        assert three > one


class TestScaling:
    def test_length_is_log_compressed(self):
        """Raw length spans three orders of magnitude in real traffic; a linear
        feature over that lets a few very long prompts dominate any fit."""
        short = extract("hi")["log_chars"]
        long = extract("word " * 2000)["log_chars"]
        assert long < 12 * short

    def test_ratios_stay_in_the_unit_interval(self):
        s = extract("WHAT IS 2+2?!! ¿cómo estás?")
        for name in ("uppercase_ratio", "digit_ratio", "punct_ratio", "non_ascii_ratio"):
            assert 0.0 <= s[name] <= 1.0

    def test_non_ascii_fires_on_other_scripts(self):
        assert extract("这句话翻译一下")["non_ascii_ratio"] > 0.5
        assert extract("translate this line")["non_ascii_ratio"] == 0.0
