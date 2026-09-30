"""`NaiveDecomposer`: split a compound request without losing any of it."""

from __future__ import annotations

import pytest

from router.agentic.decompose import LLMDecomposer, NaiveDecomposer


class TestNothingIsSilentlyDropped:
    """Found by running the agentic loop end to end, not by a unit test.

    "Do these: 1. read the file 2. compute the mean 3. plot it" came back as
    two answered tasks plus a paid call for the lead-in, and the third task was
    gone. Nothing raised. A decomposer that loses part of the request is worse
    than one that declines to split, because the caller cannot tell.
    """

    PROMPT = "Do these:\n1. read the file\n2. compute the mean\n3. plot it"

    def test_every_requested_task_survives(self):
        parts = NaiveDecomposer()(self.PROMPT)
        joined = " ".join(parts).lower()
        for task in ("read the file", "compute the mean", "plot it"):
            assert task in joined, f"{task!r} was dropped"

    def test_a_short_task_is_merged_not_dropped(self):
        """'plot it' is 7 characters, under `min_part_chars`, and a real request.

        It is carried into its neighbour rather than kept as a part of its own.
        That is deliberate (NAIVEDECOMPOSER-001): the guard counts characters,
        so a 2-word task joins the one before it. Splitting on a word count
        instead would make each of these standalone, at the cost of a more
        split-happy decomposer and one paid call per fragment. What matters
        here, and what this pins, is that the text survives somewhere.
        """
        parts = NaiveDecomposer()(self.PROMPT)
        assert "plot it" in " ".join(parts).lower()

    def test_the_lead_in_is_not_billed_as_a_task(self):
        parts = NaiveDecomposer()(self.PROMPT)
        assert not any(p.rstrip(".").strip() == "Do these:" for p in parts)
        assert parts[0].startswith("Do these:"), "the lead-in should introduce the first task"

    def test_punctuation_residue_still_does_not_become_a_call(self):
        """The guard's original purpose, which must survive the fix."""
        assert NaiveDecomposer()("Fix it. Now.") == ["Fix it. Now."]

    def test_a_bare_lead_in_does_not_crash(self):
        assert NaiveDecomposer()("Steps:") == ["Steps:"]


class _Stub:
    """A chat model that returns a canned string, and records what it was asked."""

    def __init__(self, text: str | Exception):
        self.text, self.seen = text, []

    def invoke(self, prompt):
        self.seen.append(prompt)
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


class TestLLMDecomposer:
    """The semantic decomposer -- the level-2 path.

    `NaiveDecomposer` only fires when the caller has already written the seams
    in ("do X, then Y"). Implicitly compound work -- "write a report on the
    competitive landscape" -- is one sentence with no markers, and needs a model
    to see that it is research, then synthesis, then writing. This class is that
    hook, and until now it had no test at all.
    """

    def test_it_splits_an_implicitly_compound_request(self):
        chat = _Stub("research the battery market\nsynthesise the findings\nwrite the report")
        assert LLMDecomposer(chat)("Write a report on EV batteries.") == [
            "research the battery market", "synthesise the findings", "write the report"]

    def test_the_request_reaches_the_model(self):
        chat = _Stub("a task\nb task")
        LLMDecomposer(chat)("Write a report on EV batteries.")
        assert "EV batteries" in chat.seen[0]

    def test_a_single_task_comes_back_unchanged(self):
        assert LLMDecomposer(_Stub("just the one thing"))("x") == ["just the one thing"]

    def test_numbering_the_model_adds_anyway_is_stripped(self):
        assert LLMDecomposer(_Stub("1. alpha task\n2) beta task\n- gamma task"))("x") == [
            "alpha task", "beta task", "gamma task"]

    def test_a_preamble_is_not_a_task(self):
        """'Here are the sub-tasks:' is commentary, and paying a model to answer
        it is pure waste."""
        out = LLMDecomposer(_Stub("Here are the sub-tasks:\nfirst real task\nsecond real task"))("x")
        assert out == ["first real task", "second real task"]

    def test_an_empty_answer_declines_rather_than_returning_nothing(self):
        assert LLMDecomposer(_Stub("   \n  \n"))("original prompt") == ["original prompt"]

    class TestItNeverLosesWorkSilently:
        """The bug this class exists to prevent: ask for twelve things, get
        charged for eight, and be handed an answer that reads complete."""

        def test_the_overflow_is_folded_in_rather_than_cut_off(self):
            """Twelve sub-tasks against max_parts=8 must still contain all twelve;
            the cap bounds the number of CALLS, not the amount of work carried."""
            chat = _Stub("\n".join(f"step number {i}" for i in range(1, 13)))
            out = LLMDecomposer(chat, max_parts=8)("x")
            assert len(out) == 8
            joined = " ".join(out)
            for i in range(1, 13):
                assert f"step number {i}" in joined, f"step {i} was lost"

        def test_within_the_cap_nothing_is_merged(self):
            assert LLMDecomposer(_Stub("a task\nb task"), max_parts=8)("x") == [
                "a task", "b task"]

        def test_the_naive_decomposer_folds_its_overflow_too(self):
            out = NaiveDecomposer(max_parts=3)("do a. Then do b. Then do c. Then do d. Then do e.")
            assert len(out) == 3
            assert "do e" in " ".join(out).lower(), "the tail was cut off"

    class TestItFailsSafe:
        def test_a_provider_error_falls_back_instead_of_raising(self):
            chat = _Stub(RuntimeError("provider exploded"))
            with pytest.warns(UserWarning, match="using the fallback decomposer"):
                assert LLMDecomposer(chat)("Explain quicksort. Then write it.") == [
                    "Explain quicksort.", "Then write it."]

        def test_an_over_long_prompt_falls_back_rather_than_truncating(self):
            """Truncating the request would drop its later sub-tasks before the
            model ever saw them."""
            chat = _Stub("should not be used")
            with pytest.warns(UserWarning, match="using the fallback decomposer"):
                LLMDecomposer(chat, max_prompt_chars=50)("word " * 100)
            assert chat.seen == [], "the model must not be called at all"
