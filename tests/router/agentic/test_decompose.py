"""`NaiveDecomposer`: split a compound request without losing any of it."""

from __future__ import annotations

from router.agentic.decompose import NaiveDecomposer


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

    def test_a_short_task_is_still_its_own_task(self):
        """'plot it' is 7 characters and a real request; the guard counts
        words for exactly this reason."""
        assert "plot it." in NaiveDecomposer()(self.PROMPT)

    def test_the_lead_in_is_not_billed_as_a_task(self):
        parts = NaiveDecomposer()(self.PROMPT)
        assert not any(p.rstrip(".").strip() == "Do these:" for p in parts)
        assert parts[0].startswith("Do these:"), "the lead-in should introduce the first task"

    def test_punctuation_residue_still_does_not_become_a_call(self):
        """The guard's original purpose, which must survive the fix."""
        assert NaiveDecomposer()("Fix it. Now.") == ["Fix it. Now."]

    def test_a_bare_lead_in_does_not_crash(self):
        assert NaiveDecomposer()("Steps:") == ["Steps:"]
