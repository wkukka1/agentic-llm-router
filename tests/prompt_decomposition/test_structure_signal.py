"""`StructureClassifier`, and the triage/decompose chain it feeds.

The bug that prompted this: "Summarise X, then implement Y, then analyse Z"
answered single-shot, because the hand-written marker list in
`router.agentic.triage` had `" and then "` but not `", then"`. Two places had
to be fixed -- triage, which decides *whether* to decompose, and
`router.agentic.decompose`, which does the splitting -- so the tests pin both
ends and the join between them.

Note what is *not* tested: whether the classifier is accurate on some corpus.
It is a conservative rule over named evidence, and the contract is that each
reason it gives is a thing a reader could point at in the prompt.
"""

from __future__ import annotations

import pytest

from decompose.classifiers.base import ClassificationInput
from decompose.classifiers.prompt_heads import StructureClassifier
from decompose.decomposer import PromptDecomposer
from router.agentic.decompose import NaiveDecomposer
from router.agentic.triage import HeuristicTriage

COMPOUND = "Summarise transformers, then implement attention in NumPy and analyse its complexity."


def _signals(prompt: str, **kw) -> dict:
    out = StructureClassifier(**kw).classify(ClassificationInput(prompt=prompt))
    return {s.name: s for s in out}


class TestTheCommaForm:
    def test_the_prompt_that_routed_single_shot_is_now_multi_part(self):
        s = _signals(COMPOUND)["multi_part"]
        assert s.value is True
        assert "', then'" in s.metadata["reasons"][0]

    @pytest.mark.parametrize("marker", [", then ", " and then ", "; next ", ". After that ",
                                        ", followed by ", ". Finally, "])
    def test_each_sequence_marker_fires(self, marker):
        assert _signals(f"Write the parser{marker}write the tests.")["multi_part"].value is True

    def test_a_marker_alone_is_enough_without_two_known_verbs(self):
        """Requiring two verbs from the list made this narrower than the list
        it replaced, and the verb list is the brittleness being removed."""
        s = _signals("Fix the flaky spec, then push it.")["multi_part"]
        assert s.value is True
        assert s.metadata["distinct_task_verbs"] == []   # neither verb is in the list


class TestSingleRequestsStayWhole:
    @pytest.mark.parametrize("prompt", [
        "What is 2+2?",
        "Explain how quicksort works.",
        "Write a Python function that reverses a linked list.",
        "Compare Rust and Go for systems programming.",   # 'compare' is one deliverable
    ])
    def test_a_plain_request_is_not_multi_part(self, prompt):
        assert _signals(prompt)["multi_part"].value is False

    def test_the_word_then_inside_a_conditional_is_not_a_sequence(self):
        """`if ... then` is not two deliverables. The marker must follow a
        clause boundary, which is why the regex is anchored."""
        assert _signals("If x is even then return x // 2.")["multi_part"].value is False


class TestTheOtherEvidence:
    def test_two_enumerated_items_are_enough(self):
        s = _signals("Do these:\n1. read the file\n2. print the mean")["multi_part"]
        assert s.value is True and "2 enumerated items" in s.metadata["reasons"]

    def test_one_bullet_is_not(self):
        assert _signals("Note:\n- be brief")["multi_part"].value is False

    def test_three_questions_are_enough(self):
        assert _signals("What? Why? How?")["multi_part"].value is True

    def test_a_long_prompt_is_multi_part_by_length_alone(self):
        s = _signals("word " * 200)["multi_part"]
        assert s.value is True and s.metadata["reasons"] == ["1000 characters"]

    def test_the_thresholds_are_adjustable(self):
        assert _signals("A? B?", min_questions=2)["multi_part"].value is True
        assert _signals("A? B?")["multi_part"].value is False


class TestThePartCount:
    def test_a_single_request_counts_one(self):
        assert _signals("Explain quicksort.")["part_count"].value == 1

    def test_enumerated_items_are_counted(self):
        assert _signals("1. a\n2. b\n3. c")["part_count"].value == 3

    def test_it_is_a_lower_bound_not_an_exact_split(self):
        """Three verbs are named in COMPOUND; the count says 'at least three',
        and the decomposer is free to disagree -- see the chain test below."""
        assert _signals(COMPOUND)["part_count"].value >= 2


class TestTheContract:
    def test_both_signals_carry_provenance(self):
        for s in _signals(COMPOUND).values():
            assert s.produced_by == "structure:1"

    def test_it_merges_into_the_decomposer_without_collision(self):
        signals = PromptDecomposer(classifiers=[StructureClassifier()]).extract_signals(
            ClassificationInput(prompt=COMPOUND))
        assert signals.get("multi_part").value is True
        assert signals.get("part_count").value >= 2

    def test_evidence_is_readable_on_its_own(self):
        """Triage puts this in the reason string a user sees."""
        e = StructureClassifier().evidence(COMPOUND)
        assert e["sequence_markers"] == [", then"]
        assert {"summarise", "implement", "analyse"} <= set(e["distinct_task_verbs"])


class TestTheChainItFeeds:
    def test_triage_prefers_the_signal_over_the_marker_list(self):
        """Both halves of the original bug in one assertion: the marker list
        misses this prompt, the signal catches it, and triage takes the signal."""
        flagged, why = HeuristicTriage(structure=StructureClassifier())._multipart(COMPOUND)
        assert flagged is True and "', then'" in why
        assert HeuristicTriage()._multipart(COMPOUND)[0] is False   # unfixed fallback

    def test_the_marker_list_still_answers_when_no_classifier_is_given(self):
        assert HeuristicTriage()._multipart("Do A and then do B.")[0] is True

    def test_the_agentic_router_wires_it_in_by_default(self):
        from router.agentic.router import AgenticRouter

        # the router itself is never touched during construction
        triage = AgenticRouter(router=None).triage
        assert isinstance(triage.structure, StructureClassifier)

    def test_the_decomposer_actually_splits_on_the_comma_form(self):
        """Triage flagging it is useless if the splitter then returns one part:
        that was the second half of the bug -- 'nothing to decompose'."""
        assert len(NaiveDecomposer()(COMPOUND)) == 2

    def test_an_abbreviation_is_not_a_split_point(self):
        assert len(NaiveDecomposer()("Use numpy, e.g. arrays, to compute the mean.")) == 1
