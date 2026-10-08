"""The final quality check (router.agentic.quality)."""

from __future__ import annotations

import math

import pytest

from router.agentic.decompose import SubTask
from router.agentic.quality import QualityChecker, parse_verdict

PASS = "VERDICT: PASS\nSCORE: 8\nWEAK_STEPS: none\nFOLLOW_UPS:\nnone\nISSUES: none"
FAIL = ("VERDICT: FAIL\nSCORE: 4\nWEAK_STEPS: 2, 9\nFOLLOW_UPS:\n"
        "[after 2] Fix the compile error in the Rust quicksort from step 2.\n"
        "Add a test for an empty array.\n"
        "ISSUES: Rust code does not compile.")


def test_clean_pass():
    v = parse_verdict(PASS, n_steps=3)
    assert v.passed and v.score == 8 and v.status == "ok"
    assert v.weak_steps == [] and v.follow_ups == []


def test_fail_with_weak_steps_and_follow_ups():
    v = parse_verdict(FAIL, n_steps=3)
    assert not v.passed and v.status == "ok" and v.score == 4
    assert v.weak_steps == [2]                                   # 9 is not a reviewed step
    assert v.follow_ups == [
        SubTask("Fix the compile error in the Rust quicksort from step 2.", (2,)),
        SubTask("Add a test for an empty array."),
    ]
    assert v.issues == "Rust code does not compile."


def test_weak_step_zero_is_ignored():
    assert parse_verdict(FAIL.replace("2, 9", "0"), n_steps=3).weak_steps == []


def test_pass_below_pass_score_is_a_fail():
    v = parse_verdict(PASS.replace("SCORE: 8", "SCORE: 5"), pass_score=7)
    assert not v.passed and v.status == "ok"


def test_unparseable_reply_is_never_a_pass():
    v = parse_verdict("Looks great to me!")
    assert v.status == "unparsed" and not v.passed and math.isnan(v.score)


class _Chat:
    def __init__(self, reply):
        self.reply, self.requests = reply, []

    def invoke(self, text):
        self.requests.append(text)
        return self.reply


def test_checker_sends_steps_and_answer_and_truncates_long_text():
    chat = _Chat(PASS)
    checker = QualityChecker(chat, prompt="{original}|{steps}|{answer}",
                             max_step_chars=5, max_answer_chars=4)
    assert checker("the request", [("task one", "0123456789")], "abcdefgh").passed
    assert chat.requests == [
        "the request|Step 1: task one\nAnswer: 01234 …[truncated]|abcd …[truncated]"]


def test_checker_call_failure_is_an_error_verdict():
    class Broken:
        def invoke(self, text):
            raise RuntimeError("down")

    with pytest.warns(UserWarning, match="QualityChecker call failed"):
        v = QualityChecker(Broken(), prompt="{original}{steps}{answer}")("r", [], "a")
    assert v.status == "error" and not v.passed


def test_checker_counts_against_the_call_budget():
    calls = []
    checker = QualityChecker(_Chat(PASS), prompt="{original}{steps}{answer}")
    checker.on_call = lambda: calls.append(1)
    checker("r", [("t", "a")], "a")
    assert calls == [1]


def test_shipped_verify_prompt_formats():
    from router.agentic.schemas import default_prompts
    text = default_prompts()["verify"].format(original="o", steps="s", answer="a")
    assert "VERDICT:" in text and "WEAK_STEPS:" in text and "FOLLOW_UPS:" in text


def test_empty_follow_ups_does_not_swallow_the_next_key_line():
    v = parse_verdict("VERDICT: FAIL\nSCORE: 4\nWEAK_STEPS: 1\nFOLLOW_UPS:\nISSUES: bad code", n_steps=3)
    assert v.follow_ups == [] and v.issues == "bad code" and v.weak_steps == [1]


def test_empty_weak_steps_does_not_read_digits_from_the_next_line():
    v = parse_verdict("VERDICT: FAIL\nSCORE: 4\nWEAK_STEPS:\nFOLLOW_UPS: none\nISSUES: step 2 is wrong", n_steps=3)
    assert v.weak_steps == []


def test_follow_up_on_the_same_line_as_the_key_is_kept():
    v = parse_verdict("VERDICT: FAIL\nSCORE: 4\nWEAK_STEPS: none\nFOLLOW_UPS: Add a test for an empty array.\nISSUES: none", n_steps=3)
    assert [t.text for t in v.follow_ups] == ["Add a test for an empty array."]


def test_long_original_request_carries_the_truncated_marker():
    chat = _Chat(PASS)
    checker = QualityChecker(chat, prompt="{original}|{steps}|{answer}")
    checker("x" * 2500, [("t", "a")], "ans")
    assert chat.requests[0].startswith("x" * 2000 + " …[truncated]|")
