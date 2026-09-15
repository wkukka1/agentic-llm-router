from __future__ import annotations

import json

import pytest

from evaluation.judge import client as client_module
from evaluation.judge.client import JudgeClient, _parse_verdict


def test_parse_verdict_json():
    margin, reason, ok = _parse_verdict(json.dumps({"margin": -2, "reason": "clearer"}))
    assert (margin, reason, ok) == (-2, "clearer", True)


def test_parse_verdict_regex_fallback():
    margin, _, ok = _parse_verdict('some preamble {"margin": 3, "reason": "..."} trailing')
    assert ok is True and margin == 3


def test_parse_verdict_out_of_range_fails():
    _, _, ok = _parse_verdict(json.dumps({"margin": 9}))
    assert ok is False


def test_parse_verdict_garbage_fails():
    margin, _, ok = _parse_verdict("not json at all")
    assert ok is False and margin == 0


def test_dummy_correctness_proxy_favors_more_correct_side():
    c = JudgeClient(provider="dummy", model="dummy", dummy_mode="correctness-proxy")
    v = c.judge("q", "resp a", "resp b", correct_a=0.0, correct_b=1.0)
    assert v.margin > 0   # b (candidate) is more correct -> margin favors b
    v2 = c.judge("q", "resp a", "resp b", correct_a=1.0, correct_b=0.0)
    assert v2.margin < 0
    v3 = c.judge("q", "resp a", "resp b", correct_a=1.0, correct_b=1.0)
    assert v3.margin == 0


def test_dummy_correctness_proxy_ignores_swap():
    """The dummy judge is synthetic/order-invariant -- swap must be a no-op,
    since there's no real position bias for it to have."""
    c = JudgeClient(provider="dummy", model="dummy", dummy_mode="correctness-proxy")
    v = c.judge("q", "a", "b", correct_a=0.0, correct_b=1.0)
    v_swapped = c.judge("q", "a", "b", swap=True, correct_a=0.0, correct_b=1.0)
    assert v.margin == v_swapped.margin


def test_dummy_random_is_seeded_and_bounded():
    c1 = JudgeClient(provider="dummy", model="dummy", dummy_mode="random", seed=42)
    c2 = JudgeClient(provider="dummy", model="dummy", dummy_mode="random", seed=42)
    m1 = [c1.judge("q", "a", "b").margin for _ in range(20)]
    m2 = [c2.judge("q", "a", "b").margin for _ in range(20)]
    assert m1 == m2
    assert all(-3 <= m <= 3 for m in m1)


def test_swap_negates_real_provider_margin(monkeypatch):
    c = JudgeClient(provider="dummy", model="dummy")
    # Force the "real" path even though provider="dummy", to isolate the
    # swap/negate arithmetic in JudgeClient.judge from the dummy short-circuit.
    c.provider = "fake-real"
    seen_order = []

    def fake_call_once(messages):
        seen_order.append(messages[-1]["content"])
        return json.dumps({"margin": 2, "reason": "b clearer"})

    monkeypatch.setattr(c, "_call_once", fake_call_once)
    v = c.judge("q", "RESPONSE_A_TEXT", "RESPONSE_B_TEXT", swap=False)
    assert v.margin == 2
    assert "RESPONSE_A_TEXT" in seen_order[0] and "RESPONSE_B_TEXT" in seen_order[0]

    v_swapped = c.judge("q", "RESPONSE_A_TEXT", "RESPONSE_B_TEXT", swap=True)
    # model always says "margin favors whatever is in the B slot it saw";
    # with swap=True that's the original response_a -> negated back to +2 -> -2
    # in the caller's original (a, b) frame.
    assert v_swapped.margin == -2


def test_parse_failure_tracked_in_usage(monkeypatch):
    c = JudgeClient(provider="dummy", model="dummy")
    c.provider = "fake-real"
    monkeypatch.setattr(c, "_call_once", lambda messages: "garbage, not json")
    c.judge("q", "a", "b")
    assert c.usage["n_parse_failures"] == 1
    assert c.usage["parse_failure_rate"] == 1.0


def test_parse_verdict_accepts_plus_sign_in_json():
    margin, reason, ok = _parse_verdict('{"margin": +2, "reason": "b clearer"}')
    assert (margin, reason, ok) == (2, "b clearer", True)


def test_parse_verdict_accepts_plus_sign_in_prose():
    margin, _, ok = _parse_verdict("margin: +3 because B is far more thorough")
    assert ok is True and margin == 3


def test_parse_verdict_multi_digit_is_out_of_range_not_first_digit():
    margin, _, ok = _parse_verdict("margin: 10")
    assert ok is False and margin == 0


def test_parse_verdict_conflicting_mentions_fail():
    margin, _, ok = _parse_verdict("I'd put margin = 2 at first, but final margin: -1")
    assert ok is False and margin == 0


def test_parse_verdict_agreeing_mentions_ok():
    margin, _, ok = _parse_verdict("margin = -2 ... so margin: -2")
    assert ok is True and margin == -2


def test_parse_verdict_json_object_wins_over_prose_mentions():
    raw = 'Using margin = 2 as a guide: {"margin": -1, "reason": "a clearer"}'
    assert _parse_verdict(raw) == (-1, "a clearer", True)


def test_parse_verdict_code_fence():
    raw = '```json\n{"margin": 1, "reason": "tidier"}\n```'
    assert _parse_verdict(raw) == (1, "tidier", True)


def test_parse_verdict_none_is_failure_not_crash():
    assert _parse_verdict(None) == (0, "", False)


def test_none_content_counts_as_parse_failure(monkeypatch):
    c = JudgeClient(provider="dummy", model="dummy")
    c.provider = "fake-real"
    monkeypatch.setattr(c, "_call_once", lambda messages: None)
    v = c.judge("q", "a", "b")
    assert v.parsed_ok is False and v.margin == 0
    assert c.usage["n_parse_failures"] == 1


class _FakeStatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _retry_client(monkeypatch, responses):
    """A fake-real client whose _call_once walks ``responses`` (raising exceptions)."""
    monkeypatch.setattr(client_module.time, "sleep", lambda s: None)
    c = JudgeClient(provider="dummy", model="dummy", max_retries=5)
    c.provider = "fake-real"
    calls = []

    def fake_call_once(messages):
        item = responses[min(len(calls), len(responses) - 1)]
        calls.append(item)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(c, "_call_once", fake_call_once)
    return c, calls


def test_non_transient_error_is_not_retried(monkeypatch):
    c, calls = _retry_client(monkeypatch, [_FakeStatusError(401)])
    with pytest.raises(_FakeStatusError):
        c.judge("q", "a", "b")
    assert len(calls) == 1


def test_rate_limit_is_retried_until_success(monkeypatch):
    ok = json.dumps({"margin": 1, "reason": "x"})
    c, calls = _retry_client(monkeypatch, [_FakeStatusError(429), _FakeStatusError(429), ok])
    assert c.judge("q", "a", "b").margin == 1
    assert len(calls) == 3


def test_persistent_server_error_exhausts_retries(monkeypatch):
    c, calls = _retry_client(monkeypatch, [_FakeStatusError(503)])
    with pytest.raises(RuntimeError, match="after 5 retries"):
        c.judge("q", "a", "b")
    assert len(calls) == 5
