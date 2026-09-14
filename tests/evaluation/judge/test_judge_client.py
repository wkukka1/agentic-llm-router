from __future__ import annotations

import json

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
