"""Pairwise LLM-judge client (docs/anchor_judge.md).

Three providers: "openai", "anthropic" (real API calls -- both packages are
already an optional extras group, ``pyproject.toml``'s ``labeling`` group),
and "dummy" (no network, deterministic, for tests and ``--dry-run``).

``swap`` sends (response_b, response_a) to the underlying model instead of
(response_a, response_b), then negates the returned margin so it's still
reported in the caller's original (a, b) frame -- this is the position-swap
noise-floor mechanism from the anchor-judge plan, built in here rather than
bolted on by a caller. The judge is never shown ``correct_a``/``correct_b``;
those exist only for the ``dummy`` provider's ``correctness-proxy`` self-test
mode (see docs/anchor_judge.md's validity-gate verification plan) and are
dropped before any real API call is built.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .prompts import build_prompt

_MARGIN_RE = re.compile(r'"?margin"?\s*[:=]\s*(-?\d)')


@dataclass
class JudgeVerdict:
    margin: int          # -3..+3, in the caller's (response_a, response_b) frame; positive favors b
    reason: str
    raw: str
    parsed_ok: bool


def _parse_verdict(raw: str) -> tuple[int, str, bool]:
    try:
        obj = json.loads(raw)
        margin = int(obj["margin"])
        reason = str(obj.get("reason", ""))[:400]
        if -3 <= margin <= 3:
            return margin, reason, True
    except Exception:
        pass
    m = _MARGIN_RE.search(raw)
    if m:
        margin = int(m.group(1))
        if -3 <= margin <= 3:
            return margin, raw[:400], True
    return 0, raw[:400], False


class JudgeClient:
    def __init__(
        self,
        provider: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 5,
        requests_per_minute: int = 60,
        dummy_mode: str = "correctness-proxy",   # {"correctness-proxy", "random"}
        seed: int = 0,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._min_interval = 60.0 / max(requests_per_minute, 1)
        self._last_call = 0.0
        self.dummy_mode = dummy_mode
        self._rng = np.random.default_rng(seed)
        self.n_calls = 0
        self.n_parse_failures = 0

        if provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        elif provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        elif provider == "dummy":
            self._client = None
        else:
            raise ValueError(f"unknown judge provider: {provider!r}")

    @property
    def usage(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_parse_failures": self.n_parse_failures,
            "parse_failure_rate": (self.n_parse_failures / self.n_calls) if self.n_calls else 0.0,
        }

    def judge(
        self,
        query: str,
        response_a: str,
        response_b: str,
        *,
        swap: bool = False,
        correct_a: Optional[float] = None,
        correct_b: Optional[float] = None,
    ) -> JudgeVerdict:
        self.n_calls += 1

        if self.provider == "dummy":
            # Synthetic and order-invariant by construction -- swap is a no-op
            # (there is no real "position bias" to measure the noise floor of).
            verdict = self._dummy_judge(correct_a, correct_b)
        else:
            sent_a, sent_b = (response_b, response_a) if swap else (response_a, response_b)
            verdict = self._real_judge(query, sent_a, sent_b)
            if swap:
                verdict = JudgeVerdict(margin=-verdict.margin, reason=verdict.reason,
                                        raw=verdict.raw, parsed_ok=verdict.parsed_ok)

        if not verdict.parsed_ok:
            self.n_parse_failures += 1
        return verdict

    # -- dummy -----------------------------------------------------------------
    def _dummy_judge(self, correct_a: Optional[float], correct_b: Optional[float]) -> JudgeVerdict:
        if self.dummy_mode == "random":
            margin = int(self._rng.integers(-3, 4))
        elif self.dummy_mode == "correctness-proxy":
            if correct_a is None or correct_b is None or correct_a == correct_b:
                margin = 0
            else:
                margin = 3 if correct_b > correct_a else -3
        else:
            raise ValueError(f"unknown dummy_mode: {self.dummy_mode!r}")
        return JudgeVerdict(margin=margin, reason=f"dummy:{self.dummy_mode}",
                             raw=json.dumps({"margin": margin}), parsed_ok=True)

    # -- real providers ----------------------------------------------------------
    def _real_judge(self, query: str, response_a: str, response_b: str) -> JudgeVerdict:
        messages = build_prompt(query, response_a, response_b)
        raw = self._call_with_retry(messages)
        margin, reason, ok = _parse_verdict(raw)
        return JudgeVerdict(margin=margin, reason=reason, raw=raw, parsed_ok=ok)

    def _call_with_retry(self, messages: list[dict]) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                return self._call_once(messages)
            except Exception as exc:  # rate limit / 5xx / transient -- retry
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"judge call failed after {self.max_retries} retries") from last_exc

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def _call_once(self, messages: list[dict]) -> str:
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature,
            )
            return resp.choices[0].message.content
        if self.provider == "anthropic":
            system = messages[0]["content"] if messages[0]["role"] == "system" else None
            user_msgs = [m for m in messages if m["role"] != "system"]
            resp = self._client.messages.create(
                model=self.model, system=system, messages=user_msgs,
                temperature=self.temperature, max_tokens=200,
            )
            return resp.content[0].text
        raise AssertionError(f"unreachable: provider={self.provider!r}")  # pragma: no cover
