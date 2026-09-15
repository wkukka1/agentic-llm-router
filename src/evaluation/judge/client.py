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

# Whole signed number with a trailing boundary, so "margin: 10" reads as 10
# (out of range -> failure) rather than 1.
_MARGIN_RE = re.compile(r'"?margin"?\s*[:=]\s*([+-]?\d+)(?![\d.])')
# The rubric lists "+1".."+3"; models echo the sign, which isn't valid JSON.
_PLUS_SIGN_RE = re.compile(r'("margin"\s*:\s*)\+(?=\d)')
_JSON_SPAN_RE = re.compile(r"\{.*\}", re.DOTALL)

# Statuses the provider SDKs themselves treat as retryable.
_TRANSIENT_STATUS = (408, 409, 429)


@dataclass
class JudgeVerdict:
    margin: int          # -3..+3, in the caller's (response_a, response_b) frame; positive favors b
    reason: str
    raw: str
    parsed_ok: bool


def _json_verdict(text: str) -> Optional[tuple[int, str]]:
    try:
        obj = json.loads(text)
        margin = int(obj["margin"])
        reason = str(obj.get("reason", ""))[:400]
    except Exception:
        return None
    return (margin, reason) if -3 <= margin <= 3 else None


def _parse_verdict(raw: Optional[str]) -> tuple[int, str, bool]:
    text = raw or ""   # refusals / tool-call replies carry no content
    normalized = _PLUS_SIGN_RE.sub(r"\1", text)
    candidates = [normalized]
    span = _JSON_SPAN_RE.search(normalized)   # preamble / code fence around the object
    if span and span.group(0) != normalized:
        candidates.append(span.group(0))
    for candidate in candidates:
        parsed = _json_verdict(candidate)
        if parsed is not None:
            return parsed[0], parsed[1], True
    # Regex fallback: every margin-like mention must agree, otherwise the
    # verdict is ambiguous and counted as a parse failure.
    values = {int(v) for v in _MARGIN_RE.findall(text)}
    if len(values) == 1:
        (margin,) = values
        if -3 <= margin <= 3:
            return margin, text[:400], True
    return 0, text[:400], False


def _is_transient(exc: Exception, connection_errors: tuple[type, ...]) -> bool:
    """Rate limits, timeouts, 5xx and dropped connections; not auth/4xx errors."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS or status >= 500
    return isinstance(exc, connection_errors)


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

        # SDK clients are built with max_retries=0: _call_with_retry is the
        # only retry layer, so attempts don't multiply.
        if provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
            self._connection_errors: tuple[type, ...] = (openai.APIConnectionError,)
        elif provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url, max_retries=0)
            self._connection_errors = (anthropic.APIConnectionError,)
        elif provider == "dummy":
            self._client = None
            self._connection_errors = ()
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
            except Exception as exc:
                if not _is_transient(exc, self._connection_errors):
                    raise   # auth / bad request / unknown model: retrying can't help
                last_exc = exc
                if attempt < self.max_retries - 1:
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
            return resp.choices[0].message.content or ""
        if self.provider == "anthropic":
            system = messages[0]["content"] if messages[0]["role"] == "system" else None
            user_msgs = [m for m in messages if m["role"] != "system"]
            resp = self._client.messages.create(
                model=self.model, system=system, messages=user_msgs,
                temperature=self.temperature, max_tokens=200,
            )
            return "".join(getattr(block, "text", "") for block in resp.content
                           if getattr(block, "type", None) == "text")
        raise AssertionError(f"unreachable: provider={self.provider!r}")  # pragma: no cover
