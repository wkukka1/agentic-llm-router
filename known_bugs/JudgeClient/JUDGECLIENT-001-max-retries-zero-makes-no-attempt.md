# JUDGECLIENT-001: `max_retries=0` makes zero attempts and raises

- **Status:** reproduced (`known_bugs/_repros/repro_bank_l2_and_judge_retries.py`, section 2)
- **Severity:** low
- **Class:** `evaluation.judge.client.JudgeClient._call_with_retry` ([client.py:184-196](../../src/evaluation/judge/client.py#L184-L196))
- **Found:** 2026-09-24

## What

`for attempt in range(self.max_retries):` makes `max_retries` the number of
*attempts*, not retries. With `max_retries=0` (the natural way to say "no
retries, try once") the loop body never runs and the method raises
`RuntimeError("judge call failed after 0 retries")` from `None` without ever
calling `_call_once`. The SDK clients are built with `max_retries=0` in the same
class ([client.py:116](../../src/evaluation/judge/client.py#L116)), where it does
mean "one attempt", so the same name has two meanings inside one file.

With the default `max_retries=5` the call is tried 5 times (4 retries), and the
final message says "after 5 retries", which is off by one for the same reason.

## Repro

```
c = JudgeClient("dummy", "m", max_retries=0); c._call_once = <counter>
c._call_with_retry([]) -> RuntimeError: judge call failed after 0 retries | attempts actually made: 0
```

## Suggested fix

`for attempt in range(self.max_retries + 1)` and make the sleep/raise
conditions match, or rename the parameter `max_attempts` and reject `< 1`.
