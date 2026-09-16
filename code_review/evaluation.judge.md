# Code review: evaluation.judge

**Files reviewed:** 3 (233 lines): `__init__.py`, `client.py`, `prompts.py` · **Context-only:** `scripts/data/collect_anchor_judgments.py`, `src/training/data/loaders.py`
**Summary:** 4 findings: 3 correctness, 1 design, 0 performance · 4 confirmed, 0 suspected

Downstream, failed parses become ties in the training data. That cross-package issue is filed as XA-03.

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| EJ-01 | `src/evaluation/judge/client.py:29` | correctness | medium | confirmed |
| EJ-02 | `src/evaluation/judge/client.py:40-54` | correctness | low | confirmed |
| EJ-03 | `src/evaluation/judge/client.py:166-171` | correctness | low | confirmed |
| EJ-04 | `src/evaluation/judge/client.py:149-158` | design | low | confirmed |

---

## src/evaluation/judge/client.py

### EJ-01: The margin regex rejects the `+N` form that the rubric itself uses `:29`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
29:_MARGIN_RE = re.compile(r'"?margin"?\s*[:=]\s*(-?\d)')
```

**What breaks:** The system rubric (`prompts.py:20-27`) lists `+1`, `+2`, `+3`. Models often echo that sign, as in `{"margin": +2, ...}`. That isn't valid JSON, so `json.loads` fails, and the fallback regex allows only an optional `-`, so it fails too. The verdict comes back as `margin=0` with `parsed_ok=False`. Every "B is better" answer written with a plus sign is lost, which biases the judge's measured preference toward A. The failures also become ties downstream (XA-03).
**Evidence:** Read `:29` and `:40-54` against `RUBRIC_SYSTEM`. `collect_anchor_judgments.py:256` reports `parse_failure_rate`, but nothing retries or repairs these answers.
**Direction:** Use `([+-]?\d)` in the regex, and strip a leading `+` before `json.loads`.

### EJ-02: The regex fallback accepts the first digit of any margin-like text `:40-54`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
49:    m = _MARGIN_RE.search(raw)
50:    if m:
51:        margin = int(m.group(1))
```

**What breaks:** `(-?\d)` captures one digit with no boundary check, and `search` takes the first match anywhere in the text. `margin: 10` parses as `1`. A reason that mentions `margin = 2` before the real field wins over the field. Both cases come back with `parsed_ok=True`, so they aren't even counted as failures.
**Evidence:** Read `:40-54`.
**Direction:** Anchor the pattern with `(?![\d])`, and prefer the last match or the JSON object's own field.

### EJ-03: A `None` message content crashes the collection run `:166-171`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
171:            return resp.choices[0].message.content
```

**What breaks:** OpenAI returns `content=None` for refusals and tool-call responses. `_parse_verdict(None)` catches the `TypeError` from `json.loads(None)`, but `_MARGIN_RE.search(None)` at `:49` sits outside the `try` and raises `TypeError`. The whole judgment loop aborts on one refusal.
**Evidence:** Read `:40-54` and `:166-171`.
**Direction:** Coerce with `raw = raw or ""` before parsing, and record the verdict as a parse failure.

### EJ-04: Retries apply to every exception, stacked on the SDK's own retries `:149-158`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
153:            try:
154:                return self._call_once(messages)
155:            except Exception as exc:  # rate limit / 5xx / transient -- retry
```

**What breaks:** Authentication errors, invalid model names and context-length 400s aren't transient, but each is retried `max_retries=5` times with back-off sleeps of 1, 2, 4, 8 and 16 seconds. The OpenAI and Anthropic clients already retry internally (twice by default), so one bad call can make up to 15 HTTP attempts and sleep about 31 s. With a bad API key, every item in a batch pays that before the run fails.
**Evidence:** Read `:82-87` (clients built with default `max_retries`) and `:149-158`.
**Direction:** Retry only rate-limit, timeout and 5xx errors (`openai.RateLimitError`, `APIStatusError` with status ≥ 500, and the Anthropic equivalents), and construct the SDK clients with `max_retries=0`.

---

## src/evaluation/judge/prompts.py, __init__.py

No findings.

---

## Checked and ruled out
- **`swap` being a no-op for the dummy provider:** documented. The dummy has no position bias to measure.
- **Anthropic `system=None`:** unreachable, because `build_prompt` always emits a system message.
- **`_throttle` not being thread-safe:** the client is used from a single-threaded loop.
- **The judge seeing correctness:** `correct_a` / `correct_b` reach only the dummy branch, never the real prompt.
