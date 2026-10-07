# LangChainAdapter correctness fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three open, `med`-severity `known_bugs/` defects, all in `src/router/llm/adapters/langchain_adapter.py`: `guess_provider` matches a provider hint mid-token (`gpt4all-13b-snoozy` wrongly guessed `"openai"`), an explicit `api_key` never reaches `init_chat_model`, and a half-priced profile silently prices the unset side at `$0`.

**Architecture:** Three independent, small patches to one file (`langchain_adapter.py`) and its one test file (`tests/router/llm/adapters/test_langchain_adapter.py`). No new files, no shared state between the three fixes — order doesn't matter, do them in any order.

**Tech Stack:** Python 3.12, pytest, `langchain`/`langchain_core` (already an installed dependency; no `importorskip` needed for `langchain.chat_models` in this repo's venv, but keep the existing test file's style of not requiring network access — everything here stays offline via a stub chat model or a monkeypatched `init_chat_model`).

**Spec:** `fixed_bugs/_module_level/router.llm.adapters/MOD-001-guess-provider-missing-end-boundary/bug.md`, `fixed_bugs/ProviderAdapter/LangChainAdapter/LANGCHAINADAPTER-001-api-key-not-forwarded/bug.md`, `fixed_bugs/ProviderAdapter/LangChainAdapter/LANGCHAINADAPTER-002-partial-price-defaults-to-zero/bug.md`.

## Global Constraints

- `guess_provider`'s existing passing behavior must not change: `guess_provider("o1-mini") == "openai"`, `guess_provider("yolo1") is None`, `guess_provider("gpt-4o") == "openai"`, `guess_provider("claude-3-5-sonnet") == "anthropic"`, `guess_provider("some-random-model") is None` (from `tests/router/test_review_fixes.py::test_guess_provider_matches_whole_prefixes_only` and `tests/router/agentic/test_agentic_router.py::test_guess_provider`) — only a hint matching *inside* a longer alnum run (no boundary on either side) should newly return `None`.
- `LangChainAdapter`'s three existing tests in `tests/router/llm/adapters/test_langchain_adapter.py` (`test_complete_forwards_generation_kwargs_to_invoke`, `test_stream_forwards_generation_kwargs_to_stream`, `test_complete_with_no_generation_kwargs_still_works`) construct the adapter with `chat_model=chat` directly (bypassing `_model()`'s lazy `init_chat_model` path entirely) — none of them touch `api_key`/pricing, so they must keep passing unchanged.
- Match `LLMClient.price`'s existing fix for the sibling bug LLMCLIENT-001 (`src/router/llm/client.py`, already fixed 2026-09-23): a price is only *required* for a side whose token count is actually nonzero — a `0`/`None` token count on one side doesn't force that side's price to be known.

## Review Focus

- A model id containing a hint as a true prefix/token (`"gpt-4o"`) must still match after the end-boundary fix — the risk with any regex boundary change is over-tightening and breaking the already-passing cases. (Task 1)
- `api_key=None` (the default, the overwhelmingly common case) must not add a spurious `api_key=None` kwarg to the `init_chat_model(**kw)` call in a way that changes behavior from today (LangChain's `init_chat_model` already defaults `api_key` internally) — only pass it when not `None`. (Task 2)
- Both prices unset (the existing, already-passing "no cost" case) must still return `cost=None`, not start raising. (Task 3)
- Both prices set (the existing, already-passing "fully priced" case, exercised implicitly by nothing in the current test file — add explicit coverage) must still price normally. (Task 3)
- A token count of exactly `0` (not `None`) on the unpriced side must not force that side's price to be "required" — mirrors the `LLMClient.price` precedent's `input_tokens and in_price is None` (not `input_tokens is not None`) check. (Task 3)

---

## Task 1: `guess_provider` requires a right-hand boundary too

**Files:**
- Modify: `src/router/llm/adapters/langchain_adapter.py` (`guess_provider`, ~L35-44)
- Test: `tests/router/llm/adapters/test_langchain_adapter.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `guess_provider(model_id: str) -> Optional[str]` (unchanged signature) now requires the matched hint to be followed by either the end of the string or a non-alphanumeric character, the same way it already requires that on the left.

- [ ] **Step 1: Write the failing test**

Add to `tests/router/llm/adapters/test_langchain_adapter.py`:

```python
from router.llm.adapters.langchain_adapter import LangChainAdapter, guess_provider


def test_guess_provider_requires_a_right_hand_boundary_too():
    """MOD-001 (router.llm.adapters): a hint matching mid-token (inside a
    longer alnum run) must not count as a match -- only RA-19's existing
    left-boundary guard existed before this fix."""
    assert guess_provider("gpt4all-13b-snoozy") is None
    # existing passing cases must still match -- a boundary on the right,
    # not "no match at all"
    assert guess_provider("o1-mini") == "openai"
    assert guess_provider("gpt-4o") == "openai"
    assert guess_provider("yolo1") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/router/llm/adapters/test_langchain_adapter.py::test_guess_provider_requires_a_right_hand_boundary_too -v`
Expected: FAIL — `guess_provider("gpt4all-13b-snoozy")` returns `"openai"`, not `None`.

- [ ] **Step 3: Implement**

In `src/router/llm/adapters/langchain_adapter.py`, `guess_provider`:

```python
def guess_provider(model_id: str) -> Optional[str]:
    """Best-effort LangChain ``model_provider`` from a bare model id -- a last
    resort. Each hint must be its own token on both sides (``o1-mini``
    matches ``o1``, ``yolo1`` does not, ``gpt4all-13b-snoozy`` does not).
    Prefer an explicit ``provider=``/``model=`` per pool id: a substring
    can't tell which host actually serves a given model."""
    low = str(model_id).lower()
    for needle, provider in _PROVIDER_HINTS:
        if re.search(rf"(?:^|[^a-z0-9]){re.escape(needle)}(?:$|[^a-z0-9])", low):
            return provider
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/router/llm/adapters/test_langchain_adapter.py -v tests/router/test_review_fixes.py::test_guess_provider_matches_whole_prefixes_only tests/router/agentic/test_agentic_router.py::test_guess_provider`
Expected: PASS, all of them (the two pre-existing tests included — they must not regress).

- [ ] **Step 5: Commit**

```bash
git add src/router/llm/adapters/langchain_adapter.py tests/router/llm/adapters/test_langchain_adapter.py
git commit -m "fix(llm): guess_provider requires a right-hand token boundary too (MOD-001 router.llm.adapters)"
```

---

## Task 2: `LangChainAdapter` forwards `api_key` to `init_chat_model`

**Files:**
- Modify: `src/router/llm/adapters/langchain_adapter.py` (`LangChainAdapter.__init__`/`_model`, ~L77-99)
- Test: `tests/router/llm/adapters/test_langchain_adapter.py`

**Interfaces:**
- Consumes: `langchain.chat_models.init_chat_model(name, *, model_provider=None, **kw)` (external API, unchanged) — this task only changes what's in `kw`.
- Produces: `LangChainAdapter._model()`'s lazy-construction path now includes `api_key` in the kwargs passed to `init_chat_model` when `self._api_key` (stored by the already-existing `ProviderAdapter.__init__` base class call) is not `None`.

- [ ] **Step 1: Write the failing test**

Add to `tests/router/llm/adapters/test_langchain_adapter.py`:

```python
def test_model_forwards_api_key_to_init_chat_model(monkeypatch):
    """LANGCHAINADAPTER-001: an explicit api_key must reach init_chat_model,
    not be silently dropped in favour of ambient credentials."""
    import langchain.chat_models

    captured = {}

    def fake_init_chat_model(name, *, model_provider=None, **kw):
        captured["name"] = name
        captured["model_provider"] = model_provider
        captured.update(kw)
        return _StubChatModel()

    monkeypatch.setattr(langchain.chat_models, "init_chat_model", fake_init_chat_model)

    adapter = LangChainAdapter("gpt-4", provider="openai", api_key="sk-test")
    adapter._model()

    assert captured["api_key"] == "sk-test"


def test_model_omits_api_key_when_not_given(monkeypatch):
    """No api_key= given -- must not pass a spurious api_key=None that could
    shadow init_chat_model's own default resolution."""
    import langchain.chat_models

    captured = {}

    def fake_init_chat_model(name, *, model_provider=None, **kw):
        captured.update(kw)
        return _StubChatModel()

    monkeypatch.setattr(langchain.chat_models, "init_chat_model", fake_init_chat_model)

    adapter = LangChainAdapter("gpt-4", provider="openai")
    adapter._model()

    assert "api_key" not in captured
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/router/llm/adapters/test_langchain_adapter.py::test_model_forwards_api_key_to_init_chat_model tests/router/llm/adapters/test_langchain_adapter.py::test_model_omits_api_key_when_not_given -v`
Expected: `test_model_forwards_api_key_to_init_chat_model` FAILS with `KeyError: 'api_key'`; `test_model_omits_api_key_when_not_given` PASSES already (nothing forwards it yet either way) — that's fine, it's there to pin the "don't overcorrect" behavior once Step 3 lands.

- [ ] **Step 3: Implement**

`ProviderAdapter.__init__` (`src/router/llm/adapters/base.py:22-24`) already stores `self.api_key = api_key`, and `LangChainAdapter.__init__` already calls `super().__init__(model, api_key=api_key, **client_kwargs)` — so `self.api_key` is available, just unused by `_model()`.

In `src/router/llm/adapters/langchain_adapter.py`, `_model()`:

```python
    def _model(self):
        if self._chat is None:
            from langchain.chat_models import init_chat_model  # optional dep

            name, provider, kw = self._spec
            if self.api_key is not None:
                kw = {**kw, "api_key": self.api_key}
            self._chat = init_chat_model(name, model_provider=provider, **kw)
        return self._chat
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/llm/adapters/test_langchain_adapter.py -v`
Expected: PASS, all tests in the file (including the three pre-existing ones, which use `chat_model=chat` and never exercise `_model()`'s lazy path at all).

- [ ] **Step 5: Commit**

```bash
git add src/router/llm/adapters/langchain_adapter.py tests/router/llm/adapters/test_langchain_adapter.py
git commit -m "fix(llm): LangChainAdapter forwards an explicit api_key to init_chat_model (LANGCHAINADAPTER-001)"
```

---

## Task 3: `LangChainAdapter._cost` doesn't price an unset side at `$0`

**Files:**
- Modify: `src/router/llm/adapters/langchain_adapter.py` (`LangChainAdapter._cost`, ~L122-126)
- Test: `tests/router/llm/adapters/test_langchain_adapter.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `LangChainAdapter._cost(in_tok, out_tok) -> Optional[float]` (unchanged signature) now returns `None` when a side with a nonzero token count has no configured price, instead of silently treating that side's price as `0.0`.

- [ ] **Step 1: Write the failing test**

Add to `tests/router/llm/adapters/test_langchain_adapter.py`:

```python
class _PricedChatModel(_StubChatModel):
    """Like _StubChatModel, but its response carries usage_metadata."""

    def invoke(self, prompt, **kwargs):
        msg = super().invoke(prompt, **kwargs)
        msg.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
        return msg


def test_complete_prices_nothing_when_only_one_side_is_configured():
    """LANGCHAINADAPTER-002: a half-priced adapter (only input_cost_per_token
    given) must not silently price the unset output side at $0."""
    chat = _PricedChatModel()
    adapter = LangChainAdapter("gpt-4", chat_model=chat, input_cost_per_token=1e-6)

    resp = adapter.complete("hi")

    assert resp.cost is None


def test_complete_prices_normally_when_both_sides_are_configured():
    chat = _PricedChatModel()
    adapter = LangChainAdapter(
        "gpt-4", chat_model=chat, input_cost_per_token=1e-6, output_cost_per_token=2e-6,
    )

    resp = adapter.complete("hi")

    assert resp.cost == pytest.approx(100 * 1e-6 + 50 * 2e-6)
```

(Check the top of the test file for a `pytest` import — add `import pytest` if it isn't already there.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/router/llm/adapters/test_langchain_adapter.py::test_complete_prices_nothing_when_only_one_side_is_configured -v`
Expected: FAIL — `resp.cost == 0.0001` (only the input side priced, output silently at `$0`), not `None`.

- [ ] **Step 3: Implement**

In `src/router/llm/adapters/langchain_adapter.py`, `_cost`:

```python
    def _cost(self, in_tok: Optional[int], out_tok: Optional[int]) -> Optional[float]:
        """``None`` (unknown) when there are no token counts at all, or when a
        side with a nonzero token count has no configured price -- an unpriced
        side must not be silently treated as free (LANGCHAINADAPTER-002)."""
        in_price, out_price = self._prices
        if in_tok is None and out_tok is None:
            return None
        if (in_tok and in_price is None) or (out_tok and out_price is None):
            return None
        return float((in_tok or 0) * (in_price or 0.0) + (out_tok or 0) * (out_price or 0.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/llm/adapters/test_langchain_adapter.py -v`
Expected: PASS, all tests in the file.

- [ ] **Step 5: Run the full test suite (this file's tests are the only direct coverage, but confirm no other consumer relies on the old zero-fill)**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/router/llm/adapters/langchain_adapter.py tests/router/llm/adapters/test_langchain_adapter.py
git commit -m "fix(llm): LangChainAdapter doesn't price an unconfigured side at \$0 (LANGCHAINADAPTER-002)"
```

---

## Final step: update `known_bugs/`

Update status to `fixed` (with the fixing commit summary, following the existing convention) in:
- `fixed_bugs/_module_level/router.llm.adapters/MOD-001-guess-provider-missing-end-boundary/bug.md`
- `fixed_bugs/ProviderAdapter/LangChainAdapter/LANGCHAINADAPTER-001-api-key-not-forwarded/bug.md`
- `fixed_bugs/ProviderAdapter/LangChainAdapter/LANGCHAINADAPTER-002-partial-price-defaults-to-zero/bug.md`

And the corresponding rows in `known_bugs/README.md`'s `## Bugs` table (leave the `## Coverage` table's bug counts as-is — they count bugs filed, not open bugs, per the existing convention already used for `LocalArtifactStore`/`AgenticRouter`).
