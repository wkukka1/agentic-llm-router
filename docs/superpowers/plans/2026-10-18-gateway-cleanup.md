# Gateway Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the duplicated and dead code found in `src/router/gateway/` and `src/evaluation/gateway/` without changing gateway behaviour.

**Architecture:** Small shared helpers first (they delete the most copies for the least risk), then dead-surface removal, then module moves, then the three behaviour-sensitive consolidations (lock, picker decision rule, graded kNN), each pinned by a test written before the change. No new classes; helpers are module functions.

**Tech Stack:** Python 3.12, pytest, FastAPI/httpx (gateway), numpy.

**Spec:** [code_review/gateway_cleanup.md](../../../code_review/gateway_cleanup.md). Finding IDs (D#, X#, S#) below refer to it.

## Global Constraints

- **Behaviour-preserving.** Request-log record keys, `preferences.json` / `turns.jsonl` / `graded.jsonl` / `thread-models.jsonl` formats and the HTTP API do not change. The existing `tests/router/gateway/` and `tests/evaluation/gateway/` suites stay green after every task.
- **No new classes** (`gating-new-classes` skill). Module-level functions and constants only.
- **No compatibility shims.** When a symbol moves or is deleted, update every call site and test; do not leave a re-export behind.
- **`hook` and `statusline` stay stdlib-only.** `python -c "import sys, router.gateway.cli"` must not import `yaml`, `fastapi`, `httpx` or `numpy` (pinned by Task 5).
- **Windows is the dev platform.** No POSIX-only calls in new code paths; keep the `os.name == "nt"` branches.
- Run tests with `python -m pytest tests/router/gateway tests/evaluation/gateway -q`. Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. `git add` only the files a task names: the working tree has unrelated deletions under `fixed_bugs/`.

## Review Focus

- `prompt_hash` from the interceptor (`turns.jsonl`) and from the eval modules (`history`, `pool`, `subagent_baseline`) must stay byte-identical, or graded rows stop matching captured turns. Pinned in Task 1.
- OpenAI-lane request whose fallback to Anthropic hits a network error must log lane `anthropic`, status 502, not lane `openai`. Pinned in Task 12.
- A Claude-settings lock that cannot be taken must make `_check` return silently (the user's message is never blocked or crashed). Pinned in Task 10.
- A graded file with fewer than `min_neighbors` similar prompts must still give `None` / no quality for that model after the kNN is shared. Pinned in Task 16.
- Two gateway processes writing `preferences.json`-style files on Windows while another program has them open must still succeed (`atomic_write_text` retry); do not replace it while consolidating IO helpers. Covered by existing `test_concurrency.py` (run it in Tasks 1 and 9).

---

### Task 0: Isolated branch and green baseline

**Files:** none changed.

- [ ] **Step 1:** Use `superpowers:using-git-worktrees` to create a worktree on a new branch `chore/gateway-cleanup` from the current HEAD (the main tree has unrelated uncommitted deletions).
- [ ] **Step 2:** In the worktree run `python -m pytest tests/router/gateway tests/evaluation/gateway -q`. Expected: all pass. Record the count; every later task must end with the same count plus its own new tests. If anything fails on the baseline, stop and report.

---

### Task 1: Shared helpers in `config.py` (D5, D7, D8)

**Files:**
- Modify: `src/router/gateway/config.py` (add helpers)
- Modify call sites: `interceptor.py` (sha1 at `_task_key`, `_capture`, `_conversation_key`, `prefix_signature`; `isoformat` in `_log_override`, `_capture`, `_log_record`; `_read_preferences`), `share.py` (`state`, three `isoformat`), `cli.py` (`_user_has_statusline`, `_read_claude_settings`), `src/evaluation/gateway/{history,pool,subagent_baseline,grade}.py`
- Test: `tests/router/gateway/test_gateway_config.py`

**Interfaces:**
- Produces: `short_hash(text: str) -> str`, `now_iso() -> str`, `read_json(path: Path, default: dict | None = None) -> dict`.

- [ ] **Step 1: Write failing tests**

```python
def test_short_hash_is_the_historical_prompt_hash():
    import hashlib
    assert short_hash("fix the bug") == hashlib.sha1(b"fix the bug").hexdigest()[:16]

def test_now_iso_has_seconds_precision_and_utc_offset():
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", now_iso())

def test_read_json_returns_default_for_missing_or_corrupt(tmp_path):
    assert read_json(tmp_path / "nope.json") == {}
    bad = tmp_path / "bad.json"; bad.write_text("{oops")
    assert read_json(bad, {"x": 1}) == {"x": 1}
    ok = tmp_path / "ok.json"; ok.write_text('{"a": 2}')
    assert read_json(ok) == {"a": 2}
```

- [ ] **Step 2:** Run `python -m pytest tests/router/gateway/test_gateway_config.py -q`. Expected: FAIL (ImportError).
- [ ] **Step 3:** Implement the three functions in `config.py`. `short_hash` is `hashlib.sha1(text.encode()).hexdigest()[:16]`; `read_json` swallows `OSError`/`ValueError` only. Replace every call site listed above (the sha1 sites keep their own input expression, e.g. `short_hash(_REMINDER.sub("", task or "").strip())`; `share._append`/`_version` untouched). `read_json` replaces the four `json.loads(path.read_text())` + `except (OSError, ValueError)` blocks, keeping `cli._read_claude_settings` raising `SystemExit` on invalid JSON (it needs its own message: leave that one, convert only the other three).
- [ ] **Step 4:** Run the full gateway + eval suites (Task 0 command), plus `python -m pytest tests/router/gateway/test_concurrency.py -q`. Expected: all pass.
- [ ] **Step 5: Commit** `refactor(gateway): one short_hash, now_iso and read_json helper` with the named files only.

---

### Task 2: Pricing helpers (D9, D10, D18)

**Files:**
- Modify: `src/router/gateway/pricing.py`, `interceptor.py` (`_prompt_tokens`, `session_summary`), `cli.py` (`_status`)
- Test: `tests/router/gateway/test_gateway_pricing.py`

**Interfaces:**
- Produces: `prompt_tokens(usage: Mapping) -> int` (input + cache read + cache creation), `saved_fraction(cost: float, base: float) -> float | None`.
- `interceptor._prompt_tokens` is deleted; callers use `pricing.prompt_tokens`.

- [ ] **Step 1: Failing tests**

```python
def test_prompt_tokens_sums_the_three_input_kinds():
    assert prompt_tokens({"input_tokens": 1, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 300}) == 321
    assert prompt_tokens({"input_tokens": None}) == 0

def test_saved_fraction():
    assert saved_fraction(3.0, 4.0) == 0.25
    assert saved_fraction(5.0, 4.0) == -0.25
    assert saved_fraction(1.0, 0.0) is None

def test_cost_usd_is_none_for_half_given_price():
    assert cost_usd({"input": 1.0}, {"input_tokens": 5}) is None
```

- [ ] **Step 2:** Run it; expect FAIL on the first two (ImportError); the third already passes (it pins D10).
- [ ] **Step 3:** Implement. `cost_usd` starts `if not has_price(price) or not usage: return None` and uses `prompt_tokens(usage)` for `price_at`. `session_summary` computes `saved = saved_fraction(tot["cost_usd"], base)` once and derives both the text and `plan_usage_saved` from it; `cli._status` uses `saved_fraction` with `round(..., 3)`.
- [ ] **Step 4:** Full suites. Expected: pass.
- [ ] **Step 5: Commit** `refactor(gateway): prompt_tokens and saved_fraction live in pricing`.

---

### Task 3: One list of Claude tiers; one READ_ONLY (D11, D12)

**Files:**
- Modify: `pricing.py` (constants), `config.py` (`DEFAULT_MODELS` keys), `subagents.py` (`_HAIKU`, `_SONNET`), `src/evaluation/gateway/{grade,pool_eval,history}.py`
- Test: `tests/router/gateway/test_gateway_config.py`, `tests/evaluation/gateway/test_pool_eval.py`

**Interfaces:**
- Produces in `pricing.py`: `OPUS = "claude-opus-5-5"`, `SONNET = "claude-sonnet-5-5"`, `HAIKU = "claude-haiku-5-5"`, `TIERS = (HAIKU, SONNET, OPUS)` (cheapest first).
- `grade.DEFAULT_TIERS` keeps its order `(OPUS, SONNET, HAIKU)`; `grade.DEFAULT_JUDGE = OPUS`; `pool_eval.TIERS` is `pricing.TIERS`; `history.mark_self_contained(model=HAIKU)`.

- [ ] **Step 1: Failing tests**

```python
def test_every_tier_is_configured_and_priced():
    assert set(TIERS) <= set(DEFAULT_MODELS) and set(TIERS) <= set(CLAUDE_PRICES)

def test_grade_and_pool_eval_use_the_shared_tiers():
    from evaluation.gateway import grade, pool_eval
    assert set(grade.DEFAULT_TIERS) == set(TIERS) and pool_eval.TIERS is TIERS
```

- [ ] **Step 2:** Run; expect FAIL (ImportError for `TIERS`).
- [ ] **Step 3:** Add the constants; use them as `DEFAULT_MODELS` keys for the three 5.5 entries, in `DEFAULT_ROLE_MODELS`, and in the three eval modules. In `history.mark_self_contained` replace the retyped `("--permission-mode", "plan", "--strict-mcp-config", "--safe-mode")` with `READ_ONLY` imported from `.grade` inside the function (grade imports history lazily, so no cycle).
- [ ] **Step 4:** Full suites. Expected: pass.
- [ ] **Step 5: Commit** `refactor(gateway): one definition of the Claude tiers and the read-only replay flags`.

---

### Task 4: Complete `DEFAULT_ROUTER` and read it once (D13)

**Files:**
- Modify: `config.py` (`DEFAULT_ROUTER`), `routing.py` (`build_picker`, `load_router_model`), `server.py` (`_lifespan`, `_graded_predictor`), `interceptor.py` (`_ttl`, `PRESETS`), `graded.py` (`min_prompts` default)
- Test: `tests/router/gateway/test_routing.py`, `test_gateway_config.py`

**Interfaces:**
- `DEFAULT_ROUTER` gains `"use_graded": True`, `"min_graded_prompts": 20`, `"calibration_shrink": 20.0`, `"artifact": None`.
- `build_picker` starts with `rcfg = {**DEFAULT_ROUTER, **settings["router"]}` (so hand-built test settings keep working) and uses `rcfg[key]` with no inline defaults.

- [ ] **Step 1: Failing tests**

```python
def test_default_router_names_every_key_the_code_reads():
    assert {"use_graded", "min_graded_prompts", "calibration_shrink", "artifact"} <= set(DEFAULT_ROUTER)

def test_balanced_preset_is_the_default_margin():
    assert PRESETS["balanced"] == DEFAULT_ROUTER["margin"]

def test_picker_built_from_sparse_router_settings_uses_defaults(...):
    # settings["router"] == {"enabled": True}; with a fake router_model the picker
    # prices with horizon 20 and switch_margin 0.05: assert pick()["quality_margin"] == 0.05
```

- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement; delete every `.get(key, literal)` default for these keys in the four files; `PRESETS["balanced"] = DEFAULT_ROUTER["margin"]` (import from `.config`); `graded_predictor`'s `min_prompts` default is `DEFAULT_ROUTER["min_graded_prompts"]`.
- [ ] **Step 4:** Full suites. Expected: pass.
- [ ] **Step 5: Commit** `refactor(gateway): router defaults are defined once`.

---

### Task 5: `paths.py` (stdlib-only) and `cli` helpers (D14-D17, D22)

**Files:**
- Create: `src/router/gateway/paths.py`
- Modify: `config.py` (re-import path functions from `paths`; they stay importable from `config` because they are used there), `cli.py`, `server.py`, `src/evaluation/gateway/grade.py`
- Test: `tests/router/gateway/test_cli.py`, `test_paths.py`

**Interfaces:**
- Produces in `paths.py` (move, don't copy): `log_path()`, `gateway_home()`, `pid_file()`, `graded_path()`, `config_path()`, `local_models_path()`.
- Produces in `cli.py`: `_host() -> str`, `_port() -> int`, `_restart() -> None` (`_stop()` then `_ensure_running(_base_url(), quiet=True)`), `_detached() -> dict` (Popen kwargs), `_self_argv() -> list[str]` (`[sys.executable, "-m", "router.gateway.cli"]`), `_has_login_service() -> bool`.
- Deleted: `cli._home`, `_config_path`, `_pid_file`, `server._pid_file`, `grade._home`.

- [ ] **Step 1: Failing tests**

```python
def test_importing_cli_stays_stdlib_light():
    out = subprocess.run([sys.executable, "-c",
        "import sys, router.gateway.cli as c; "
        "print(sorted(m for m in ('yaml','fastapi','httpx','numpy') if m in sys.modules))"],
        capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"

def test_base_url_follows_host_and_port_env(monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_HOST", "0.0.0.0"); monkeypatch.setenv("AAR_GATEWAY_PORT", "9000")
    assert cli._base_url() == "http://0.0.0.0:9000" and cli._port() == 9000

def test_restart_stops_then_starts(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_stop", lambda: calls.append("stop"))
    monkeypatch.setattr(cli, "_ensure_running", lambda *a, **k: calls.append("start"))
    cli._restart(); assert calls == ["stop", "start"]
```

- [ ] **Step 2:** Run; expect FAIL (`_port`, `_restart` missing; the import test may already pass: keep it, it guards later steps).
- [ ] **Step 3:** Create `paths.py` by moving the six functions out of `config.py` (`config.py` imports the ones it still uses; update `from .config import log_path…` sites to `from .paths import …` where convenient, but `config` must still export nothing it no longer defines: no re-export shim). `cli.py` imports `paths` at module top. `_service_serves_this_port` compares against `str(_port())` default. Replace the three stop+start pairs (`_local`, `_setup`, `restart`) with `_restart()`, the two `DETACHED_PROCESS`/`start_new_session` dicts with `_detached()` (the `_check` one passes only `creationflags=DETACHED_PROCESS` today: use the same helper, adding `CREATE_NEW_PROCESS_GROUP`), the four `-m router.gateway.cli` argv lists with `_self_argv()`, and the three darwin/linux tests with `_has_login_service()`. In `server._lifespan` rename the local `pid_file` variable to `pid` and use the imported function.
- [ ] **Step 4:** Full suites. Expected: pass.
- [ ] **Step 5: Commit** `refactor(gateway): stdlib-only paths module; cli env, restart and spawn helpers`.

---

### Task 6: `translate.py` cleanup (X5, X9, D19, D21)

**Files:**
- Modify: `translate.py`, `interceptor.py`, `subagents.py`, `src/evaluation/gateway/history.py`
- Test: `tests/router/gateway/test_translate.py`, `test_text_of.py`

**Interfaces:**
- Public renames: `_blocks` → `blocks`, `REMINDER` imported without alias.
- Produces: `parse_sse_line(line: str) -> dict | None` (None for non-data lines, `[DONE]`, or bad JSON) replacing `iter_openai_sse`; `_stop_reason(finish: str | None, saw_tool: bool) -> str`; `_tool_id() -> str` (`toolu_` + 24 hex).
- Deleted: `iter_openai_sse`, `history._text`, `subagents._system_text` (inline `text_of(..., sep="\n", typed=False)`).

- [ ] **Step 1: Failing tests**

```python
def test_parse_sse_line():
    assert parse_sse_line('data: {"a": 1}') == {"a": 1}
    assert parse_sse_line("data: [DONE]") is None
    assert parse_sse_line(": keep-alive") is None
    assert parse_sse_line("data: {oops") is None

def test_stop_reason_maps_and_upgrades_stop_with_tools():
    assert _stop_reason("length", False) == "max_tokens"
    assert _stop_reason("stop", True) == "tool_use"
    assert _stop_reason(None, False) == "end_turn"

def test_tool_id_shape():
    assert re.fullmatch(r"toolu_[0-9a-f]{24}", _tool_id())
```

- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement; `openai_to_anthropic` and `OpenAIStreamTranslator.finish` both call `_stop_reason`; both `toolu_` sites call `_tool_id`; the interceptor stream loop becomes `for line in …: chunk = parse_sse_line(line); if chunk: for event in translator.feed(chunk): yield event`.
- [ ] **Step 4:** Full suites, then `python -m pytest tests/e2e -q -k "not claude_e2e"` if that selection exists. Expected: pass.
- [ ] **Step 5: Commit** `refactor(gateway): translate helpers are public and single-sourced`.

---

### Task 7: Delete test-only setters and the server shim (X1, X2, X4)

**Files:**
- Modify: `interceptor.py` (delete `set_model_picker`, `set_upstream_client`), `server.py` (delete `main`), tests that call them (`grep -rn "set_model_picker\|set_upstream_client" tests`)
- Test: the same test files, edited

- [ ] **Step 1:** Run `grep -rn "router.gateway.server" src tests docs configs scripts pyproject.toml`. If any non-test hit relies on `server.main`/`python -m router.gateway.server`, keep `server.main` and say so in the commit; otherwise delete it. Ask the user to run `grep -rn "router.gateway.server" ~/.claude/settings.json` before deleting (old sessions' hooks), and wait for the answer.
- [ ] **Step 2:** Rewrite the tests: `set_model_picker(p)` → pass `p` as the `picker` argument of `interceptor.configure(settings, p)`; `set_upstream_client(c)` → `monkeypatch.setattr(interceptor, "_client", c)` in a fixture in `tests/router/gateway/conftest.py` named `upstream_client(monkeypatch)` returning a setter. Run them: expect PASS against the unchanged source.
- [ ] **Step 3:** Delete the two setters (and `server.main` if cleared in Step 1).
- [ ] **Step 4:** Full suites. Expected: pass.
- [ ] **Step 5: Commit** `chore(gateway): delete test-only setters`.

---

### Task 8: One router-status accessor; server cleanups (X3, X10, S5)

**Files:**
- Modify: `interceptor.py`, `server.py`
- Test: `tests/router/gateway/test_server.py`

**Interfaces:**
- Produces: `interceptor.router_status() -> dict` with keys `enabled: bool`, `error: str | None`, `kind: str | None`, `source: str | None`, `models: list[str]`, and `calibration` when the picker has `calibration_status`. `/health` returns `{"status": "ok", "pid": os.getpid(), "router": interceptor.router_status()}` (same JSON shape as today).

- [ ] **Step 1: Failing tests** (against the running test app):

```python
def test_health_router_block_when_router_disabled(client):
    r = client.get("/health").json()["router"]
    assert r["enabled"] is False and set(r) >= {"enabled", "error", "kind", "source", "models"}

def test_preferences_400s_report_the_message(client):
    assert client.post("/api/preferences", json={"mode": "bogus"}).status_code == 400
    assert client.post("/api/preferences", json={"preset": "nope"}).status_code == 400
    assert client.post("/api/preferences", json={}).status_code == 400
```

- [ ] **Step 2:** Run; the first fails only if `error`/`source` keys are missing today, the second passes: keep both as the characterization for Step 3.
- [ ] **Step 3:** Move the `_router_status` computation into `interceptor.configure` (store `_router_info`) and `router_status()`; delete `server._router_status`, the `getattr(interceptor, "_picker")` and the `rcfg` block in `_lifespan`. Hoist `asyncio` / `StreamingResponse` imports in `events`. In `update_preferences` use one `try/except ValueError → HTTPException(400)` around the whole dispatch.
- [ ] **Step 4:** Full suites. Expected: pass.
- [ ] **Step 5: Commit** `refactor(gateway): router health from one accessor; simpler preferences route`.

---

### Task 9: Log readers reuse `read_jsonl`; share queue appends via `append_jsonl` (D4)

**Files:**
- Modify: `interceptor.py` (`_rotated_records`, `_current_records`), `share.py` (`_queue_line`)
- Test: `tests/router/gateway/test_jsonl.py`

- [ ] **Step 1:** Add characterization tests first (they must pass on current code): a rotated file `requests.jsonl.1` with one cut-short last line yields only the complete records; a current log with a partial trailing line yields it only after the newline arrives; `read_records()` returns rotated + current oldest first. Run: PASS.
- [ ] **Step 2:** `_rotated_records` calls `read_jsonl(path)` inside its existing cache-key check. `_current_records` keeps the tail-offset logic. `share._queue_line` writes with `append_jsonl(path, json.loads(line))` is *not* wanted (double encode): change `_queue_line(line)` to `_queue_record(record: dict) -> int` and `_append` passes the dict.
- [ ] **Step 3:** Run the Step-1 tests plus `test_concurrency.py` and `test_share.py`. Expected: pass.
- [ ] **Step 4: Commit** `refactor(gateway): log and share queue reuse the jsonl helpers`.

---

### Task 10: Claude-settings lock uses `locked_file` (D3)

**Files:**
- Modify: `cli.py` (`_settings_lock`)
- Test: `tests/router/gateway/test_filelock.py`

**Interfaces:**
- `_settings_lock(path: Path, *, wait: float = 10.0)` keeps its name and context-manager use, implemented as `locked_file(path, timeout=wait)`; a `TimeoutError` becomes `SystemExit(f"{lock} is held by another aar-gateway command; try again")`. The `stale` parameter and the `.aar-lock` file are removed.

- [ ] **Step 1: Failing tests**

```python
def test_settings_lock_times_out_with_systemexit(tmp_path):
    p = tmp_path / "settings.json"
    with locked_file(p):
        with pytest.raises(SystemExit):
            with cli._settings_lock(p, wait=0.2): pass

def test_settings_lock_released_after_exception(tmp_path):
    p = tmp_path / "settings.json"
    with pytest.raises(RuntimeError):
        with cli._settings_lock(p): raise RuntimeError
    with cli._settings_lock(p, wait=0.2): pass

def test_check_returns_quietly_when_the_lock_is_held(tmp_path, monkeypatch, capsys):
    # gateway unhealthy, settings lock held elsewhere -> cli._check(False) returns, prints nothing
```

- [ ] **Step 2:** Run; expect FAIL on the first (the old lock is a different file, so nothing blocks it).
- [ ] **Step 3:** Implement with a lazy `from .config import locked_file` inside the function (keeps `cli` import light).
- [ ] **Step 4:** Full suites; also `python -m pytest tests/router/gateway/test_filelock.py tests/router/gateway/test_cli.py -q`. Expected: pass.
- [ ] **Step 5: Commit** `refactor(gateway): one cross-process file lock`.

---

### Task 11: Move dashboard aggregation to `stats.py` (S1)

**Files:**
- Create: `src/router/gateway/stats.py`
- Modify: `interceptor.py` (remove ~190 lines), `server.py`, tests that import `read_records` / `summarize*` from `interceptor` (`grep -rn "interceptor.read_records\|interceptor.summarize\|RECENT_RECORDS\|SESSION_ROWS" tests src`)

**Interfaces:**
- Produces in `stats.py` (moved verbatim, then adjusted): `read_records() -> list[dict]`, `summarize(records: list[dict] | None = None, *, strongest: str | None = None) -> dict`, `summarize_sessions(records) -> list[dict]`, `summarize_subagents(records) -> list[dict]`, constants `RECENT_RECORDS`, `SESSION_ROWS`; private `_log_cache`, `_rotated_cache`, `_baseline_cost`, `_expected_thread_expiry`.
- `summarize` uses `strongest` as the fallback where it used `_settings.get("strongest")`. Add `interceptor.strongest_model() -> str | None`; `server` calls `stats.summarize(strongest=interceptor.strongest_model())`.
- Stays in `interceptor`: `session_summary` (reads `_cc_sessions`), `_append_record`, `configure` (now `from . import stats` for the profile seed).

- [ ] **Step 1:** Run the current suites; confirm the aggregation tests (`test_interceptor.py`, `test_server.py`) pass: they are the characterization.
- [ ] **Step 2:** Add one test in `tests/router/gateway/test_stats.py` that builds a fixed list of 6 records (priced, unpriced, subagent, a 404 "No thread state") and asserts `summarize(records)["totals"]` values: `requests`, `errors` (the thread expiry is not an error), `priced_requests`, and that `summarize_sessions` treats an unpriced baseline as `cost_usd` while totals skip it. Run against the *current* `interceptor.summarize` first (import path adjusted after the move): PASS.
- [ ] **Step 3:** Move the code; update `server.py` and tests (sed `interceptor.read_records` → `stats.read_records` etc.). No re-exports.
- [ ] **Step 4:** Full suites. Expected: pass. `wc -l src/router/gateway/interceptor.py` is about 1,120 or less.
- [ ] **Step 5: Commit** `refactor(gateway): dashboard aggregation moves to stats.py`.

---

### Task 12: Collapse the `_unguarded` pairs (S2)

**Files:**
- Modify: `interceptor.py`
- Test: `tests/router/gateway/test_interceptor_routing.py`

**Interfaces:**
- Produces: `async def _route(turn: _Turn, lane: str) -> Response` — calls `_via_anthropic(turn)` or `_via_openai(turn)` (the former `_unguarded` bodies, renamed) inside one `try/except httpx.HTTPError as exc: return _unreachable(turn, "Anthropic API" if lane == "anthropic" else turn.target, exc, lane)`.
- `handle_messages` and the openai→anthropic fallback inside `_via_openai` both call `_route(turn, "anthropic")` / `_route(turn, lane)`.

- [ ] **Step 1: Failing/characterization tests**

```python
async def test_unreachable_anthropic_logs_a_502_on_the_anthropic_lane(...): ...   # passes today
async def test_openai_failure_falling_back_to_unreachable_anthropic_logs_anthropic_502(...):
    # routed model on openai lane returns 500; fallback to requested Claude model raises httpx.ConnectError
    # assert response 502 and last record lane == "anthropic", status == 502, fallback is True
```

- [ ] **Step 2:** Run. The second must PASS on current code (this is the Review Focus behaviour); if it fails, stop and tell the user: that is an existing bug, not a refactor.
- [ ] **Step 3:** Refactor as specified.
- [ ] **Step 4:** Full suites plus `tests/router/gateway/test_concurrency.py`. Expected: pass.
- [ ] **Step 5: Commit** `refactor(gateway): one network-error guard for both lanes`.

---

### Task 13: One CLI dispatcher (S3)

**Files:**
- Modify: `cli.py` (`main`)
- Test: `tests/router/gateway/test_cli.py`

**Interfaces:**
- Pre-argparse fast path keeps only `code`, `hook`, `check`, `statusline` (stdlib-only, argument-passthrough). `local`, `share`, `agents` become real subparsers taking `args: nargs=argparse.REMAINDER` and calling `_local(args.args)`, `_share(args.args)`, `_agents(args.args)`; the help-only duplicates of `hook`/`statusline` stay so `--help` lists them.

- [ ] **Step 1: Failing/characterization tests**

```python
@pytest.mark.parametrize("argv", [["local"], ["share", "status"], ["agents"]])
def test_subcommands_still_dispatch(argv, monkeypatch): ...   # each calls its handler with argv[1:]

def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit): cli.main(["--help"])
    out = capsys.readouterr().out
    for c in ("code", "start", "stop", "restart", "status", "logs", "setup", "enable", "disable",
              "doctor", "local", "share", "config", "dashboard", "service", "serve", "agents",
              "statusline", "hook"):
        assert c in out
```

- [ ] **Step 2:** Run; the dispatch tests pass today, the help test passes: keep as characterization.
- [ ] **Step 3:** Refactor; `share on|off|…` words must reach `_share` unmodified (REMAINDER, not parsed as options).
- [ ] **Step 4:** Full suites. Expected: pass.
- [ ] **Step 5: Commit** `refactor(gateway): cli dispatches through argparse except the stdlib fast path`.

---

### Task 14: Lift the picker's pure rules to module level (S6, X6)

**Files:**
- Modify: `routing.py`
- Test: `tests/router/gateway/test_routing.py`

**Interfaces:**
- Produces: `warm_prefix_tokens(model: str, ctx: Mapping) -> float`;
  `request_cost(spec: Mapping, price: Mapping, ctx: Mapping, *, warm: float, horizon: int, write_mult: float, default_out_tokens: float) -> float` (the body of today's `estimate`, `warm` already resolved by the caller);
  `stays_on_current(current_quality: float, best_quality: float, margin: float, switch_margin: float) -> bool` (`current_quality >= best_quality - margin - switch_margin`);
  `calibration_offsets(...) -> tuple[dict[str, float], int]` (offsets and the number of graded prompts; `build_picker.offsets` no longer calls `load_graded` a second time).

- [ ] **Step 1: Failing tests** (numbers hand-computed from the current formula)

```python
ANTHROPIC = {"lane": "anthropic"}; PRICE = {"input": 1.0, "output": 5.0, "cache_read": 0.1}
CTX = {"est_prompt_tokens": 1000, "usage_profile": {"output_tokens": 100}}

def test_request_cost_when_the_model_holds_the_whole_prefix():
    assert request_cost(ANTHROPIC, PRICE, CTX, warm=1000, horizon=20, write_mult=2.0, default_out_tokens=800) == pytest.approx(6.0e-4)

def test_request_cost_when_cold_pays_one_rewrite_over_the_horizon():
    assert request_cost(ANTHROPIC, PRICE, CTX, warm=0, horizon=20, write_mult=2.0, default_out_tokens=800) == pytest.approx(6.95e-4)

def test_request_cost_openai_lane_has_no_cache_terms():
    assert request_cost({"lane": "openai"}, {"input": 2.0, "output": 10.0}, {"est_prompt_tokens": 1000, "usage_profile": {"output_tokens": 100}},
                        warm=0, horizon=20, write_mult=2.0, default_out_tokens=800) == pytest.approx(3.0e-3)

def test_stays_on_current_rule():
    assert stays_on_current(0.80, 0.90, 0.05, 0.05) is True    # gap 0.10 == 0.05+0.05 boundary
    assert stays_on_current(0.79, 0.90, 0.05, 0.05) is False

def test_warm_prefix_tokens_prefers_the_conversations_own_prompt():
    assert warm_prefix_tokens("m", {"last_model": "m", "last_prompt_tokens": 500, "warm_tokens": {"m": 900}}) == 500
    assert warm_prefix_tokens("m", {"last_model": "x", "warm_tokens": {"m": 900}}) == 900
```

- [ ] **Step 2:** Run; expect FAIL (ImportError). If a hand-computed value disagrees with the current closure, run the current `estimate` on the same inputs and trust it (the plan's arithmetic is a check, not the oracle).
- [ ] **Step 3:** Lift the closures; `build_picker` calls them. `calibration_offsets` returns the tuple; update `build_picker.offsets` and its tests.
- [ ] **Step 4:** Full suites. Expected: pass.
- [ ] **Step 5: Commit** `refactor(gateway): picker cost and stay rules are module functions with tests`.

---

### Task 15: Picker decides with `routing_decision` (D2)

**Files:**
- Modify: `routing.py` (`pick`)
- Test: `tests/router/gateway/test_routing.py`

**Interfaces:**
- Consumes: `router.nirt.routing_decision.routing_decision(pred: ndarray[1, M], *, model_costs: ndarray[M], margin: float) -> ndarray[1]`.
- Produces: `choose_within_margin(qualities: list[float], costs: list[float], margin: float) -> int` in `routing.py` — index of the cheapest candidate whose quality is within `margin` of the best; candidates are first ordered by quality descending (stable), and `margin == 0` is replaced by `np.finfo(float).tiny`, exactly as `MarginRoutingPolicy.decide` does, so ties keep the better-ranked model.

- [ ] **Step 1: Failing tests**

```python
def test_choose_within_margin_picks_cheapest_inside_the_band():
    assert choose_within_margin([0.90, 0.88, 0.60], [10.0, 2.0, 1.0], 0.05) == 1

def test_choose_within_margin_zero_margin_breaks_exact_ties_by_cost():
    assert choose_within_margin([0.9, 0.9], [5.0, 1.0], 0.0) == 1

def test_choose_within_margin_equal_cost_keeps_the_better_ranked():
    assert choose_within_margin([0.80, 0.90], [1.0, 1.0], 0.2) == 1

def test_pick_matches_the_old_policy_path_on_random_pools():
    # property test: 200 random (quality, cost, margin) draws; the new pick's chosen model
    # equals MarginRoutingPolicy.decide's selected_model_id on the same ModelScore list
```

The last test imports `MarginRoutingPolicy` itself as its oracle; it lives only in the test file and is deleted with Step 4 if the user prefers.

- [ ] **Step 2:** Run; expect FAIL on the first three (ImportError); the property test is written against the policy and must pass once Step 3 lands.
- [ ] **Step 3:** Implement `choose_within_margin`; in `pick` replace the `RoutingRequest`/`RoutingConstraints`/`RoutingContext`/`PromptSignals`/`LLMProfile`/`ModelScore` construction with plain lists of `(model, quality, expected_cost, output_price)`; keep the reason string `f"cheapest within {margin:g} utility of the best"` and the stay-on-current block (using `stays_on_current`; "pricier" is `output price` compare as today). Delete the now-unused imports (`PromptSignals`, `RoutingConstraints`, `RoutingContext`, `RoutingRequest`, `ModelScore`, `LLMProfile`, `MarginRoutingPolicy`).
- [ ] **Step 4:** Full suites, including `tests/router/gateway/test_live.py` selection that does not need the network. Expected: pass; `python -c "import router.gateway.routing, sys; print([m for m in sys.modules if m.startswith(('router.policy','router.context','decompose'))])"` prints `[]` or only modules imported for other reasons — report what it prints.
- [ ] **Step 5: Commit** `refactor(gateway): the picker calls routing_decision directly`.

---

### Task 16: One graded-prompt kNN, shared by the gateway and `pool_eval` (D1, D2b)

**Files:**
- Modify: `graded.py`, `src/evaluation/gateway/pool_eval.py`
- Test: `tests/router/gateway/test_graded.py`, `tests/evaluation/gateway/test_pool_eval.py`

**Interfaces:**
- Produces in `graded.py`: `neighbor_quality(sims: np.ndarray, scores: np.ndarray, *, k: int, min_similarity: float, min_neighbors: int) -> tuple[np.ndarray, int, float] | None` — for one query's similarities `sims[N]` and `scores[N, M]` (nan = ungraded), returns `(quality[M] with nan where fewer than min_neighbors graded neighbours, n_neighbors, max_similarity)`, or `None` when fewer than `min_neighbors` neighbours reach `min_similarity`. Weights are `np.clip(sim, 1e-3, None)`.
- `graded_predictor.predict` builds its dict from it (drops nan models); `pool_eval.loo_knn` calls it per row with the diagonal set to `-inf`, `min_similarity=-inf`, `min_neighbors=1`.
- `pool_eval.margin_pick(pred: ndarray[N, M], margin: float) -> ndarray[N]` becomes `routing_decision(pred, model_costs=np.arange(M, dtype=float), margin=margin or tiny)`.

- [ ] **Step 1: Failing/characterization tests**

```python
def test_neighbor_quality_weights_by_similarity():
    sims = np.array([0.9, 0.6, 0.1]); scores = np.array([[1.0], [0.0], [0.5]])
    q, n, top = neighbor_quality(sims, scores, k=2, min_similarity=0.5, min_neighbors=2)
    assert q[0] == pytest.approx((0.9 * 1.0 + 0.6 * 0.0) / 1.5) and n == 2 and top == pytest.approx(0.9)

def test_neighbor_quality_none_below_min_neighbors():
    assert neighbor_quality(np.array([0.9, 0.2]), np.array([[1.0], [0.0]]), k=2, min_similarity=0.5, min_neighbors=2) is None

def test_neighbor_quality_nan_for_models_with_too_few_grades():
    sims = np.array([0.9, 0.8]); scores = np.array([[1.0, np.nan], [0.0, np.nan]])
    q, *_ = neighbor_quality(sims, scores, k=2, min_similarity=0.5, min_neighbors=2)
    assert q[0] == pytest.approx(0.5714285) and np.isnan(q[1])

def test_margin_pick_matches_the_old_rule():
    pred = np.array([[0.5, 0.52, 0.9], [0.8, 0.8, 0.8]])
    assert list(margin_pick(pred, 0.05)) == [2, 0]
```

Before Step 3, capture today's outputs: `graded_predictor` on the existing `test_graded.py` fixtures and `loo_knn` on a seeded 20×3 random matrix; save them as golden arrays in the tests so refactored output is compared with the old, not just with the new formulas.

- [ ] **Step 2:** Run; expect FAIL on the `neighbor_quality` tests; the golden and `margin_pick` tests pass on current code.
- [ ] **Step 3:** Implement and rewire. The one intended numeric difference is `graded_predictor` clipping weights at `1e-3`: with `min_similarity >= 0.5` that is a no-op, so the goldens must still match to 1e-9; if they do not, stop and report.
- [ ] **Step 4:** Full suites. Expected: pass.
- [ ] **Step 5: Commit** `refactor(gateway): the evaluation measures the production kNN`.

---

### Task 17: Final verification and notes

**Files:** `code_review/gateway_cleanup.md` (append a "Done / deferred" section), memory entry.

- [ ] **Step 1:** Run the full suite `python -m pytest tests -q -x` (not just gateway). Expected: pass with the Task 0 count plus new tests. Run `ruff check src/router/gateway src/evaluation/gateway` if the repo configures ruff (pyproject `[tool.ruff]`); fix unused imports left by the moves.
- [ ] **Step 2:** Count: `wc -l src/router/gateway/*.py` — report before/after totals honestly (the spec predicted roughly 400-500 lines out of 5,300, with the rest of the win in single-sourced rules).
- [ ] **Step 3:** Append to `code_review/gateway_cleanup.md` which findings were fixed and which stay deferred (below). Add a project memory `gateway-cleanup.md` with the branch name, the finding list that remains, and a line in `MEMORY.md`.
- [ ] **Step 4:** Use `superpowers:finishing-a-development-branch`.

## Deferred (not in this plan, with reason)

- **D6** mtime-reload cache (3 sites of different shape; a helper saves ~10 lines and needs a design decision).
- **D23** `CONFIG_TEMPLATE` vs `gateway.example.yaml`.
- **S1 part b** moving dashboard preferences out of `interceptor.py`: it entangles with 12 module globals; do it with the "one state object" change (collapse_scan B1), which needs a class and therefore a `gating-new-classes` review.
- **S4** splitting `cli.py` into a package (Task 5 and 13 remove its duplication first; the split then is a pure move).
- **S7** expressing `request_cost` through `cost_usd`; **X7** function attributes on the picker; **X8** `role_decision` with dummy arguments; **X11** private path functions.
- **S8** the `proxy` stand-in policy question: your decision, not a cleanup.
