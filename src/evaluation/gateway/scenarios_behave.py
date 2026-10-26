"""The gateway's behaviour, end to end, at $0: scripted request sequences
through an isolated gateway (its own home, port and log; the live config's
router) whose upstream is the recorder stub. The live gateway on :8787 is
never sent anything, and :func:`run_suite` checks that afterwards.

Each scenario is a contract: preconditions (unmet -> ``skip``, never a pass),
expectations, and ``forbid`` checks for what must not happen. Mechanics
(turn affinity, threads, fallback, isolation) set up their own state through
the dashboard's routing modes, so they don't depend on what the router
predicts; the two policy scenarios read the request's own setting (margin,
routing mode) and assert against it, never "hard means Opus".

Wave 1 is B01-B08; wave 2 (B09-B12) covers retries, session isolation,
unknown threads and concurrency.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from aar_gateway.config import read_jsonl
from aar_gateway.interceptor import CORRECTION_HOLD_TURNS, MARGIN_HEADER

from evaluation.gateway import router_bench as rb
from evaluation.gateway import router_compare as rc
from evaluation.gateway.pool_eval import TIERS

HAIKU, SONNET, OPUS = TIERS
LIVE_LOG = Path.home() / ".aar-gateway" / "requests.jsonl"
TOOL = {"name": "Read", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
#: hand-chosen fixtures: the scenarios test the gateway, not the generator's labels
EASY = "Rename the variable `tmp` to `total` in this function:\n\ndef add(a, b):\n    tmp = a + b\n    return tmp"
HARD = ("Design a lock-free bounded multi-producer multi-consumer queue in Rust, give the memory ordering of "
        "every atomic operation with the reason for it, and argue that it is linearizable.")
CORRECTION = "That's wrong, try again."


# -- talking to one isolated gateway --------------------------------------------------------------------


def _hex(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def new_session(gw: dict, label: str, model: str = OPUS) -> dict:
    """A conversation in its own Claude Code session. Its id starts ``scn`` so the
    log's 12-character ``claude_session`` names it."""
    h = _hex(gw["scenario"], label, gw["run"])
    sid = f"scn{h[:9]}-{h[9:13]}-{h[13:17]}-{h[17:21]}-{h[21:33]}"
    gw["sids"].add(sid[:12])
    return {"gw": gw, "sid": sid, "model": model, "conv": [], "step": 0}


def _post(s: dict, body: dict, *, margin: float | None = None) -> dict:
    """One request; returns the status, the reply, this request's log row and its upstream calls."""
    gw = s["gw"]
    s["step"] += 1
    bid = _hex(s["sid"], str(s["step"]))[:16]
    body = {"max_tokens": 64, "stream": False, "tools": [TOOL],
            "system": [{"type": "text", "text": f"aar-bench-id: {bid}"}],
            "metadata": {"user_id": f"user_aar_scn_session_{s['sid']}"}, **body}
    headers = {"authorization": "Bearer aar-bench-dummy", "x-claude-code-session-id": s["sid"],
               "anthropic-version": "2023-06-01"}
    if margin is not None:
        headers[MARGIN_HEADER] = str(margin)
    seq = rc.call_count()
    rows_before = len(_rows(gw, s["sid"]))
    resp = gw["http"].post(f"{gw['base']}/v1/messages", json=body, headers=headers)
    try:
        reply = resp.json()
    except ValueError:
        reply = {}
    rows = _rows(gw, s["sid"])
    deadline = time.monotonic() + 5
    while len(rows) <= rows_before and time.monotonic() < deadline:  # the row is written as the reply ends
        time.sleep(0.05)
        rows = _rows(gw, s["sid"])
    return {"status": resp.status_code, "reply": reply, "row": rows[-1] if len(rows) > rows_before else {},
            "calls": [c for c in rc.calls_since(seq) if c["bench_id"] == bid and c["kind"] != "count_tokens"]}


def say(s: dict, text: str, *, margin: float | None = None, thread: dict | None = None) -> dict:
    """The user types ``text``: a new turn on top of the conversation so far."""
    s["conv"].append({"role": "user", "content": [{"type": "text", "text": text}]})
    out = _post(s, {"model": s["model"], "messages": list(s["conv"]), **({"thread": thread} if thread else {})},
                margin=margin)
    s["conv"].append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{s['step']}",
                                                        "name": "Read", "input": {"path": "a.py"}}]})
    return out


def tool_result(s: dict) -> dict:
    """The client answers the model's tool call: the same turn continues."""
    s["conv"].append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{s['step']}",
                                                   "content": "def add(a, b): ..."}]})
    out = _post(s, {"model": s["model"], "messages": list(s["conv"])})
    s["conv"].append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{s['step']}",
                                                        "name": "Read", "input": {"path": "b.py"}}]})
    return out


def thread_continue(s: dict, previous_id: str, text: str) -> dict:
    """A server-side thread request: only the new message, continuing ``previous_id``."""
    return _post(s, {"model": s["model"], "thread": {"previous_message_id": previous_id},
                     "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]})


def mode(gw: dict, routing: str, pinned: str | None = None) -> None:
    body = {"mode": routing, **({"pinned_model": pinned} if pinned else {})}
    gw["http"].post(f"{gw['base']}/api/preferences", json=body).raise_for_status()


def _rows(gw: dict, sid: str) -> list[dict]:
    return [r for r in read_jsonl(gw["log"]) if r.get("claude_session") == sid[:12]]


# -- checks ---------------------------------------------------------------------------------------------


def _check(checks: list, name: str, ok: bool, detail="", *, kind: str = "expect") -> bool:
    checks.append({"name": name, "kind": kind, "ok": bool(ok), "detail": str(detail)[:240]})
    return bool(ok)


def pre(checks: list, name: str, ok: bool, detail="") -> bool:
    return _check(checks, name, ok, detail, kind="pre")


def forbid(checks: list, name: str, happened: bool, detail="") -> bool:
    return _check(checks, name, not happened, detail, kind="forbid")


def _models(out: dict) -> list[str]:
    return [c["model"] for c in out["calls"]]


def _pinned_turn(gw: dict, checks: list, label: str, text: str = EASY, **kw) -> tuple[dict, dict]:
    """A session whose first turn ran on Haiku (pinned), so its state names a cheaper model."""
    s = new_session(gw, label)
    mode(gw, "pin", HAIKU)
    first = say(s, text, **kw)
    pre(checks, f"{label}: first turn ran on {HAIKU}", first["row"].get("routed_model") == HAIKU, first["row"])
    return s, first


# -- scenarios ------------------------------------------------------------------------------------------


def b01_margin_one_takes_a_cheaper_model(gw: dict, checks: list) -> str | None:
    pre(checks, "router loaded", gw["health"].get("router", {}).get("enabled"), gw["health"].get("router"))
    out = say(new_session(gw, "a"), EASY, margin=1.0)
    _check(checks, "answered", out["status"] == 200, out["status"])
    routed = out["row"].get("routed_model")
    _check(checks, f"margin 1.0 routes below {OPUS}", routed in (HAIKU, SONNET), out["row"].get("reason"))
    forbid(checks, "fell back", out["row"].get("fallback"), out["row"].get("error"))
    return None


def b02_routing_off_keeps_the_asked_model(gw: dict, checks: list) -> str | None:
    mode(gw, "off")
    s = new_session(gw, "a")
    outs = [say(s, HARD), say(s, EASY, margin=1.0)]
    for i, out in enumerate(outs):
        _check(checks, f"turn {i + 1} on {OPUS}", out["row"].get("routed_model") == OPUS, out["row"].get("reason"))
        _check(checks, f"turn {i + 1} says routing off", "routing off" in (out["row"].get("reason") or ""),
               out["row"].get("reason"))
        forbid(checks, f"turn {i + 1} reached another model upstream", set(_models(out)) - {OPUS}, _models(out))
    return None


def b03_tool_loop_stays_on_the_turns_model(gw: dict, checks: list) -> str | None:
    s, _ = _pinned_turn(gw, checks, "a")
    mode(gw, "off")  # a mode change mid-turn must not move the turn
    for i in range(3):
        out = tool_result(s)
        _check(checks, f"tool result {i + 1} on {HAIKU}", out["row"].get("routed_model") == HAIKU, out["row"])
        _check(checks, f"tool result {i + 1} is the same turn",
               out["row"].get("reason") == "same turn as the previous request", out["row"].get("reason"))
        forbid(checks, f"tool result {i + 1} reached {OPUS}", OPUS in _models(out), _models(out))
    return None


def b04_thread_continues_on_its_model(gw: dict, checks: list) -> str | None:
    s, first = _pinned_turn(gw, checks, "a", thread={"type": "create"})
    pre(checks, "thread created", first["row"].get("thread") == "create", first["row"].get("thread"))
    msg_id = first["reply"].get("id")
    pre(checks, "reply carries a message id", bool(msg_id), first["reply"])
    mode(gw, "off")
    out = thread_continue(s, msg_id, "Now add a docstring.")
    _check(checks, "logged as a thread continue", out["row"].get("thread") == "continue", out["row"].get("thread"))
    _check(checks, f"stays on {HAIKU}", out["row"].get("routed_model") == HAIKU, out["row"].get("reason"))
    _check(checks, "reason names the thread", "server-side thread" in (out["row"].get("reason") or ""),
           out["row"].get("reason"))
    forbid(checks, f"continued on {OPUS}", OPUS in _models(out), _models(out))
    return None


def b05_correction_holds_then_expires(gw: dict, checks: list) -> str | None:
    s, first = _pinned_turn(gw, checks, "a")
    forbid(checks, "first turn flagged as a correction", first["row"].get("correction"))
    mode(gw, "auto")
    fix = say(s, CORRECTION)
    _check(checks, "correction flagged", fix["row"].get("correction") is True, fix["row"])
    _check(checks, f"correction answered on {OPUS}", fix["row"].get("routed_model") == OPUS, fix["row"].get("reason"))
    held = 1
    for i in range(CORRECTION_HOLD_TURNS - 1):
        out = say(s, f"Next, handle the empty-input case number {i + 1} in add().", margin=1.0)
        if "after a correction" in (out["row"].get("reason") or "") and out["row"].get("routed_model") == OPUS:
            held += 1
    _check(checks, f"held for {CORRECTION_HOLD_TURNS} turns", held == CORRECTION_HOLD_TURNS, f"held {held}")
    after = say(s, "Now add type hints to add().", margin=1.0)
    forbid(checks, "still held after the hold ran out", "after a correction" in (after["row"].get("reason") or ""),
           after["row"].get("reason"))
    return None


def b06_resume_routes_against_the_first_asked_model(gw: dict, checks: list) -> str | None:
    def route(sid: str, model: str, **extra) -> dict:
        r = gw["http"].post(f"{gw['base']}/api/route", json={"text": "rename x", "model": model, "session_id": sid,
                                                            **extra})
        r.raise_for_status()
        return r.json()

    def mod_record(sid: str) -> None:
        gw["http"].post(f"{gw['base']}/api/mod-record", json={
            "session_id": sid, "requested_model": OPUS, "routed_model": HAIKU, "new_turn": True,
            "input_tokens": 5, "output_tokens": 50, "cache_read_input_tokens": 9000,
            "cache_creation_input_tokens": 100}).raise_for_status()

    a, b = new_session(gw, "resumed")["sid"], new_session(gw, "model-choice")["sid"]
    for sid in (a, b):
        route(sid, OPUS)
        mod_record(sid)
    resumed = route(a, HAIKU, resumed=True)
    _check(checks, f"a resumed session is routed against {OPUS}", resumed.get("requested") == OPUS, resumed)
    chosen = route(b, HAIKU)
    forbid(checks, f"a /model choice of {HAIKU} (not a resume) rewritten to {OPUS}", chosen.get("requested") == OPUS,
           chosen)
    return None


def b07_a_failed_reroute_falls_back(gw: dict, checks: list) -> str | None:
    s = new_session(gw, "a")
    mode(gw, "pin", HAIKU)
    with rc.failing({HAIKU: 529}):
        out = say(s, EASY)
    _check(checks, "answered", out["status"] == 200, out["status"])
    _check(checks, "fallback logged", out["row"].get("fallback") is True, out["row"])
    _check(checks, f"attempted {HAIKU}", out["row"].get("attempted_model") == HAIKU, out["row"].get("attempted_model"))
    _check(checks, f"served by {OPUS}", out["row"].get("routed_model") == OPUS, out["row"].get("routed_model"))
    _check(checks, f"upstream saw {HAIKU} then {OPUS}", _models(out) == [HAIKU, OPUS], _models(out))
    forbid(checks, "more than two upstream attempts", len(out["calls"]) > 2, _models(out))
    return None


def b08_all_upstreams_fail_bounded(gw: dict, checks: list) -> str | None:
    s = new_session(gw, "a")
    mode(gw, "pin", HAIKU)
    with rc.failing({HAIKU: 529, OPUS: 529}):
        t0 = time.monotonic()
        out = say(s, EASY)
        took = time.monotonic() - t0
    _check(checks, "an error reaches the client", out["status"] >= 400, out["status"])
    _check(checks, "error is Anthropic-shaped", out["reply"].get("type") == "error", out["reply"])
    forbid(checks, "more than two upstream attempts", len(out["calls"]) > 2, _models(out))
    forbid(checks, "took over 30 s", took > 30, f"{took:.1f}s")
    _check(checks, "logged with the error status", (out["row"].get("status") or 0) >= 400, out["row"].get("status"))
    return None


def b09_client_retry_is_documented(gw: dict, checks: list) -> str | None:
    """The gateway has no idempotency key: a retried request is a second request.
    Recorded, not asserted, so a change in behaviour shows up in the report."""
    mode(gw, "off")
    s = new_session(gw, "a")
    s["conv"].append({"role": "user", "content": [{"type": "text", "text": EASY}]})
    body = {"model": OPUS, "messages": list(s["conv"])}
    first = _post(s, body)
    s["step"] -= 1  # the same bench id: byte-identical retry
    again = _post(s, body)
    _check(checks, "both answered", first["status"] == again["status"] == 200, (first["status"], again["status"]))
    return (f"retry: {len(first['calls']) + len(again['calls'])} upstream call(s), "
            f"{len(_rows(gw, s['sid']))} log row(s); the second row's reason: {again['row'].get('reason')!r}")


def b10_sessions_do_not_share_state(gw: dict, checks: list) -> str | None:
    a, _ = _pinned_turn(gw, checks, "a")
    mode(gw, "auto")
    fix = say(a, CORRECTION)
    pre(checks, "session A is held", fix["row"].get("correction") is True, fix["row"])
    b = say(new_session(gw, "b"), EASY, margin=1.0)
    forbid(checks, "session B inherits A's correction hold", "after a correction" in (b["row"].get("reason") or "")
           or b["row"].get("correction"), b["row"].get("reason"))
    forbid(checks, "session B continues A's turn", b["row"].get("reason") == "same turn as the previous request",
           b["row"].get("reason"))
    again = say(a, "Next, handle negative numbers in add().", margin=1.0)
    _check(checks, "session A is still held", "after a correction" in (again["row"].get("reason") or ""),
           again["row"].get("reason"))
    return None


def b11_unknown_thread_fails_safe(gw: dict, checks: list) -> str | None:
    _pinned_turn(gw, checks, "a", thread={"type": "create"})  # another session's thread lives on Haiku
    mode(gw, "off")
    out = thread_continue(new_session(gw, "b"), "msg_aar_bench_unknown", "Continue from there.")
    _check(checks, "answered", out["status"] == 200, out["status"])
    _check(checks, f"kept on the asked model {OPUS}", out["row"].get("routed_model") == OPUS, out["row"].get("reason"))
    forbid(checks, "moved to another session's thread model", HAIKU in _models(out), _models(out))
    return None


def b12_concurrent_sessions_keep_their_models(gw: dict, checks: list) -> str | None:
    a, _ = _pinned_turn(gw, checks, "a")
    mode(gw, "off")
    b = new_session(gw, "b")
    first_b = say(b, HARD)
    pre(checks, f"session B's turn on {OPUS}", first_b["row"].get("routed_model") == OPUS, first_b["row"])

    def loop(s: dict) -> list[dict]:
        return [tool_result(s) for _ in range(5)]

    with ThreadPoolExecutor(2) as pool:
        got_a, got_b = pool.map(loop, (a, b))
    wrong_a = [o["row"].get("routed_model") for o in got_a if o["row"].get("routed_model") != HAIKU]
    wrong_b = [o["row"].get("routed_model") for o in got_b if o["row"].get("routed_model") != OPUS]
    _check(checks, f"session A's 5 tool results on {HAIKU}", not wrong_a, wrong_a)
    _check(checks, f"session B's 5 tool results on {OPUS}", not wrong_b, wrong_b)
    forbid(checks, "a request answered with an error", any(o["status"] != 200 for o in [*got_a, *got_b]),
           [o["status"] for o in [*got_a, *got_b]])
    return None


#: id, wave, kind (mechanics | policy | observe), the scenario
SCENARIOS: list[tuple[str, int, str, Callable]] = [
    ("B01", 1, "policy", b01_margin_one_takes_a_cheaper_model),
    ("B02", 1, "policy", b02_routing_off_keeps_the_asked_model),
    ("B03", 1, "mechanics", b03_tool_loop_stays_on_the_turns_model),
    ("B04", 1, "mechanics", b04_thread_continues_on_its_model),
    ("B05", 1, "mechanics", b05_correction_holds_then_expires),
    ("B06", 1, "mechanics", b06_resume_routes_against_the_first_asked_model),
    ("B07", 1, "mechanics", b07_a_failed_reroute_falls_back),
    ("B08", 1, "mechanics", b08_all_upstreams_fail_bounded),
    ("B09", 2, "observe", b09_client_retry_is_documented),
    ("B10", 2, "mechanics", b10_sessions_do_not_share_state),
    ("B11", 2, "mechanics", b11_unknown_thread_fails_safe),
    ("B12", 2, "mechanics", b12_concurrent_sessions_keep_their_models),
]


def status_of(checks: list, kind: str) -> str:
    if any(not c["ok"] for c in checks if c["kind"] == "pre"):
        return "skip"
    if any(not c["ok"] for c in checks):
        return "fail"
    return "info" if kind == "observe" else "pass"


def run_scenario(gw: dict, sid: str, kind: str, fn: Callable) -> dict:
    checks: list = []
    note = None
    try:
        note = fn(gw, checks)
    except Exception as exc:  # noqa: BLE001 -- one broken scenario is a failure, not the end of the suite
        _check(checks, "ran without an exception", False, f"{type(exc).__name__}: {exc}")
    return {"id": sid, "kind": kind, "title": fn.__name__.partition("_")[2].replace("_", " ") or fn.__name__,
            "status": status_of(checks, kind), "checks": checks, "note": note}


def isolation(live_log: Path, offset: int, sids: set[str]) -> dict:
    """Whether any scenario session reached the live gateway's log since ``offset``."""
    leaked = []
    try:
        with live_log.open("rb") as f:
            f.seek(offset)
            for line in f.read().decode("utf-8", errors="replace").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("claude_session") in sids:
                    leaked.append(rec["claude_session"])
    except OSError:
        pass
    return {"ok": not leaked, "detail": f"{len(leaked)} live log row(s) from scenario sessions" if leaked
            else f"no scenario session in {live_log} ({len(sids)} session ids checked)"}


def run_suite(out_dir: Path, waves: tuple[int, ...] = (1, 2), only: tuple[str, ...] = (),
              live_log: Path = LIVE_LOG) -> dict:
    """Every scenario of ``waves`` in a fresh isolated gateway; writes ``behave.json``."""
    run = time.strftime("%Y%m%d-%H%M%S")
    root = out_dir / "behave-run" / run
    offset = live_log.stat().st_size if live_log.exists() else 0
    sids: set[str] = set()
    results = []
    with rc.recorder() as upstream, httpx.Client(timeout=60) as http:
        for sid, wave, kind, fn in SCENARIOS:
            if wave not in waves or (only and sid not in only):
                continue
            run_dir = root / sid
            spec = rb.aar_spec(run_dir, upstream, "aar")
            try:
                with rc.running(f"aar-{sid}", spec["cmd"], env=spec["env"], cwd=spec["cwd"],
                                health=spec["health"], log_dir=run_dir):
                    gw = {"base": spec["base"], "http": http, "scenario": sid, "run": run, "sids": sids,
                          "log": Path(spec["env"]["AAR_GATEWAY_LOG"]),
                          "health": http.get(spec["health"]).json()}
                    results.append(run_scenario(gw, sid, kind, fn))
            except RuntimeError as exc:  # the gateway didn't start
                results.append({"id": sid, "kind": kind, "title": fn.__name__, "status": "fail", "note": None,
                                "checks": [{"name": "gateway started", "kind": "expect", "ok": False,
                                            "detail": str(exc)[:240]}]})
            print(f"  {sid} {results[-1]['status']}", flush=True)
    out = {"run": run, "waves": list(waves), "results": results, "isolation": isolation(live_log, offset, sids)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "behave.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out
