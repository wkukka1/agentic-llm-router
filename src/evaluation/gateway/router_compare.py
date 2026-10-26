"""Installed Claude Code routers against aar-gateway, on graded prompts, for $0.

A zero-cost, stub-backed routing-policy bake-off: each router runs as its own
process with its upstream pointed at a local recorder stub, so nothing leaves
the machine. Every graded prompt is sent through each router in a Claude Code
request; the recorder sees which model the router forwarded to, and that tier's
graded score and usage (``pool_eval``) are what the decision is worth.

It measures three things, reported apart:

* **routing decision** -- the tier read from the router's one upstream call;
* **protocol compatibility** -- whether the router took the request and gave back
  a reply Claude Code can read;
* **estimated outcome** -- the graded score and usage of the selected tier, from
  the grading replays (historical usage at API prices; this run spends nothing).

The stub's replies are canned, so the answers themselves are never judged: this
is not an end-to-end quality benchmark.

    python -m evaluation.gateway.router_compare capture
    python -m evaluation.gateway.router_compare run --graded ~/.aar-gateway/eval-rules/graded.jsonl \\
        --graded ~/.aar-gateway/eval40/graded.jsonl

Router launchers: :mod:`evaluation.gateway.router_bench`; installs and config templates:
``scripts/router_bench/``.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np

from evaluation.gateway.pool_eval import TIERS, evaluate

BENCH_HOME = Path(os.environ.get("AAR_ROUTER_BENCH", Path.home() / ".aar-router-bench"))
RECORDER_HOST, RECORDER_PORT = "127.0.0.1", 8899
RECORDER_URL = f"http://{RECORDER_HOST}:{RECORDER_PORT}"
#: the template's user text, replaced by each graded prompt
TEMPLATE_PROMPT = "hi"
#: a request's id rides in its own system block, which every router forwards (headers often don't survive)
BENCH_ID = re.compile(rb"aar-bench-id: ([0-9a-f]{16})")
OPUS = len(TIERS) - 1

#: exact upstream ids -> tier index. Anything else is ``unknown_model``, never a default tier.
TIER_OF: dict[str, int] = {}
for _i, _t in enumerate(TIERS):
    TIER_OF.update({_t: _i, f"anthropic/{_t}": _i})


# -- the recorder: an Anthropic + OpenAI-shaped stub that logs where each call went ------------------

#: one entry per upstream call: endpoint, model, stream flag, bench id. Never a body or prompt text.
_calls: list[dict] = []
_calls_lock = threading.Lock()
#: ``{"path": Path}`` while capturing a template; the first tool-carrying request body is written there
_capture: dict = {}
#: model -> HTTP status the stub answers it with instead of a reply; set only inside :func:`failing`
_status: dict[str, int] = {}


@contextlib.contextmanager
def failing(statuses: Mapping[str, int]) -> Iterator[None]:
    """For the length of the block, the stub answers each named model with its status
    (an Anthropic error body), e.g. ``{"claude-haiku-5-5": 529}``; cleared on exit."""
    with _calls_lock:
        _status.update(statuses)
    try:
        yield
    finally:
        with _calls_lock:
            _status.clear()


def calls_since(seq: int) -> list[dict]:
    with _calls_lock:
        return [dict(c) for c in _calls[seq:]]


def call_count() -> int:
    with _calls_lock:
        return len(_calls)


def _anthropic_reply(model: str, stream: bool, seq: int = 0) -> tuple[str, bytes]:
    message = {"id": f"msg_aar_bench_{seq}","type": "message", "role": "assistant", "model": model,
               "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
               "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}
    if not stream:
        return "application/json", json.dumps(message).encode()
    start = {**message, "content": [], "stop_reason": None, "usage": {"input_tokens": 1, "output_tokens": 0}}
    events = [("message_start", {"type": "message_start", "message": start}),
              ("content_block_start", {"type": "content_block_start", "index": 0,
                                       "content_block": {"type": "text", "text": ""}}),
              ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                       "delta": {"type": "text_delta", "text": "ok"}}),
              ("content_block_stop", {"type": "content_block_stop", "index": 0}),
              ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                 "usage": {"output_tokens": 1}}),
              ("message_stop", {"type": "message_stop"})]
    return "text/event-stream", "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events).encode()


def _openai_reply(model: str, stream: bool) -> tuple[str, bytes]:
    if not stream:
        return "application/json", json.dumps({
            "id": "chatcmpl-aar-bench", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
    chunk = {"id": "chatcmpl-aar-bench", "object": "chat.completion.chunk", "created": 0, "model": model}
    parts = [{**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
             {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}]
    return "text/event-stream", ("".join(f"data: {json.dumps(p)}\n\n" for p in parts) + "data: [DONE]\n\n").encode()


def record_call(path: str, raw: bytes, headers: Mapping[str, str]) -> tuple[int, str, bytes]:
    """Log one upstream call and answer it: ``(status, content type, body)``."""
    try:
        body = json.loads(raw) if raw else {}
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    found = BENCH_ID.search(raw)
    count_only = path.rstrip("/").endswith("count_tokens")
    openai = path.rstrip("/").endswith("chat/completions")
    model = str(body.get("model") or "")
    stream = bool(body.get("stream"))
    if "path" in _capture and not count_only and not openai and body.get("tools") and not _capture.get("done"):
        keep = {k: v for k, v in headers.items() if k.lower() in ("anthropic-beta", "anthropic-version", "user-agent")}
        Path(_capture["path"]).write_text(json.dumps({"body": body, "headers": keep}, indent=1), encoding="utf-8")
        _capture["done"] = True
    with _calls_lock:
        seq = len(_calls)
        status = 200 if count_only else _status.get(model, 200)
        _calls.append({"seq": seq, "endpoint": path, "model": model, "stream": stream,
                       "kind": "count_tokens" if count_only else "openai" if openai else "anthropic",
                       "bench_id": found.group(1).decode() if found else None, "status": status})
    if count_only:
        return 200, "application/json", json.dumps({"input_tokens": max(1, len(raw) // 4)}).encode()
    if status != 200:
        return status, "application/json", json.dumps({"type": "error", "error": {
            "type": "overloaded_error" if status == 529 else "api_error", "message": f"stub: {model} failing"}}).encode()
    kind, out = _openai_reply(model, stream) if openai else _anthropic_reply(model, stream, seq)
    return 200, kind, out


def recorder_app():
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    async def upstream(request: Request) -> Response:
        status, kind, out = record_call(request.url.path, await request.body(), dict(request.headers))
        return Response(out, status_code=status, media_type=kind)

    async def models(_request: Request) -> Response:
        return JSONResponse({"object": "list", "data": [{"id": t, "object": "model", "type": "model"} for t in TIERS]})

    return Starlette(routes=[Route("/{path:path}", upstream, methods=["POST"]),
                             Route("/{path:path}", models, methods=["GET"])])


@contextlib.contextmanager
def recorder(capture: Path | None = None) -> Iterator[str]:
    """The stub on :data:`RECORDER_PORT`, in a thread, for the length of the block."""
    import uvicorn

    _capture.clear()
    if capture is not None:
        _capture["path"] = capture
    server = uvicorn.Server(uvicorn.Config(recorder_app(), host=RECORDER_HOST, port=RECORDER_PORT,
                                           log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError(f"recorder did not start on {RECORDER_URL}")
        time.sleep(0.05)
    try:
        yield RECORDER_URL
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        _capture.clear()


# -- requests -----------------------------------------------------------------------------------------


def bench_id(router: str, track: str, condition: str, prompt_hash: str) -> str:
    return hashlib.sha256(f"{router}|{track}|{condition}|{prompt_hash}".encode()).hexdigest()[:16]


def _session_uuid(bid: str) -> str:
    h = hashlib.sha256(bid.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def load_template(path: Path) -> dict:
    """A captured Claude Code request: ``{"body": ..., "headers": ...}``."""
    template = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(template.get("body"), dict) or not template["body"].get("tools"):
        raise ValueError(f"{path}: not a captured Claude Code request (no tools)")
    return template


def build_request(template: Mapping, prompt: str, *, model: str, bid: str, thinking: bool = True) -> dict:
    """The captured Claude Code request with ``prompt`` as the user's text, ``model`` as the model it
    asks for, and a fresh session per request (no router carries cache or turn state across prompts).
    ``thinking=False`` drops the thinking setting (the CCR "think" rule reads it)."""
    body = copy.deepcopy(dict(template["body"]))
    body["model"] = model
    if not thinking:
        body.pop("thinking", None)
    messages = body.get("messages") or []
    users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not users:
        raise ValueError("template has no user message")
    # every other message stays as captured (newer Claude Code appends a ``system`` message)
    message = messages[users[-1]]
    content = message.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    replaced = False
    for block in content:
        if block.get("type") == "text" and block.get("text", "").strip() == TEMPLATE_PROMPT:
            block["text"], replaced = prompt, True
    if not replaced:
        raise ValueError(f"template's user message has no {TEMPLATE_PROMPT!r} text block")
    for block in content:
        block.pop("cache_control", None)
    message["content"] = content
    system = body.get("system") or []
    if isinstance(system, str):
        system = [{"type": "text", "text": system}]
    body["system"] = [*system, {"type": "text", "text": f"aar-bench-id: {bid}"}]
    sid = _session_uuid(bid)
    meta = dict(body.get("metadata") or {})
    user = str(meta.get("user_id") or "")
    if user.startswith("{"):  # newer Claude Code: a JSON object with a session_id field
        try:
            parsed = json.loads(user)
            parsed["session_id"] = sid
            meta["user_id"] = json.dumps(parsed)
        except ValueError:
            meta["user_id"] = f"user_aar_bench_session_{sid}"
    else:
        meta["user_id"] = (user.split("_session_", 1)[0] or "user_aar_bench") + f"_session_{sid}"
    body["metadata"] = meta
    return body


def native_request(template: Mapping, prompt: str, *, model: str, bid: str, thinking: bool = True) -> dict:
    """The same turn as a plain Messages API request, GA fields only: the template's system text and
    custom tools, one user message, standard ``enabled`` thinking. What a router built on the public
    API is written against; :func:`build_request` is what Claude Code sends today."""
    src = template["body"]
    system = src.get("system") or []
    if isinstance(system, str):
        system = [{"type": "text", "text": system}]
    body = {"model": model, "max_tokens": 32000, "stream": True,
            "system": [{"type": "text", "text": s["text"]} for s in system if s.get("type") == "text" and s.get("text")]
            + [{"type": "text", "text": f"aar-bench-id: {bid}"}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "tools": [{k: t[k] for k in ("name", "description", "input_schema") if k in t}
                      for t in src.get("tools") or [] if t.get("input_schema")],
            "metadata": {"user_id": f"user_aar_bench_session_{_session_uuid(bid)}"}}
    if thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": 16000}
    return body


def request_headers(template: Mapping, bid: str) -> dict[str, str]:
    """Claude Code's version/beta headers from the capture; a dummy key (the stub checks none)."""
    kept = {k.lower(): v for k, v in (template.get("headers") or {}).items()}
    return {"content-type": "application/json", "anthropic-version": kept.get("anthropic-version", "2023-06-01"),
            **({"anthropic-beta": kept["anthropic-beta"]} if kept.get("anthropic-beta") else {}),
            "user-agent": kept.get("user-agent", "claude-cli/aar-bench"),
            "x-api-key": "aar-bench-dummy", "authorization": "Bearer aar-bench-dummy",
            "x-claude-code-session-id": _session_uuid(bid), "x-aar-bench-id": bid}


def openai_request(prompt: str, *, model: str, bid: str) -> dict:
    return {"model": model, "stream": False,
            "messages": [{"role": "system", "content": f"aar-bench-id: {bid}"}, {"role": "user", "content": prompt}]}


def reply_ok(kind: str, status: int, text: str) -> str | None:
    """``None`` when ``text`` is a reply a client of ``kind`` can read, else why not."""
    if status != 200:
        return f"http {status}"
    if kind == "anthropic":
        if "event: message_stop" in text or '"type":"message_stop"' in text.replace(" ", ""):
            return None
        try:
            return None if json.loads(text).get("type") == "message" else "not an Anthropic message"
        except ValueError:
            return "unreadable Anthropic reply"
    if "[DONE]" in text:
        return None
    try:
        return None if json.loads(text).get("choices") else "no choices in reply"
    except ValueError:
        return "unreadable OpenAI reply"


# -- attribution ----------------------------------------------------------------------------------------


def attribute(window: Sequence[Mapping], bid: str) -> dict:
    """The routing decision behind one request, from the upstream calls made while it was in flight
    (replay is sequential, so they are all its own). Exactly one selection call is a pick; none or
    several is an anomaly, never a pick."""
    sel = [c for c in window if c.get("kind") != "count_tokens"]
    out = {"upstream_calls": len(sel), "upstream_endpoints": sorted({c["endpoint"] for c in sel}),
           "upstream_models": [c["model"] for c in sel], "selected_model": None, "tier": None,
           "mapping_reason": None, "error": None}
    if not sel:
        out["error"] = "no_upstream_call"
        return out
    if len(sel) > 1:
        out["error"] = "multiple_upstream_calls"
        return out
    call = sel[0]
    out["selected_model"] = call["model"]
    if call["model"] not in TIER_OF:
        out["error"] = "unknown_model"
        return out
    out["tier"] = TIERS[TIER_OF[call["model"]]]
    out["mapping_reason"] = "bench id matched" if call.get("bench_id") == bid else "sequential window (bench id not seen)"
    return out


def send_one(client, url: str, body: Mapping, headers: Mapping[str, str], *, kind: str, bid: str,
             timeout: float = 120.0) -> dict:
    """Send one request through a router and attribute what it did upstream."""
    start = call_count()
    status, problem = None, None
    try:
        r = client.post(url, content=json.dumps(body), headers=dict(headers), timeout=timeout)
        status, problem = r.status_code, reply_ok(kind, r.status_code, r.text)
    except Exception as exc:  # noqa: BLE001 -- a router that hangs or drops the connection is a result
        problem = f"{type(exc).__name__}"
    time.sleep(0.05)  # a router may log its upstream call a moment after replying
    out = attribute(calls_since(start), bid)
    out.update(router_http_status=status, compat_error=problem)
    if problem and out["error"] is None and out["tier"] is None:
        out["error"] = "router_error"
    return out


# -- scoring --------------------------------------------------------------------------------------------


def picks_of(rows: Sequence[Mapping], prompts: Sequence[Mapping]) -> tuple[np.ndarray, np.ndarray]:
    """``(picks, failed)`` aligned with ``prompts``. A failed or missing decision is charged as
    Opus (the conservative cost) and flagged, so it stays in the denominator."""
    by = {r["prompt_hash"]: r for r in rows}
    picks = np.full(len(prompts), OPUS)
    failed = np.ones(len(prompts), dtype=bool)
    for i, p in enumerate(prompts):
        r = by.get(p["prompt_hash"])
        if r and r.get("tier") in TIERS and not r.get("error"):
            picks[i], failed[i] = TIERS.index(r["tier"]), False
    return picks, failed


def paired_bootstrap(prompts: Sequence[Mapping], a: np.ndarray, b: np.ndarray, *, n: int = 10_000,
                     seed: int = 0) -> dict:
    """95% intervals for ``a - b`` in mean quality and summed usage, resampling prompts in pairs."""
    S = np.stack([p["score"] for p in prompts])
    C = np.stack([p["cost"] for p in prompts])
    idx = np.arange(len(prompts))
    dq = S[idx, a] - S[idx, b]
    dc = C[idx, a] - C[idx, b]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(prompts), size=(n, len(prompts)))
    q, c = dq[draws].mean(axis=1), dc[draws].sum(axis=1)

    def ci(point, sample):
        lo, hi = np.percentile(sample, [2.5, 97.5])
        return {"delta": float(point), "lo": float(lo), "hi": float(hi),
                "verdict": "no detectable difference" if lo <= 0 <= hi else ("higher" if lo > 0 else "lower")}
    return {"quality": ci(dq.mean(), q), "est_cost_usd": ci(dc.sum(), c)}


def add_savings(row: dict, prompts: Sequence[Mapping], baseline: int) -> dict:
    """Savings against no router: every prompt served by the model Claude Code asked for
    (``baseline``), the gateway's own headline (``vs_no_router``). Cold cache on both sides."""
    S = np.stack([p["score"] for p in prompts])
    C = np.stack([p["cost"] for p in prompts])
    base_cost = float(C[:, baseline].sum())
    saved = base_cost - row["est_cost_usd"]
    row.update(no_router_usd=base_cost, saved_usd=saved, saved_pct=saved / base_cost if base_cost else 0.0,
               dq_vs_no_router=row["quality"] - float(S[:, baseline].mean()))
    return row


def score_rows(prompts: Sequence[Mapping], results: Mapping[str, Sequence[Mapping]], *,
               baseline: int = OPUS) -> list[dict]:
    """``evaluate`` per router, plus the columns it can't express: errors, unknown models, anomalies,
    and savings against the requested model (``baseline``, a tier index)."""
    picks, failed = {}, {}
    for name, rows in results.items():
        picks[name], failed[name] = picks_of(rows, prompts)
    out = []
    for r in evaluate(list(prompts), picks=picks):
        rows = results[r["policy"]]
        r.update(n_prompts=len(prompts), n_errors=int(failed[r["policy"]].sum()),
                 unknown_models=sum(1 for x in rows if x.get("error") == "unknown_model"),
                 anomalies=sum(1 for x in rows if x.get("error") in ("no_upstream_call", "multiple_upstream_calls")),
                 actual_opus_share=float(((picks[r["policy"]] == OPUS) & ~failed[r["policy"]]).mean()),
                 est_cost_usd=r.pop("usage_usd"))
        out.append(add_savings(r, prompts, baseline))
    return out


def table(rows: Sequence[Mapping], title: str) -> str:
    head = (f"{'router':40s} {'quality':>7s} {'vs mix':>7s} {'match':>6s} {'est $':>7s} {'saved $':>8s} "
            f"{'saved':>6s} {'dQ':>7s} {'H/S/O':>9s} {'err':>4s} {'unk':>4s} {'anom':>4s}")
    lines = [title, head]
    for r in rows:
        mix = r["mix"]
        hso = f"{mix.get('haiku', 0)}/{mix.get('sonnet', 0)}/{mix.get('opus', 0)}"
        lines.append(f"{r['policy'][:40]:40s} {r['quality']:7.3f} {r['vs_mix']:+7.3f} {r['matches_best']:6.0%} "
                     f"{r['est_cost_usd']:7.2f} {r['saved_usd']:+8.2f} {r['saved_pct']:6.0%} "
                     f"{r['dq_vs_no_router']:+7.3f} {hso:>9s} {r['n_errors']:4d} {r['unknown_models']:4d} "
                     f"{r['anomalies']:4d}")
    return "\n".join(lines)


# -- routers --------------------------------------------------------------------------------------------


@contextlib.contextmanager
def running(name: str, cmd: Sequence[str], *, env: Mapping[str, str], cwd: Path, health: str,
            log_dir: Path, ready_timeout: float = 180.0) -> Iterator[subprocess.Popen]:
    """Start one router, wait until ``health`` answers, and always stop it by the pid started here."""
    import httpx

    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / f"{name}.log", "wb")  # noqa: SIM115 -- closed in finally
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(list(cmd), cwd=cwd, env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, creationflags=flags)
    try:
        deadline = time.monotonic() + ready_timeout
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited with {proc.returncode} before it was ready (see {log.name})")
            try:
                if httpx.get(health, timeout=2).status_code < 500:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"{name} not ready after {ready_timeout:.0f}s")
            time.sleep(0.5)
        yield proc
    finally:
        if proc.poll() is None:
            if os.name == "nt":  # the whole tree this pid started (node and uv spawn children)
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
            else:
                proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=15)
        log.close()


def replay(name: str, base: str, path: str, prompts: Sequence[Mapping], template: Mapping, *,
           track: str, condition: str, model: str, thinking: bool = True, fmt: str = "anthropic",
           send_model: str | None = None, extra_headers: Mapping[str, str] | None = None) -> list[dict]:
    """Every prompt through one running router, one at a time. ``track`` ``compat`` sends the captured
    Claude Code request; ``native`` the plain Messages API one (:func:`native_request`)."""
    import httpx

    rows = []
    with httpx.Client() as client:
        for p in prompts:
            bid = bench_id(name, track, condition, p["prompt_hash"])
            if fmt == "openai":
                body, headers = openai_request(p["prompt"], model=send_model or model, bid=bid), \
                    {"content-type": "application/json", "authorization": "Bearer aar-bench-dummy"}
            else:
                shape = build_request if track == "compat" else native_request
                body = shape(template, p["prompt"], model=send_model or model, bid=bid, thinking=thinking)
                headers = request_headers(template, bid)
            got = send_one(client, base + path, body, {**headers, **(extra_headers or {})}, kind=fmt, bid=bid)
            rows.append({"router": name, "track": track, "condition": condition, "prompt_hash": p["prompt_hash"],
                         "bench_id": bid, "requested_model": model, **got})
    return rows


Launcher = Callable[[Path, str], dict]


def main(argv: list[str] | None = None) -> int:
    from evaluation.gateway import router_bench as launchers  # the router-specific half

    ap = argparse.ArgumentParser(prog="python -m evaluation.gateway.router_compare", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture", help="capture a Claude Code request template (sent to the stub only)")
    cap.add_argument("--out", default=str(BENCH_HOME / "template.json"))
    run = sub.add_parser("run", help="replay graded prompts through the routers")
    run.add_argument("--graded", action="append", required=True)
    run.add_argument("--routers", default=",".join(launchers.ROUTERS))
    run.add_argument("--template", default=str(BENCH_HOME / "template.json"))
    run.add_argument("--out", default=str(BENCH_HOME / "results"))
    run.add_argument("--sensitivity", action="store_true", help="also requested Sonnet/Haiku and no-thinking")
    run.add_argument("--smoke", action="store_true", help="also one real `claude -p` through each router (to the stub)")
    run.add_argument("--margins", default="", help="comma list: sweep aar-gateway's cost aversion (router.margin) "
                                                    "through the running gateways, e.g. 0,0.05,0.1,0.15")
    args = ap.parse_args(argv)
    if args.cmd == "capture":
        return launchers.capture_template(Path(args.out))
    return launchers.run_all(args)


if __name__ == "__main__":
    sys.exit(main())
