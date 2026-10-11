"""The interceptor: Claude Code's requests in, routed, and out to the right model.

Claude Code pointed at this gateway (``ANTHROPIC_BASE_URL``) sends its real
requests here -- its own login, every tool definition, thinking settings,
``stream: true``. For each ``/v1/messages`` request:

1. **Decide the model, once per human turn.** A request whose last user
   message carries new human text starts a turn: the picker
   (:mod:`router.gateway.routing`) may move it to another model. Requests
   inside the turn's tool loop (tool results only) stay on the turn's model,
   so the loop never changes model halfway through. A Claude Code subagent
   is a conversation of its own whose first message is its task; when its
   role is known (:mod:`router.gateway.subagents`) the role picks its tier.
2. **Send it down the model's lane.**
   ``anthropic``: forwarded to Anthropic close to untouched -- client's own
   credentials, tools, thinking, beta headers; only ``model`` may change.
   ``openai``: translated to Chat Completions with tools
   (:mod:`router.gateway.translate`), sent with that provider's key from the
   environment, translated back -- streaming included.
3. **Fail open.** If a rerouted request errors, it is retried on the model
   the client asked for; the router never leaves Claude Code worse off.
4. **Log it.** One JSON line per request: models, tokens, latency, observed
   cost, and two counterfactuals -- the cost on the model the client asked
   for, and on the strongest configured model -- with the token counts
   adjusted for what those models would have had cached. Never headers,
   credentials or prompt text.

These records are a stand-in for :class:`~router.tracing.traces.ExecutionTrace`,
which needs a full ``RoutingDecision``/``PromptSignals`` per request.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import load_settings
from .pricing import canonical_model, cost_usd
from .routing import ModelPicker, price_for, unavailable_reason
from .subagents import DEFAULT_ROLE_MODELS, role_decision, spawn_decision
from .subagents import detect as detect_subagent
from .translate import OpenAIStreamTranslator, anthropic_to_openai, iter_openai_sse, openai_to_anthropic

_REQUEST_HOP_HEADERS = {"host", "content-length", "connection", "accept-encoding",
                        "transfer-encoding", "keep-alive"}
#: httpx decodes the upstream body, so its encoding/length no longer apply
_RESPONSE_HOP_HEADERS = {"content-length", "content-encoding", "transfer-encoding", "connection",
                         "keep-alive"}
_MAX_SESSIONS = 2000

_settings: dict = load_settings(None)
_picker: ModelPicker | None = None
_client: httpx.AsyncClient | None = None
_sessions: OrderedDict[str, dict] = OrderedDict()
#: prefix signature (system prompt + tools) -> (last seen, fewest prompt tokens
#: seen with it): the shared part every conversation with that prefix reuses
_prefixes: dict[str, tuple[float, int]] = {}
#: (model, prefix signature) -> when that model last cached that prefix
_warm: dict[tuple[str, str], float] = {}
#: running per-request token profile of all agent traffic -- the default for a
#: conversation that has none of its own yet
_global_profile: dict = {}
_PROFILE_KEYS = ("input_tokens", "cache_creation_input_tokens", "output_tokens")
_PROFILE_ALPHA = 0.3


def _update_profile(profile: dict, usage: Mapping) -> dict:
    """Exponential moving average of the tokens a request actually used --
    the measured profile the router prices on each candidate model."""
    out = dict(profile)
    for k in _PROFILE_KEYS:
        v = float(usage.get(k) or 0)
        out[k] = v if k not in profile else (1 - _PROFILE_ALPHA) * profile[k] + _PROFILE_ALPHA * v
    return out


#: the subagent role -> model map from the config file, before the user's
#: dashboard overrides are laid over it
_config_roles: dict = {}
ROUTING_MODES = ("auto", "off", "pin")

#: subagent task hash -> the decision the spawn hook got (``record_spawn``),
#: so the subagent's own requests are labelled with it
_spawns: OrderedDict[str, dict] = OrderedDict()
_MAX_SPAWNS = 500

#: Claude Code session id -> running totals, for its status line
_cc_sessions: OrderedDict[str, dict] = OrderedDict()
#: (model, client user-agent) -> why the upstream rejected that model for
#: that client (e.g. "Claude Code x.y does not support this model")
_blocked: dict[tuple[str, str], str] = {}


def configure(settings: dict, picker: ModelPicker | None) -> None:
    global _settings, _picker
    _settings, _picker = settings, picker
    _sessions.clear()
    _cc_sessions.clear()
    _global_profile.clear()
    _prefixes.clear()
    _warm.clear()
    _blocked.clear()
    _spawns.clear()
    for r in read_records()[-500:]:  # seed the default profile from past agent traffic
        side = (r.get("reason") or "").startswith("side request")
        if r.get("status") == 200 and r.get("lane") == "anthropic" and not side:
            _global_profile.update(_update_profile(_global_profile, r))
    saved = _read_preferences()
    if saved.get("quality_margin") is not None:
        _settings["router"]["margin"] = saved["quality_margin"]
    _settings["routing"] = {"mode": saved.get("mode", "auto"), "pinned_model": saved.get("pinned_model"),
                            "allow_upgrade": bool(saved.get("allow_upgrade", False))}
    _config_roles.clear()
    _config_roles.update(_settings["subagents"].get("roles") or {})
    _settings["subagents"]["roles"] = {**_config_roles, **(saved.get("role_models") or {})}


#: per-request override, e.g. ``ANTHROPIC_CUSTOM_HEADERS="x-aar-quality-margin: 0.1"``
MARGIN_HEADER = "x-aar-quality-margin"
#: named cost preferences for the dashboard: how much predicted quality a turn
#: may give up to reach a cheaper model
#: Calibrated on LLMRouterBench's held-out split (1,738 prompts, 13 models,
#: kNN router) against always-GPT-5 (0.596 quality, $28.3 per 1k prompts):
#: 0.01 -> 0.615 at $24.4; 0.05 -> 0.596 at $17.2; 0.08 -> 0.580 at $12.1.
PRESETS = {"best_quality": 0.01, "balanced": 0.05, "save_more": 0.08}


def preferences_path() -> Path:
    return log_path().parent / "preferences.json"


def _read_preferences() -> dict:
    try:
        return json.loads(preferences_path().read_text())
    except (OSError, ValueError):
        return {}


def preferences() -> dict:
    routing = _settings.get("routing") or {"mode": "auto", "pinned_model": None}
    overrides = _read_preferences().get("role_models") or {}
    defaults = {**DEFAULT_ROLE_MODELS, **_config_roles}
    return {"quality_margin": _settings["router"]["margin"], "presets": PRESETS,
            "router_enabled": _picker is not None, "strongest_model": _settings.get("strongest"),
            "header": MARGIN_HEADER, "mode": routing["mode"], "pinned_model": routing["pinned_model"],
            "allow_upgrade": bool(routing.get("allow_upgrade")),
            "models": list(_settings["models"]),
            "roles": [{"role": r, "model": _settings["subagents"]["roles"].get(r, d), "default": d,
                       "overridden": r in overrides} for r, d in defaults.items()]}


def _task_key(task: str) -> str:
    return hashlib.sha1(_REMINDER.sub("", task or "").strip().encode()).hexdigest()[:16]


def record_spawn(session_id: str, subagent_type: str | None, task: str,
                 asked_model: str | None = None) -> dict:
    """The model for a subagent Claude Code is about to spawn (asked by the
    ``PreToolUse`` hook, ``aar-gateway hook``). Kept, so the subagent's own
    requests are labelled with it when they arrive. Respects the dashboard:
    routing off or pinned leaves the spawn alone."""
    routing = _settings.get("routing") or {}
    if routing.get("mode") in ("off", "pin") or not _settings["subagents"].get("enabled", True):
        decision = {"role": None, "source": None, "why": None, "model": None, "alias": None,
                    "reason": f"routing {routing.get('mode')}: spawn left as asked"}
    else:
        session_model = (_cc_sessions.get(session_id) or {}).get("requested")
        decision = spawn_decision(subagent_type, task, session_model, _settings, asked_model)
    _spawns[_task_key(task)] = decision
    _spawns.move_to_end(_task_key(task))
    while len(_spawns) > _MAX_SPAWNS:
        _spawns.popitem(last=False)
    return decision


def overrides_path() -> Path:
    """Every manual routing change from the dashboard, one JSON line each:
    labelled data on where the router's or a role's default was wrong."""
    return log_path().parent / "overrides.jsonl"


def _save_preferences(**changes) -> None:
    path = preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**_read_preferences(), **changes}))


def _log_override(record: dict) -> None:
    path = overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(UTC).isoformat(timespec="seconds"), **record}) + "\n")


def set_routing(mode: str, pinned_model: str | None = None) -> dict:
    """``auto``: the router decides; ``off``: every request keeps the model
    Claude Code asked for; ``pin``: every agent turn goes to
    ``pinned_model`` (it may be pricier than the one asked for -- the user
    chose it). Applies from the next turn; kept across restarts."""
    if mode not in ROUTING_MODES:
        raise ValueError(f"mode must be one of {ROUTING_MODES}, got {mode!r}")
    if mode == "pin" and canonical_model(pinned_model) not in _settings["models"]:
        raise ValueError(f"pin needs a configured model, got {pinned_model!r}")
    before = dict(_settings.get("routing") or {})
    pinned = canonical_model(pinned_model) if mode == "pin" else None
    _settings["routing"] = {**before, "mode": mode, "pinned_model": pinned}
    _save_preferences(mode=mode, pinned_model=pinned)
    _log_override({"kind": "routing", "from": before, "to": _settings["routing"]})
    return preferences()


def set_allow_upgrade(allow: bool) -> dict:
    """Whether the router may move a hard turn to a model pricier than the
    one Claude Code asked for (off by default: then routing only ever saves)."""
    before = dict(_settings.get("routing") or {})
    _settings["routing"] = {**before, "allow_upgrade": bool(allow)}
    _save_preferences(allow_upgrade=bool(allow))
    _log_override({"kind": "allow_upgrade", "from": bool(before.get("allow_upgrade")), "to": bool(allow)})
    return preferences()


def set_role_model(role: str, model: str | None) -> dict:
    """A subagent role's tier: a configured model, ``"session"`` (keep the
    session's model) or ``"default"`` (drop the override)."""
    defaults = {**DEFAULT_ROLE_MODELS, **_config_roles}
    if role not in defaults:
        raise ValueError(f"unknown subagent role {role!r}")
    if model not in ("session", "default") and canonical_model(model) not in _settings["models"]:
        raise ValueError(f"unknown model {model!r}")
    overrides = dict(_read_preferences().get("role_models") or {})
    before = _settings["subagents"]["roles"].get(role, defaults[role])
    if model == "default":
        overrides.pop(role, None)
    else:
        overrides[role] = None if model == "session" else canonical_model(model)
    _settings["subagents"]["roles"] = {**_config_roles, **overrides}
    _save_preferences(role_models=overrides)
    _log_override({"kind": "role", "role": role, "from": before,
                   "to": _settings["subagents"]["roles"].get(role, defaults[role])})
    return preferences()


def set_quality_margin(margin: float) -> dict:
    """The user's cost preference, applied to every new turn from now on and
    kept across restarts. 0 = always the best predicted model."""
    margin = float(margin)
    if not 0.0 <= margin <= 1.0:
        raise ValueError(f"quality_margin must be in [0, 1], got {margin}")
    _settings["router"]["margin"] = margin
    _save_preferences(quality_margin=margin)
    return preferences()


def _request_margin(request: Request) -> float:
    asked = request.headers.get(MARGIN_HEADER)
    if asked is not None:
        try:
            return min(max(float(asked), 0.0), 1.0)
        except ValueError:
            pass
    return float(_settings["router"]["margin"])


def set_model_picker(picker) -> None:
    """``picker(body, ctx)`` may return a model id, a decision dict with a
    ``model`` key, or ``None``."""
    global _picker
    _picker = picker


def set_upstream_client(client: httpx.AsyncClient | None) -> None:
    global _client
    _client = client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    return _client


def _anthropic_url(path: str) -> str:
    return os.environ.get("AAR_GATEWAY_UPSTREAM", "https://api.anthropic.com").rstrip("/") + "/" + path


def has_client_credentials(request: Request) -> bool:
    return "authorization" in request.headers or "x-api-key" in request.headers


def lane_of(model: str | None) -> str:
    return (_settings["models"].get(canonical_model(model)) or {}).get("lane", "anthropic")


def turns_path() -> Path:
    """Captured human turns (opt-in, ``interceptor.capture_prompts``): the raw
    material ``aar-gateway-grade`` replays and grades."""
    return log_path().parent / "turns.jsonl"


_CWD = re.compile(r"Primary working directory: ([^\n]+)")


def working_directory(body: Mapping) -> str | None:
    """Where Claude Code was run -- it states it in an environment block of
    the messages (or the system prompt). A replay of the turn runs there."""
    for block in _blocks(body.get("system")) + [b for m in body.get("messages") or []
                                                 for b in _blocks(m.get("content"))]:
        found = _CWD.search(block.get("text", "") if isinstance(block, Mapping) else "")
        if found:
            return found.group(1).strip()
    return None


def _capture(turn: _Turn) -> None:
    """Only the human's own words of a new agent turn, plus where it ran --
    never the conversation, tool results or credentials. For a subagent,
    the "human" is the main agent and its words are the subagent's task."""
    path = turns_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "conversation": turn.key,
            "prompt_hash": hashlib.sha1(turn.human.encode()).hexdigest()[:16],
            "prompt": turn.human,
            "cwd": working_directory(turn.body),
            "requested_model": turn.requested,
            "routed_model": turn.target,
            "subagent": turn.agent is not None,
            "agent_role": (turn.agent or {}).get("role"),
        }) + "\n")


def log_path() -> Path:
    return Path(os.environ.get("AAR_GATEWAY_LOG", Path.home() / ".aar-gateway" / "requests.jsonl"))


#: rotate the request log past this size (one previous file is kept, ``.1``)
MAX_LOG_BYTES = 50 * 1024 * 1024


def _append_record(record: dict) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.stat().st_size > MAX_LOG_BYTES:
            path.replace(path.with_name(path.name + ".1"))
    except FileNotFoundError:
        pass
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    _notify()


#: open dashboard streams (``GET /api/events``): (their event loop, queue)
_listeners: set = set()


def subscribe() -> asyncio.Queue:
    """A queue that gets an item each time a request is logged."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    _listeners.add((asyncio.get_running_loop(), queue))
    return queue


def unsubscribe(queue: asyncio.Queue) -> None:
    for entry in [e for e in _listeners if e[1] is queue]:
        _listeners.discard(entry)


def _offer(queue: asyncio.Queue) -> None:
    if not queue.full():  # one pending "something changed" is enough
        queue.put_nowait(True)


def _notify() -> None:
    for loop, queue in list(_listeners):
        try:
            loop.call_soon_threadsafe(_offer, queue)
        except RuntimeError:  # that loop is gone
            _listeners.discard((loop, queue))


def _blocks(content) -> list:
    if content is None:
        return []
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)


_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def human_text(body: Mapping) -> str | None:
    """The new human text that starts a turn, or ``None`` for a tool-loop
    continuation. Trailing ``system`` messages (Claude Code appends hook
    output after the user's message) are skipped, and ``<system-reminder>``
    sections are removed -- they are Claude Code's words, not the human's."""
    msgs = [m for m in body.get("messages") or [] if m.get("role") != "system"]
    if not msgs or msgs[-1].get("role") != "user":
        return None
    blocks = _blocks(msgs[-1].get("content"))
    if any(b.get("type") == "tool_result" for b in blocks):
        return None
    texts = [_REMINDER.sub("", b.get("text", "")).strip() for b in blocks if b.get("type") == "text"]
    text = "\n".join(t for t in texts if t)
    return text or None


def claude_session_id(request: Request, body: Mapping) -> str:
    """Claude Code's session id: its header, else the ``_session_`` suffix
    of ``metadata.user_id`` (the same place claude-code-router reads it)."""
    sid = request.headers.get("x-claude-code-session-id")
    if sid:
        return sid
    user = str((body.get("metadata") or {}).get("user_id") or "")
    return user.split("_session_", 1)[1] if "_session_" in user else ""


def _conversation_key(request: Request, body: Mapping) -> str:
    """Stable across one conversation (history is append-only) and distinct
    for side requests that share Claude Code's session id -- subagents, title
    generation -- whose first message differs."""
    first = _blocks(((body.get("messages") or [{}])[0]).get("content"))
    seed = json.dumps([(b.get("type"), b.get("text") or b.get("id") or "") for b in first])[:4000]
    sid = request.headers.get("x-claude-code-session-id", "")
    return hashlib.sha1((sid + seed).encode()).hexdigest()[:16]


def _prompt_tokens(usage: Mapping) -> int:
    return sum(usage.get(k) or 0 for k in
               ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))


def prefix_signature(body: Mapping) -> str:
    """What a prompt cache is keyed on, roughly: system prompt and tools.
    Claude Code sends the same ones in every session, so a new session
    reuses a cache an earlier session wrote -- on the same model."""
    seed = json.dumps([body.get("system"), body.get("tools")], sort_keys=True, default=str)
    return hashlib.sha1(seed.encode()).hexdigest()[:16]


def _ttl() -> float:
    return 3600.0 if _settings["router"].get("cache_ttl", "1h") == "1h" else 300.0


def _shared_tokens(sig: str) -> int:
    seen = _prefixes.get(sig)
    return seen[1] if seen and time.monotonic() - seen[0] < _ttl() else 0


def _warm_usage(usage: Mapping, warm: int) -> dict:
    """``usage`` as it would have been on a model holding ``warm`` prefix
    tokens in cache: that much of what was written is a read instead."""
    out = dict(usage)
    read = usage.get("cache_read_input_tokens") or 0
    moved = min(max(min(warm, _prompt_tokens(usage)) - read, 0), usage.get("cache_creation_input_tokens") or 0)
    if moved:
        out["cache_creation_input_tokens"] -= moved
        out["cache_read_input_tokens"] = read + moved
        split = dict(usage.get("cache_creation") or {})
        left = moved
        for k in ("ephemeral_1h_input_tokens", "ephemeral_5m_input_tokens"):
            take = min(left, split.get(k) or 0)
            if take:
                split[k] -= take
                left -= take
        out["cache_creation"] = split
    return out


class _SSEUsageTap:
    """Token usage read off an Anthropic SSE stream as it passes through."""

    def __init__(self):
        self._buf = b""
        self.usage: dict = {}
        self.stop_reason: str | None = None

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        while b"\n\n" in self._buf:
            event, self._buf = self._buf.split(b"\n\n", 1)
            for line in event.split(b"\n"):
                if line.startswith(b"data:"):
                    self._read(line[5:].strip())

    def _read(self, data: bytes) -> None:
        try:
            payload = json.loads(data)
        except ValueError:
            return
        if payload.get("type") == "message_start":
            self.usage.update((payload.get("message") or {}).get("usage") or {})
        elif payload.get("type") == "message_delta":
            self.usage.update({k: v for k, v in (payload.get("usage") or {}).items() if v is not None})
            self.stop_reason = (payload.get("delta") or {}).get("stop_reason") or self.stop_reason


class _Turn:
    """Everything known about one ``/v1/messages`` request, for routing and
    for its log record."""

    def __init__(self, request: Request, raw: bytes, body: dict):
        self.request, self.raw, self.body = request, raw, body
        self.requested = body.get("model")
        self.target = self.requested
        self.key = _conversation_key(request, body)
        self.claude_session = claude_session_id(request, body)
        self.sig = prefix_signature(body)
        self.shared_before = _shared_tokens(self.sig)
        self.session = dict(_sessions.get(self.key) or {})
        self.human = human_text(body)
        self.stop_reason: str | None = None
        # a subagent's role comes from its task, which only its first request
        # carries as new text; later requests reuse what that one found
        self.agent = (self.session["agent"] if self.human is None and "agent" in self.session
                      else detect_subagent(body, self.human))
        spawned = _spawns.get(_task_key(self.human)) if self.agent and self.human else None
        if spawned and spawned.get("role"):  # the spawn hook already decided this one
            self.agent = {"role": spawned["role"], "source": "hook", "why": spawned.get("why")}
        self.decision: dict = {"reason": "client's choice"}
        self.fallback = False
        self.attempted: str | None = None
        self.t0 = time.perf_counter()

    def body_for(self, model: str) -> bytes:
        return self.raw if model == self.requested else json.dumps({**self.body, "model": model}).encode()

    def finish(self, status: int, usage: Mapping | None, lane: str) -> None:
        """Baselines ("on the model you asked for", "on the strongest") assume
        that model had served all the traffic: it would hold this
        conversation's previous prefix, or at least the system prompt and
        tools any recent request with the same prefix cached."""
        usage = dict(usage or {})
        # a subagent's baseline is the session's own model: what it would have
        # run on with no routing, even when its tier was set at spawn time
        asked = self.requested
        if self.agent:
            asked = (_cc_sessions.get(self.claude_session) or {}).get("requested") or self.requested
        prev_tokens = self.session.get("prompt_tokens") or 0
        baseline = _warm_usage(usage, max(prev_tokens, self.shared_before))
        strongest = _settings.get("strongest")
        if status == 200 and usage:
            now, tokens = time.monotonic(), _prompt_tokens(usage)
            if lane == "anthropic":
                _warm[(canonical_model(self.target), self.sig)] = now
            fewest = min(self.shared_before or tokens, tokens)
            _prefixes[self.sig] = (now, fewest)
            profile = _update_profile(self.session.get("profile") or {}, usage)
            _sessions[self.key] = {"model": self.target, "prompt_tokens": tokens, "profile": profile,
                                   "agent": self.agent}
            if self.body.get("tools"):
                _global_profile.update(_update_profile(_global_profile, usage))
            _sessions.move_to_end(self.key)
            while len(_sessions) > _MAX_SESSIONS:
                _sessions.popitem(last=False)
        d = self.decision
        cost = cost_usd(price_for(_settings, self.target), usage)
        cost_asked = cost_usd(price_for(_settings, asked), baseline)
        if self.claude_session and status == 200:
            tot = _cc_sessions.setdefault(self.claude_session, {"cost_usd": 0.0, "cost_if_requested_usd": 0.0,
                                                                "requests": 0, "rerouted": 0})
            tot["requests"] += 1
            tot["cost_usd"] += cost or 0.0
            tot["cost_if_requested_usd"] += cost_asked if cost_asked is not None else (cost or 0.0)
            if (self.body.get("tools") and not self.agent) or "model" not in tot:
                # the main agent's model: not side requests, not subagents
                tot.update(model=self.target, requested=self.requested)
            tot["rerouted"] += self.target != asked
            _cc_sessions.move_to_end(self.claude_session)
            while len(_cc_sessions) > _MAX_SESSIONS:
                _cc_sessions.popitem(last=False)
        _append_record({
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "claude_session": self.claude_session[:12] or None,
            "conversation": self.key,
            "new_turn": self.human is not None,
            "subagent": self.agent is not None,
            "agent_role": (self.agent or {}).get("role"),
            "role_source": (self.agent or {}).get("source"),
            "baseline_model": asked if asked != self.requested else None,
            # a subagent's final answer -- all the main agent gets back
            "report_tokens": usage.get("output_tokens") if self.agent and self.stop_reason == "end_turn" else None,
            "requested_model": self.requested,
            "routed_model": self.target,
            "lane": lane,
            "reason": d.get("reason"),
            "fallback": self.fallback,
            "attempted_model": self.attempted,
            "status": status,
            "latency_s": round(time.perf_counter() - self.t0, 3),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cost_usd": cost,
            "cost_if_requested_usd": cost_asked,
            "strongest_model": strongest,
            "cost_if_strongest_usd": cost_usd(price_for(_settings, strongest), baseline),
            "predicted_quality": d.get("predicted_quality"),
            "best_predicted_quality": d.get("best_predicted_quality"),
            "candidates": d.get("candidates"),
            "quality_margin": d.get("quality_margin"),
            "quality_source": d.get("quality_source"),
        })


def _warm_tokens(sig: str) -> dict[str, int]:
    """Models that recently cached this prefix, with how many tokens of it."""
    shared, now = _shared_tokens(sig), time.monotonic()
    return {m: shared for (m, s), t in _warm.items() if s == sig and now - t < _ttl()} if shared else {}


def _blocked_for(turn: _Turn) -> dict[str, str]:
    ua = turn.request.headers.get("user-agent", "")
    return {m: why for (m, agent), why in _blocked.items() if agent == ua}


def _est_prompt_tokens(turn: _Turn) -> int:
    return max(turn.session.get("prompt_tokens") or 0, len(turn.raw) // 4)


async def _decide(turn: _Turn) -> None:
    if turn.human is None and turn.session.get("model"):
        turn.target = turn.session["model"]
        turn.decision = {"reason": "same turn as the previous request"}
        return
    if turn.human is None:
        return
    routing = _settings.get("routing") or {}
    if routing.get("mode") == "off":
        turn.decision = {"reason": "routing off (dashboard): the model Claude Code asked for"}
        return
    if routing.get("mode") == "pin" and turn.body.get("tools"):
        pinned = routing["pinned_model"]
        why = unavailable_reason(pinned, turn.body, _settings, _blocked_for(turn), _est_prompt_tokens(turn))
        if why:
            turn.decision = {"reason": f"pinned {pinned} can't take this request ({why}); kept the client's model"}
        else:
            turn.target, turn.decision = pinned, {"reason": f"pinned to {pinned} (dashboard)"}
        return
    if turn.agent and turn.body.get("tools") and _settings["subagents"].get("enabled", True):
        by_role = role_decision(turn.agent, turn.body, turn.requested, _settings,
                                _blocked_for(turn), _est_prompt_tokens(turn))
        if by_role:
            turn.target, turn.decision = by_role["model"], by_role
            return
    if _picker is None:
        return
    if not turn.body.get("tools"):
        # Claude Code's side requests (titles, summaries) carry no tools and
        # cost little; rerouting them saves nothing and risks a model that
        # rejects their settings. Only the agent conversation is routed.
        turn.decision = {"reason": "side request (no tools), not routed"}
        return
    ctx = {"requested": turn.requested, "human_text": turn.human,
           "est_prompt_tokens": _est_prompt_tokens(turn),
           "last_model": turn.session.get("model"),
           "last_prompt_tokens": turn.session.get("prompt_tokens"),
           "quality_margin": _request_margin(turn.request),
           "allow_upgrade": bool(routing.get("allow_upgrade")),
           "usage_profile": turn.session.get("profile") or dict(_global_profile) or None,
           "warm_tokens": _warm_tokens(turn.sig),
           "blocked": _blocked_for(turn)}
    try:
        picked = await run_in_threadpool(_picker, turn.body, ctx)
    except Exception as exc:  # a router failure must never break the client
        turn.decision = {"reason": f"router error, kept client's model: {exc!r}"[:300]}
        return
    if isinstance(picked, str):
        picked = {"model": picked, "reason": "picker"}
    if picked and picked.get("model"):
        turn.target, turn.decision = picked["model"], picked


async def _send_anthropic(request: Request, path: str, content: bytes) -> httpx.Response:
    client = _get_client()
    upstream_request = client.build_request(
        request.method, _anthropic_url(path), params=request.query_params,
        headers={k: v for k, v in request.headers.items() if k.lower() not in _REQUEST_HOP_HEADERS},
        content=content)
    return await client.send(upstream_request, stream=True)


async def _relay_anthropic(upstream: httpx.Response, on_done) -> Response:
    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _RESPONSE_HOP_HEADERS}
    if upstream.headers.get("content-type", "").startswith("text/event-stream"):
        tap = _SSEUsageTap()

        async def relay():
            try:
                async for chunk in upstream.aiter_bytes():
                    tap.feed(chunk)
                    yield chunk
            finally:
                await upstream.aclose()
                on_done(upstream.status_code, tap.usage, tap.stop_reason)

        return StreamingResponse(relay(), status_code=upstream.status_code, headers=headers)

    content = await upstream.aread()
    await upstream.aclose()
    usage = stop_reason = None
    if upstream.status_code == 200:
        try:
            message = json.loads(content)
            usage, stop_reason = message.get("usage"), message.get("stop_reason")
        except (ValueError, AttributeError):
            pass
    on_done(upstream.status_code, usage, stop_reason)
    return Response(content, status_code=upstream.status_code, headers=headers)


def _unreachable(turn: _Turn | None, where: str, exc: Exception, lane: str) -> JSONResponse:
    """A network failure reaching the provider: an Anthropic-shaped 502
    (Claude Code retries those) instead of an unlogged crash."""
    if turn is not None:
        turn.decision = {**turn.decision, "reason": f"{where} unreachable: {type(exc).__name__}"}
        turn.finish(502, None, lane)
    return _error(502, f"{where} unreachable: {exc!r}"[:300])


async def _via_anthropic(turn: _Turn) -> Response:
    try:
        return await _via_anthropic_unguarded(turn)
    except httpx.HTTPError as exc:
        return _unreachable(turn, "Anthropic API", exc, "anthropic")


async def _via_anthropic_unguarded(turn: _Turn) -> Response:
    upstream = await _send_anthropic(turn.request, "v1/messages", turn.body_for(turn.target))
    if upstream.status_code >= 400 and turn.target != turn.requested:
        detail = (await upstream.aread()).decode(errors="replace")[:300]
        await upstream.aclose()
        if upstream.status_code == 400:  # the request is valid for the client's model, not this one
            _blocked[(turn.target, turn.request.headers.get("user-agent", ""))] = detail[:160]
        turn.fallback, turn.attempted, turn.target = True, turn.target, turn.requested
        turn.decision = {**turn.decision, "reason": f"fell back: {turn.attempted} returned "
                                                    f"{upstream.status_code}: {detail[:160]}"}
        upstream = await _send_anthropic(turn.request, "v1/messages", turn.raw)
    def done(status: int, usage: Mapping | None, stop_reason: str | None = None) -> None:
        turn.stop_reason = stop_reason
        turn.finish(status, usage, "anthropic")

    return await _relay_anthropic(upstream, done)


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"type": "error", "error": {"type": "api_error", "message": message}},
                        status_code=status)


async def _via_openai(turn: _Turn) -> Response:
    try:
        return await _via_openai_unguarded(turn)
    except httpx.HTTPError as exc:
        return _unreachable(turn, turn.target, exc, "openai")


async def _via_openai_unguarded(turn: _Turn) -> Response:
    spec = _settings["models"][canonical_model(turn.target)]
    key_env = spec.get("api_key_env")
    headers = {"content-type": "application/json"}
    if key_env:
        headers["authorization"] = f"Bearer {os.environ[key_env]}"
    payload = anthropic_to_openai(turn.body, spec.get("upstream_model", turn.target),
                                  max_tokens_param=spec.get("max_tokens_param", "max_tokens"))
    client = _get_client()
    upstream = await client.send(client.build_request(
        "POST", spec["base_url"].rstrip("/") + "/chat/completions", headers=headers, json=payload),
        stream=True)
    if upstream.status_code >= 400:
        detail = (await upstream.aread()).decode(errors="replace")[:500]
        await upstream.aclose()
        if turn.target != turn.requested and lane_of(turn.requested) == "anthropic" \
                and has_client_credentials(turn.request):
            turn.fallback, turn.attempted, turn.target = True, turn.target, turn.requested
            turn.decision = {**turn.decision, "reason": f"fell back: {turn.attempted} returned "
                                                        f"{upstream.status_code}: {detail[:160]}"}
            return await _via_anthropic(turn)
        turn.finish(upstream.status_code, None, "openai")
        return _error(upstream.status_code, f"{turn.target}: {detail}")

    if payload["stream"]:
        translator = OpenAIStreamTranslator(turn.target)

        async def relay():
            try:
                async for line in upstream.aiter_lines():
                    for chunk in iter_openai_sse([line]):
                        for event in translator.feed(chunk):
                            yield event
                for event in translator.finish():
                    yield event
            finally:
                await upstream.aclose()
                turn.finish(200, translator.usage, "openai")

        return StreamingResponse(relay(), media_type="text/event-stream")

    message = openai_to_anthropic(json.loads(await upstream.aread()), turn.target)
    await upstream.aclose()
    turn.finish(200, message["usage"], "openai")
    return JSONResponse(message)


async def handle_messages(request: Request) -> Response | None:
    """``None`` means "not mine": no credentials to forward and the model is
    not on an ``openai`` lane -- the server answers 401."""
    raw = await request.body()
    body = json.loads(raw or b"{}")
    turn = _Turn(request, raw, body)
    await _decide(turn)
    if _settings.get("capture_prompts") and turn.human is not None and body.get("tools"):
        _capture(turn)
    if lane_of(turn.target) == "openai":
        return await _via_openai(turn)
    if not has_client_credentials(request):
        return None
    return await _via_anthropic(turn)


async def forward(request: Request, path: str) -> Response:
    """Any other Anthropic endpoint Claude Code calls (``count_tokens`` ...).
    For a model on an ``openai`` lane there is no Anthropic endpoint to ask,
    so token counts are estimated at ~4 characters per token."""
    raw = await request.body()
    if path == "v1/messages/count_tokens" and raw:
        model = json.loads(raw).get("model")
        if lane_of(model) == "openai":
            return JSONResponse({"input_tokens": len(raw) // 4})
    try:
        upstream = await _send_anthropic(request, path, raw)
    except httpx.HTTPError as exc:
        return _unreachable(None, "Anthropic API", exc, "anthropic")
    return await _relay_anthropic(upstream, lambda status, usage, stop_reason=None: None)


#: the request log as parsed so far: a dashboard refresh parses only the
#: lines appended since the last one, not the whole file
_log_cache: dict = {"key": None, "offset": 0, "partial": b"", "records": []}


def read_records() -> list[dict]:
    path = log_path()
    try:
        st = path.stat()
    except FileNotFoundError:
        _log_cache.update(key=None, offset=0, partial=b"", records=[])
        return []
    key = (str(path), st.st_ino)
    if _log_cache["key"] != key or st.st_size < _log_cache["offset"]:  # new, rotated or rewritten
        _log_cache.update(key=key, offset=0, partial=b"", records=[])
    if st.st_size > _log_cache["offset"]:
        with path.open("rb") as f:
            f.seek(_log_cache["offset"])
            data = f.read()
        _log_cache["offset"] += len(data)
        lines = (_log_cache["partial"] + data).split(b"\n")
        _log_cache["partial"] = lines.pop()  # an unfinished last line waits for the rest
        _log_cache["records"].extend(json.loads(line) for line in lines if line.strip())
    return list(_log_cache["records"])


#: newest requests sent to the dashboard, which pages through them
RECENT_RECORDS = 200


def summarize(records: list[dict] | None = None) -> dict:
    """Totals and per-model rows for ``GET /stats`` and the dashboard."""
    records = read_records() if records is None else records
    token_keys = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")

    def blank() -> dict:
        return {"requests": 0, "errors": 0, "rerouted": 0, "fallbacks": 0,
                **dict.fromkeys(token_keys, 0), "cost_usd": 0.0}

    total = {**blank(), "cost_if_requested_usd": 0.0, "cost_if_strongest_usd": 0.0,
             "priced_requests": 0, "turns_routed": 0}
    models: dict = {}
    quality, best_quality = [], []
    for r in records:
        m = models.setdefault(r.get("routed_model") or "unknown", blank())
        for row in (total, m):
            row["requests"] += 1
            row["errors"] += r.get("status") != 200
            row["rerouted"] += r.get("routed_model") != r.get("requested_model")
            row["fallbacks"] += bool(r.get("fallback"))
            for k in token_keys:
                row[k] += r.get(k) or 0
            row["cost_usd"] += r.get("cost_usd") or 0.0
        if None not in (r.get("cost_usd"), r.get("cost_if_requested_usd"), r.get("cost_if_strongest_usd")):
            total["priced_requests"] += 1
            total["cost_if_requested_usd"] += r["cost_if_requested_usd"]
            total["cost_if_strongest_usd"] += r["cost_if_strongest_usd"]
        if r.get("predicted_quality") is not None:
            total["turns_routed"] += 1
            quality.append(r["predicted_quality"])
            best_quality.append(r.get("best_predicted_quality") or r["predicted_quality"])
    total["predicted_quality"] = sum(quality) / len(quality) if quality else None
    total["subagent_requests"] = sum(1 for r in records if r.get("subagent"))
    total["best_predicted_quality"] = sum(best_quality) / len(best_quality) if best_quality else None
    strongest = next((r.get("strongest_model") for r in reversed(records) if r.get("strongest_model")),
                     _settings.get("strongest"))
    return {"log": str(log_path()), "strongest_model": strongest, "totals": total, "models": models,
            "subagents": summarize_subagents(records), "sessions": summarize_sessions(records),
            "recent": list(reversed(records[-RECENT_RECORDS:]))}


#: Claude Code sessions listed on the dashboard, newest first
SESSION_ROWS = 30


def summarize_sessions(records: list[dict]) -> list[dict]:
    """Plan usage per Claude Code session, against the same work on the
    models it asked for and on the strongest. Requests logged before the
    gateway recorded session ids are one row, ``session: None``."""
    rows: dict = {}
    for r in records:
        sid = r.get("claude_session")
        row = rows.setdefault(sid, {"session": sid, "first_ts": r.get("ts"), "last_ts": r.get("ts"),
                                    "requests": 0, "turns": 0, "subagent_spawns": set(), "rerouted": 0,
                                    "models": {}, "cost_usd": 0.0, "cost_if_requested_usd": 0.0,
                                    "cost_if_strongest_usd": 0.0})
        row["last_ts"] = r.get("ts") or row["last_ts"]
        row["requests"] += 1
        row["turns"] += bool(r.get("new_turn") and not r.get("subagent")
                             and not (r.get("reason") or "").startswith("side request"))
        if r.get("subagent"):
            row["subagent_spawns"].add(r.get("conversation"))
        row["rerouted"] += r.get("routed_model") != r.get("requested_model")
        model = r.get("routed_model") or "unknown"
        row["models"][model] = row["models"].get(model, 0) + 1
        cost = r.get("cost_usd") or 0.0
        row["cost_usd"] += cost
        row["cost_if_requested_usd"] += r["cost_if_requested_usd"] if r.get("cost_if_requested_usd") is not None else cost
        row["cost_if_strongest_usd"] += r["cost_if_strongest_usd"] if r.get("cost_if_strongest_usd") is not None else cost
    out = [{**row, "subagent_spawns": len(row["subagent_spawns"])} for row in rows.values()]
    return sorted(out, key=lambda x: x["last_ts"] or "", reverse=True)[:SESSION_ROWS]


def summarize_subagents(records: list[dict]) -> list[dict]:
    """One row per subagent role: spawns, the models that answered, plan
    usage against the session's own model, and how much of the prompt was
    read from cache."""
    rows: dict = {}
    for r in records:
        if not r.get("subagent"):
            continue
        role = r.get("agent_role") or "unrecognised"
        row = rows.setdefault(role, {"role": role, "source": r.get("role_source"), "conversations": set(),
                                     "requests": 0, "models": {}, "cost_usd": 0.0,
                                     "cost_if_requested_usd": 0.0, "cache_read_input_tokens": 0,
                                     "prompt_tokens": 0, "latency_s": 0.0, "reports": []})
        row["conversations"].add(r.get("conversation"))
        row["requests"] += 1
        model = r.get("routed_model") or "unknown"
        row["models"][model] = row["models"].get(model, 0) + 1
        row["cost_usd"] += r.get("cost_usd") or 0.0
        row["cost_if_requested_usd"] += r.get("cost_if_requested_usd") or r.get("cost_usd") or 0.0
        row["cache_read_input_tokens"] += r.get("cache_read_input_tokens") or 0
        row["prompt_tokens"] += _prompt_tokens(r)
        row["latency_s"] += r.get("latency_s") or 0.0
        if r.get("report_tokens") is not None:
            row["reports"].append(r["report_tokens"])
    out = []
    for row in rows.values():
        spawns = len(row.pop("conversations"))
        reports = row.pop("reports")
        row["avg_report_tokens"] = round(sum(reports) / len(reports)) if reports else None
        prompt = row["prompt_tokens"]
        out.append({**row, "spawns": spawns,
                    "cache_hit_rate": row["cache_read_input_tokens"] / prompt if prompt else None})
    return sorted(out, key=lambda x: -x["cost_usd"])


def session_summary(session_id: str) -> dict:
    """What Claude Code's status line shows for one session: the model that
    answered the last turn, and how much less plan usage the session took
    than without the router. Usage is scaled by API price -- the relative cost
    of the same work on each model -- since on a subscription the dollars
    themselves are never billed."""
    tot = _cc_sessions.get(session_id)
    if not tot:
        return {"line": "aar router · waiting for the first turn"}
    model = tot.get("model") or "?"
    how = "routed" if model != tot.get("requested") else "as asked"
    base = tot["cost_if_requested_usd"]
    if base > 0:
        pct = 1 - tot["cost_usd"] / base
        usage = f"{pct:.0%} less plan usage" if pct >= 0 else f"{-pct:.0%} more plan usage"
    else:
        usage = "no usage yet"
    return {**tot, "plan_usage_saved": (1 - tot["cost_usd"] / base) if base > 0 else None,
            "line": f"aar router · {model} ({how}) · {usage}"}
