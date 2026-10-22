"""What would a router file have saved on your own recent Claude Code sessions?

Replays the typed turns of your session transcripts (``~/.claude/projects/*/*.jsonl``,
the main conversation; grading/bench temp sessions and subagents left out) through
the gateway's own picker (``aar_gateway.routing.build_picker``) with the session's
context -- the tier the previous turn ran on, the prompt's length, a running usage
profile -- at each lambda (and the margin rule). Every request of a turn is
repriced on the tier picked for the turn, from the request's REAL token usage
(``aar_gateway.pricing.cost_usd``); a switch to a different tier than the previous
turn's turns the first request's cache reads into a 1-hour cache write (the cache
is per model), so switching is never free. No upgrades: a turn can only move to a
model no pricier than the one it ran on.

Quality is not measured here (these turns have no grades): the "predicted quality"
columns are the router's own estimate of the picked tier vs the one used.
Prompt text is read locally to route it and never printed or written.

    python scripts/routing/session_savings.py --days 7 \\
        --router aar-gateway/src/aar_gateway/routers/manifold-nirt-claude-judged.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

LAMS = (None, 0.02, 0.2, 0.3)          # None = the margin rule (router.margin)
#: project folders of automated runs, not the person's own work: grading and bench
#: sessions run from temp folders (on Windows "...-AppData-Local-Temp-...", which
#: ``aar_gateway.grading.history._TEMP_PROJECTS`` -- macOS/Linux prefixes -- misses)
_AUTOMATED = ("AppData-Local-Temp", "aar-router-bench", "aar-grade")


def _automated(folder: str) -> bool:
    from aar_gateway.grading.history import _TEMP_PROJECTS

    return folder.startswith(_TEMP_PROJECTS) or any(a in folder for a in _AUTOMATED)
USAGE_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def session_turns(path: Path) -> list[dict]:
    """``[{"text", "requests": [{"model", "usage"}]}]`` for one transcript's typed turns.
    One API response spans several transcript lines with the same ``message.id``
    (one per content block, each repeating the usage): it is counted once."""
    from aar_gateway.translate import REMINDER, text_of

    from evaluation.gateway.multiturn import _IDE, _read_jsonl

    turns, current, seen = [], None, {}
    for rec in _read_jsonl(path):
        if rec.get("isSidechain") or rec.get("isMeta"):
            continue
        msg = rec.get("message") or {}
        if rec.get("type") == "user":
            content = msg.get("content")
            blocks = content if isinstance(content, list) else []
            origin = (rec.get("origin") or {}).get("kind")
            typed = origin == "human" if origin else not any(b.get("type") == "tool_result" for b in blocks)
            raw = text_of(content, sep="\n").lstrip()
            text = _IDE.sub("", REMINDER.sub("", raw)).strip()
            if typed and text and not raw.startswith(("/", "<command-", "<local-command", "<task-notification")):
                current = {"text": text, "requests": []}
                turns.append(current)
        elif rec.get("type") == "assistant" and current is not None and msg.get("usage") and msg.get("model"):
            key = msg.get("id") or id(rec)
            req = {"model": msg["model"], "usage": {k: msg["usage"].get(k) or 0 for k in USAGE_KEYS}}
            split = msg["usage"].get("cache_creation") or {}
            if split:
                req["usage"]["cache_creation"] = dict(split)
            if key in seen:
                seen[key].update(req)        # the last line of a response carries its final usage
            else:
                seen[key] = req
                current["requests"].append(req)
    return [t for t in turns if t["requests"]]


def _profile(prev: dict | None, usage: dict, alpha: float = 0.3) -> dict:
    keys = ("input_tokens", "cache_creation_input_tokens", "output_tokens")
    if prev is None:
        return {k: float(usage[k]) for k in keys}
    return {k: (1 - alpha) * prev[k] + alpha * float(usage[k]) for k in keys}


def replay(sessions: dict[str, list[dict]], pick, lam, prices) -> dict:
    """Route every turn of every session at ``lam`` and price it; returns the totals."""
    from aar_gateway.pricing import canonical_model, cost_usd, prompt_tokens

    body = {"model": "x", "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "x"}]}
    tot = {"turns": 0, "routed_turns": 0, "kept": 0, "base_usd": 0.0, "routed_usd": 0.0, "switches": 0,
           "mix": {}, "pred_q_used": 0.0, "pred_q_picked": 0.0, "pred_n": 0, "unpriced_requests": 0}
    for turns in sessions.values():
        last_model = last_prompt = prev_q = profile = None
        for t in turns:
            used = canonical_model(t["requests"][0]["model"])
            first = t["requests"][0]["usage"]
            ctx = {"requested": used, "human_text": t["text"], "est_prompt_tokens": prompt_tokens(first),
                   "last_model": last_model, "last_prompt_tokens": last_prompt, "prev_quality": prev_q,
                   "quality_margin": None, "cost_weight": lam, "allow_upgrade": False, "turn_smoothing": True,
                   "usage_profile": profile, "warm_tokens": {}, "blocked": {}}
            d = pick(body, ctx) if used in prices else None
            target = canonical_model(d["model"]) if d and d.get("model") else used
            tot["turns"] += 1
            if not d or d.get("reason") == "no eligible candidates":
                tot["kept"] += 1
            else:
                tot["routed_turns"] += 1
                q = d.get("model_quality") or {}
                if used in q and target in q:
                    tot["pred_q_used"] += q[used]
                    tot["pred_q_picked"] += q[target]
                    tot["pred_n"] += 1
                prev_q = q or prev_q
            tot["mix"][target] = tot["mix"].get(target, 0) + 1
            switched = last_model is not None and target != last_model
            tot["switches"] += int(switched)
            for i, r in enumerate(t["requests"]):
                u = dict(r["usage"])
                base = cost_usd(prices.get(canonical_model(r["model"])), u)
                if i == 0 and switched and u.get("cache_read_input_tokens"):
                    # the new tier holds no cache for this conversation: its prefix is written again (1h TTL)
                    split = dict(u.get("cache_creation") or {})
                    split["ephemeral_1h_input_tokens"] = (split.get("ephemeral_1h_input_tokens") or 0) \
                        + u["cache_read_input_tokens"]
                    split.setdefault("ephemeral_5m_input_tokens",
                                     max(u["cache_creation_input_tokens"]
                                         - (u.get("cache_creation") or {}).get("ephemeral_1h_input_tokens", 0), 0))
                    u["cache_creation_input_tokens"] += u["cache_read_input_tokens"]
                    u["cache_read_input_tokens"] = 0
                    u["cache_creation"] = split
                routed = cost_usd(prices.get(target), u)
                if base is None or routed is None:
                    tot["unpriced_requests"] += 1
                    continue
                tot["base_usd"] += base
                tot["routed_usd"] += routed
                profile = _profile(profile, r["usage"])
            last_model, last_prompt = target, prompt_tokens(t["requests"][-1]["usage"])
    return tot


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", required=True, help="the router file to replay")
    ap.add_argument("--days", type=float, default=7.0, help="sessions modified in the last N days")
    ap.add_argument("--margin", type=float, default=0.05, help="router.margin for the margin-rule row")
    ap.add_argument("--out", default="investigations/profile_manifold/session_savings.json")
    args = ap.parse_args(argv)

    from aar_gateway.config import load_settings
    from aar_gateway.grading.history import PROJECTS_DIR
    from aar_gateway.pricing import CLAUDE_PRICES
    from aar_gateway.routing import build_picker, load_router_model

    cutoff = time.time() - args.days * 86400
    paths = [p for p in sorted(PROJECTS_DIR.glob("*/*.jsonl"))
             if not _automated(p.parent.name) and p.stat().st_mtime >= cutoff]
    sessions = {p.stem: session_turns(p) for p in paths}
    sessions = {k: v for k, v in sessions.items() if v}
    n_turns = sum(len(v) for v in sessions.values())
    n_req = sum(len(t["requests"]) for v in sessions.values() for t in v)
    used = {}
    for v in sessions.values():
        for t in v:
            used[t["requests"][0]["model"]] = used.get(t["requests"][0]["model"], 0) + 1
    print(f"{len(sessions)} sessions in the last {args.days:g} days: {n_turns} typed turns, {n_req} requests; "
          f"models the turns ran on: {used}")

    cfg = Path(tempfile.mkdtemp()) / "gateway.yaml"
    cfg.write_text(f"interceptor:\n  router: {{enabled: true, artifact: {json.dumps(str(Path(args.router).resolve()))}, "
                   f"margin: {args.margin}}}\n", encoding="utf-8")
    settings = load_settings(cfg)
    pick = build_picker(settings, router_model=load_router_model(settings["router"]))
    prices = {m: spec.get("prices") or CLAUDE_PRICES.get(m) for m, spec in settings["models"].items()}

    rows = []
    for lam in LAMS:
        tot = replay(sessions, pick, lam, prices)
        saved = 1 - tot["routed_usd"] / tot["base_usd"] if tot["base_usd"] else math.nan
        dq = ((tot["pred_q_picked"] - tot["pred_q_used"]) / tot["pred_n"]) if tot["pred_n"] else math.nan
        rows.append({"lambda": lam, **tot, "saved_pct": saved, "pred_quality_delta": dq})
        label = f"margin {args.margin:g}" if lam is None else f"lambda {lam:g}"
        mix = ", ".join(f"{m.replace('claude-', '')} {n}" for m, n in sorted(tot["mix"].items()))
        print(f"{label:12s} no router ${tot['base_usd']:8.2f} -> routed ${tot['routed_usd']:8.2f}  saved {saved:6.1%}  "
              f"switches {tot['switches']:4d}  predicted quality {dq:+.3f}  | {mix}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"router": args.router, "days": args.days, "sessions": len(sessions),
                               "turns": n_turns, "requests": n_req, "models_used": used, "rows": rows},
                              indent=1, default=float), encoding="utf-8")
    print(f"results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
