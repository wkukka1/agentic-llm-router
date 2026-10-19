"""Read-only questions over the gateway's request log. Prints JSON.

    python query.py savings [--days N]     plan usage vs no router and vs always Opus
    python query.py errors [--days N]      failed requests by status and error type
    python query.py switches [--days N]    model switches and what they cost in cache
    python query.py threads [--days N]     server-side thread requests and their outcomes
    python query.py session ID [--days N]  one Claude Code session

Reads ``AAR_GATEWAY_LOG`` (default ``~/.aar-gateway/requests.jsonl``) only.
It never reads prompt text (``turns.jsonl``) and never writes anything.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


def log_path() -> Path:
    return Path(os.environ.get("AAR_GATEWAY_LOG", Path.home() / ".aar-gateway" / "requests.jsonl"))


def load(days: int | None) -> list[dict]:
    path = log_path()
    if not path.exists():
        return []
    cut = datetime.now(UTC) - timedelta(days=days) if days else None
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if cut is None or datetime.fromisoformat(r["ts"]) >= cut:
            rows.append(r)
    return rows


def _priced(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("status") == 200 and r.get("cost_usd") is not None
            and r.get("cost_if_requested_usd") is not None]


def _saving(actual: float, base: float) -> dict:
    return {"usage_usd": round(actual, 2), "baseline_usd": round(base, 2),
            "saved_usd": round(base - actual, 2), "saved_pct": round(100 * (1 - actual / base), 1) if base else None}


def savings(rows: list[dict]) -> dict:
    pr = _priced(rows)
    main = [r for r in pr if not r.get("subagent")]
    sub = [r for r in pr if r.get("subagent")]
    strongest = [r for r in pr if r.get("cost_if_strongest_usd") is not None]
    out = {
        "requests": len(rows), "priced_requests": len(pr),
        "vs_no_router": _saving(sum(r["cost_usd"] for r in pr), sum(r["cost_if_requested_usd"] for r in pr)),
        "vs_always_opus": _saving(sum(r["cost_usd"] for r in strongest), sum(r["cost_if_strongest_usd"] for r in strongest)),
        "main_agent": _saving(sum(r["cost_usd"] for r in main), sum(r["cost_if_requested_usd"] for r in main)),
        "subagents": _saving(sum(r["cost_usd"] for r in sub), sum(r["cost_if_requested_usd"] for r in sub)),
        "note": "vs_no_router is the honest headline; vs_always_opus prices cheap requests as Opus and reads higher",
    }
    return out


def errors(rows: list[dict]) -> dict:
    expired = [r for r in rows if r.get("status") == 404 and "No thread state" in (r.get("error") or "")]
    bad = [r for r in rows if r.get("status") != 200 and r not in expired]  # same rule as the dashboard
    by_status = collections.Counter(str(r.get("status")) for r in bad)
    by_type = collections.Counter(((r.get("error") or "").split(":", 1)[0] or "unknown") for r in bad)
    return {"requests": len(rows), "failed": len(bad), "by_status": dict(by_status), "by_error_type": dict(by_type),
            "expired_threads_not_counted": len(expired),
            "fallbacks": sum(1 for r in rows if r.get("fallback"))}


def switches(rows: list[dict]) -> dict:
    sw = [r for r in _priced(rows) if r.get("routed_model") != r.get("requested_model")]
    pairs = collections.Counter(f"{r['requested_model']} -> {r['routed_model']}" for r in sw)
    costlier = [r for r in sw if r["cost_usd"] > r["cost_if_requested_usd"] * 1.0001]
    return {"switches": len(sw), "pairs": dict(pairs.most_common(8)),
            "costlier_than_no_router": len(costlier),
            "cache_written_tokens_on_switches": sum(r.get("cache_creation_input_tokens") or 0 for r in sw),
            "saved_on_switches": _saving(sum(r["cost_usd"] for r in sw), sum(r["cost_if_requested_usd"] for r in sw))}


def threads(rows: list[dict]) -> dict:
    th = [r for r in rows if r.get("thread")]
    return {"thread_requests": len(th), "by_mode": dict(collections.Counter(r["thread"] for r in th)),
            "by_status": dict(collections.Counter(str(r.get("status")) for r in th)),
            "mismatch_rejections": sum(1 for r in th if "differing field" in (r.get("error") or "")),
            "expired": sum(1 for r in th if "No thread state" in (r.get("error") or ""))}


def session(rows: list[dict], session_id: str) -> dict:
    mine = [r for r in rows if (r.get("claude_session") or "").startswith(session_id)]
    return {"session": session_id, "requests": len(mine), "models": dict(collections.Counter(r.get("routed_model") for r in mine)),
            "errors": sum(1 for r in mine if r.get("status") != 200), **{k: v for k, v in savings(mine).items() if k == "vs_no_router"}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("savings", "errors", "switches", "threads", "session"))
    ap.add_argument("id", nargs="?", help="session id prefix (for session)")
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args(argv)
    rows = load(args.days)
    if args.what == "session" and not args.id:
        ap.error("session needs an id")
    result = session(rows, args.id) if args.what == "session" else globals()[args.what](rows)
    print(json.dumps({"window_days": args.days, **result}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
