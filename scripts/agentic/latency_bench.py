"""Latency bench for ``AgenticRouter``: replay multi-turn conversations turn by
turn and time each stage of request -> routing decision -> answer.

Turn ``k`` of a conversation is sent with the real turns ``1..k-1`` (user +
replayed assistant) in front of it. Where ``AgenticRouter.run`` accepts a
message list (``router.agentic.conversation`` exists) the list is passed as-is;
otherwise -- the pre-multi-turn baseline -- the user turns are joined with
blank lines, which is exactly how the router's training data rendered a
conversation.

Stage times:

* ``construct``   -- ``NIRTRouter.from_run`` (checkpoint load)
* ``warmup``      -- ``AgenticRouter.warmup()`` when it exists and ``--no-warmup``
                     isn't given
* ``first_route`` -- the first routing decision (cold caches, if no warm-up)
``--db runs.sqlite`` also records every run (see ``router.tracing.run_store``).

* per turn: ``route`` (``route_decision`` alone, measured separately) and
  ``total`` (``run``), plus every key of ``AgenticResult.timings`` when the
  result carries one (``route`` / ``triage`` / ``dispatch`` / ``ttft``)

Echo clients by default (no network). ``--anthropic`` serves every pool model
through ``ClientRegistry.anthropic`` using ``--serve-map``.

    python scripts/agentic/latency_bench.py
    python scripts/agentic/latency_bench.py --limit 20 --json out.json
    python scripts/agentic/latency_bench.py --anthropic --limit 5
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from router.agentic.conversation import as_messages, routing_text  # noqa: E402


def load_set(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def _accepts_messages() -> bool:
    return importlib.util.find_spec("router.agentic.conversation") is not None


def turn_input(conv: dict, k: int, *, as_messages: bool):
    """What ``run`` receives for turn ``k`` (0-based) of ``conv``."""
    turns = conv["turns"]
    if as_messages:
        msgs = []
        for t in turns[:k]:
            msgs.append({"role": "user", "content": t["user"]})
            if t.get("assistant"):
                msgs.append({"role": "assistant", "content": t["assistant"]})
        msgs.append({"role": "user", "content": turns[k]["user"]})
        return msgs
    return "\n\n".join(t["user"] for t in turns[: k + 1])


def build_clients(args, pool):
    if not args.anthropic:
        return None
    import yaml

    from router.agentic import ClientRegistry

    cfg = yaml.safe_load(Path(args.serve_map).read_text(encoding="utf-8"))
    return ClientRegistry.anthropic(cfg, pool=pool)


def replay(agent, conv: dict, *, as_messages: bool, stream: bool) -> list[dict]:
    rows = []
    for k in range(len(conv["turns"])):
        x = turn_input(conv, k, as_messages=as_messages)
        t0 = time.perf_counter()
        agent.route_decision(routing_text(as_messages(x)))
        route_only = time.perf_counter() - t0

        kw = {}
        if as_messages and "conversation_id" in inspect.signature(agent.run).parameters:
            kw["conversation_id"] = conv["conversation_id"]
        t0 = time.perf_counter()
        if stream:
            gen = agent.stream(x)
            first = None
            try:
                while True:
                    next(gen)
                    if first is None:
                        first = time.perf_counter() - t0
            except StopIteration as stop:
                result = stop.value
        else:
            first, result = None, agent.run(x, **kw)
        total = time.perf_counter() - t0

        row = {"conversation_id": conv["conversation_id"], "turn": k + 1,
               "mode": result.mode, "model": result.selected_model_id,
               "route_only": route_only, "total": total}
        if first is not None:
            row["first_chunk"] = first
        row.update({f"t_{k2}": v for k2, v in (getattr(result, "timings", None) or {}).items()})
        for step in getattr(result, "steps", []) or []:
            for leaf in step.leaves():
                meta = getattr(leaf, "meta", None) or {}
                if meta.get("input_tokens"):
                    row["input_tokens"] = row.get("input_tokens", 0) + int(meta["input_tokens"])
                    row["cache_read"] = row.get("cache_read", 0) + int(meta.get("cache_read_input_tokens") or 0)
        rows.append(row)
    return rows


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summarize(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, float)})
    out = {}
    for k in keys:
        xs = [r[k] for r in rows if k in r]
        out[k] = {"n": len(xs), "p50_ms": 1e3 * statistics.median(xs),
                  "p95_ms": 1e3 * _pct(xs, 0.95), "mean_ms": 1e3 * statistics.fmean(xs)}
    by_turn = defaultdict(list)
    for r in rows:
        by_turn[min(r["turn"], 4)].append(r["route_only"])
    out["route_only_p50_ms_by_turn"] = {t: 1e3 * statistics.median(v) for t, v in sorted(by_turn.items())}
    hits = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("input_tokens"):
            h = hits[min(r["turn"], 4)]
            h[0] += r["cache_read"]
            h[1] += r["input_tokens"]
    if hits:
        out["cache_read_share_by_turn"] = {t: round(a / b, 3) for t, (a, b) in sorted(hits.items())}
    modes = defaultdict(int)
    for r in rows:
        modes[r["mode"]] += 1
    out["modes"] = dict(modes)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", default=str(REPO / "data/eval/arena_multiturn_test.jsonl"))
    p.add_argument("--run", default="nirt-2d-projected")
    p.add_argument("--limit", type=int, default=None, help="conversations to replay")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--anthropic", action="store_true")
    p.add_argument("--serve-map", default=str(REPO / "configs/serve_anthropic.yaml"))
    p.add_argument("--json", default=None, help="write per-turn rows + summary here")
    p.add_argument("--db", default=None,
                   help="also record every run (prompts, answers, expected vs actual cost) "
                        "to this SQLite file; see router.tracing.run_store")
    args = p.parse_args()

    from router.agentic import AgenticRouter
    from router.routing import NIRTRouter

    stages: dict[str, float] = {}
    t0 = time.perf_counter()
    router = NIRTRouter.from_run(args.run)
    stages["construct"] = time.perf_counter() - t0

    sink = None
    if args.db:
        from router.tracing.run_store import sqlite_sink
        sink = sqlite_sink(args.db, router=args.run)
    agent = AgenticRouter(router, build_clients(args, router.model_ids), result_sink=sink)

    if hasattr(agent, "warmup") and not args.no_warmup:
        t0 = time.perf_counter()
        agent.warmup()
        stages["warmup"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    agent.route_decision("What is the capital of France?")
    stages["first_route"] = time.perf_counter() - t0

    as_messages = _accepts_messages()
    convs = load_set(Path(args.set), args.limit)
    rows = []
    for conv in convs:
        rows.extend(replay(agent, conv, as_messages=as_messages, stream=args.stream))

    summary = {"stages_s": stages, "conversations": len(convs), "turns": len(rows),
               "input": "messages" if as_messages else "joined-user-turns",
               **summarize(rows)}
    if sink is not None:
        from router.agentic.router import flush_bookkeeping
        flush_bookkeeping()                                    # let queued writes finish
        summary["db"] = str(args.db)
    print(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
