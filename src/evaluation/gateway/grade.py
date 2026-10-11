"""Grade captured Claude Code turns on the real Claude tiers -- routing data
from the user's own work instead of benchmark stand-ins.

    aar-gateway-grade --n 20        # replay + judge up to 20 ungraded turns

For each captured turn (``~/.aar-gateway/turns.jsonl``, written by the
gateway when ``interceptor.capture_prompts: true``):

1. **Replay** the human's message on each tier (default Opus 5.5, Sonnet 5.5,
   Haiku 5.5) through the ``claude`` CLI on the user's own login, in the turn's
   working directory, read-only (``--permission-mode plan``): the model can
   read the code it was asked about and cannot change anything. Calls go
   straight to Anthropic, never through the gateway.
2. **Judge** the answers in one call: anonymised and shuffled, each scored
   0-10 for whether a careful senior engineer would accept it as a correct,
   complete response to the request.
3. **Append** one row per (turn, model) to ``~/.aar-gateway/graded.jsonl``:
   prompt, model, score in [0, 1], the judge's reason, plan usage. The gateway
   reads this file (:mod:`router.gateway.graded`) and, once there is enough
   of it, routes on these scores instead of the benchmark stand-ins.

Every replay and judge call uses the user's plan: ``--n`` bounds it, and a
turn already graded is never redone.

Known limits: the replay sees the repository as it is now, not as it was at
the turn; read-only means an edit request is graded on its plan, not on an
edit; and the judge is a Claude model, so a family preference is possible --
answers are anonymised and shuffled to limit it, not to remove it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from router.llm.adapters.claude_subscription import ClaudeSubscriptionAdapter

DEFAULT_TIERS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
DEFAULT_JUDGE = "claude-opus-5-5"
#: per answer, in the judge prompt -- long enough for a code review, bounded
#: so one runaway answer can't crowd the others out
ANSWER_CHARS = 12_000
#: replays and the judge run read-only, and without the user's MCP servers --
#: found live: with them loaded, an answer can wander into the servers' auth
#: status, which the judge rightly penalises but which says nothing about the
#: model's answer to the request
READ_ONLY = ("--permission-mode", "plan", "--strict-mcp-config")

JUDGE_INSTRUCTIONS = """You are grading answers to a developer's request to an AI coding assistant.
The answers come from different assistants and are labelled only by letter, in random order.

For EACH answer, score 0-10: would a careful senior engineer accept it as a correct and
complete response to this request, as written? 10 = fully correct and complete; 7 = acceptable
with minor gaps; 4 = partly right or incomplete; 0 = wrong, irrelevant or refused. Judge
correctness and completeness first, then clarity. Do not reward length for its own sake.
Some answers were produced in a read-only mode: for a request to change code, a correct,
specific plan counts as a complete answer.

Also say, for EACH answer, whether it would serve the developer as well as the best answer here
("matches_best": true/false) -- the best answer itself is true; so is any answer that is as correct
and complete, even if worded differently.

Reply with ONLY one JSON object, nothing else:
{"A": {"score": <0-10>, "matches_best": <true|false>, "reason": "<at most 25 words>"}, "B": {...}, ...}"""

REFERENCE_NOTE = """A reference solution follows the request: the change the project's maintainers actually
merged. Judge correctness against it. An answer that fixes the problem a different but correct way
is fine; one that contradicts the reference on the root cause is not correct."""


def _home() -> Path:
    from router.gateway.interceptor import log_path  # same folder as the gateway's own files

    return log_path().parent


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pending_turns(turns: Iterable[dict], graded: Iterable[dict], *,
                  subagents: bool | None = None) -> list[dict]:
    """One turn per distinct prompt that has no grade yet, skipping ones
    marked as needing their conversation (``self_contained: false``).
    ``subagents``: only subagent tasks (``True``), only the user's own
    messages (``False``), or both (``None``)."""
    done = {g["prompt_hash"] for g in graded}
    seen: dict[str, dict] = {}
    for t in turns:
        if t.get("self_contained") is False:  # can't be answered out of its conversation
            continue
        if subagents is not None and bool(t.get("subagent")) != subagents:
            continue
        if t["prompt_hash"] not in done:
            seen.setdefault(t["prompt_hash"], t)
    return list(seen.values())


def parse_scores(text: str, labels: Iterable[str]) -> dict[str, tuple[float | None, str]]:
    """``{label: (score in [0, 1] or None, reason)}`` from the judge's JSON --
    tolerant of prose or fences around it, never invents a score."""
    return {k: (v[0], v[1]) for k, v in parse_verdicts(text, labels).items()}


def parse_verdicts(text: str, labels: Iterable[str]) -> dict[str, tuple[float | None, str, bool | None]]:
    """``{label: (score in [0, 1] or None, reason, matches_best or None)}``."""
    found = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(found.group(0)) if found else {}
    except ValueError:
        data = {}
    out = {}
    for label in labels:
        entry = data.get(label) or {}
        try:
            score = min(max(float(entry.get("score")), 0.0), 10.0) / 10.0
        except (TypeError, ValueError):
            score = None
        matches = entry.get("matches_best")
        out[label] = (score, str(entry.get("reason", ""))[:300], matches if isinstance(matches, bool) else None)
    return out


def judge_prompt(request: str, answers: dict[str, str], reference: str | None = None) -> str:
    parts = [JUDGE_INSTRUCTIONS]
    if reference:
        parts.append(f"\n\n{REFERENCE_NOTE}")
    parts.append(f"\n\n=== Developer request ===\n{request}")
    if reference:
        parts.append(f"\n\n=== Reference solution ===\n{reference[:ANSWER_CHARS]}")
    for label, answer in answers.items():
        parts.append(f"\n\n=== Answer {label} ===\n{answer[:ANSWER_CHARS]}")
    return "".join(parts)


def grade_turn(turn: dict, tiers: tuple[str, ...], judge_model: str, *, timeout: float,
               rng: random.Random) -> list[dict]:
    cwd = turn.get("cwd") if turn.get("cwd") and Path(turn["cwd"]).is_dir() else None
    replays: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="aar-grade-") as scratch:
        for model in tiers:
            try:
                resp = ClaudeSubscriptionAdapter(model, timeout=timeout).complete(
                    turn["prompt"], cwd=cwd or scratch, extra_args=READ_ONLY)
                replays[model] = {"answer": resp.content,
                                  "usage_usd": (resp.meta or {}).get("subscription_cost_usd_equivalent")}
            except Exception as exc:  # one failed replay must not lose the others
                replays[model] = {"answer": None, "error": f"{type(exc).__name__}: {exc}"[:300]}

        answered = [m for m in tiers if replays[m].get("answer")]
        labels = [chr(ord("A") + i) for i in range(len(answered))]
        order = answered[:]
        rng.shuffle(order)
        by_label = dict(zip(labels, order, strict=True))
        scores: dict[str, tuple[float | None, str, bool | None]] = {}
        judge_error = None
        if answered:
            try:
                verdict = ClaudeSubscriptionAdapter(judge_model, timeout=timeout).complete(
                    judge_prompt(turn["prompt"], {lab: replays[m]["answer"] for lab, m in by_label.items()},
                                 turn.get("reference")),
                    cwd=scratch, extra_args=READ_ONLY)
                scores = parse_verdicts(verdict.content, labels)
            except Exception as exc:
                judge_error = f"{type(exc).__name__}: {exc}"[:300]

    model_score = {m: scores.get(lab, (None, "", None)) for lab, m in by_label.items()}
    now = datetime.now(UTC).isoformat(timespec="seconds")
    rows = []
    for model in tiers:
        score, reason, matches = model_score.get(model, (None, "", None))
        rep = replays[model]
        rows.append({
            "ts": now, "prompt_hash": turn["prompt_hash"], "prompt": turn["prompt"],
            "cwd": turn.get("cwd"), "model": model, "score": score, "reason": reason,
            "answer": (rep.get("answer") or "")[:4000], "usage_usd": rep.get("usage_usd"),
            "judge_model": judge_model, "error": rep.get("error") or judge_error,
            "subagent": bool(turn.get("subagent")), "agent_role": turn.get("agent_role"),
            "matches_best": matches, "source": turn.get("source"), "group": turn.get("group"),
            "with_reference": bool(turn.get("reference")),
        })
    return rows


def grade(n: int, *, tiers: tuple[str, ...] = DEFAULT_TIERS, judge_model: str = DEFAULT_JUDGE,
          workers: int = 2, timeout: float = 600.0, seed: int = 0,
          turns_file: Path | None = None, out_file: Path | None = None,
          subagents: bool | None = None) -> list[dict]:
    turns_file = turns_file or _home() / "turns.jsonl"
    out_file = out_file or _home() / "graded.jsonl"
    todo = pending_turns(read_jsonl(turns_file), read_jsonl(out_file), subagents=subagents)
    rng = random.Random(seed)
    rng.shuffle(todo)
    todo = todo[:n]
    print(f"grading {len(todo)} turn(s) on {', '.join(tiers)}; judge {judge_model}", file=sys.stderr)
    written: list[dict] = []

    def one(i_turn):
        i, turn = i_turn
        return grade_turn(turn, tiers, judge_model, timeout=timeout, rng=random.Random(seed + i))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        for rows in pool.map(one, enumerate(todo)):
            with out_file.open("a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            written.extend(rows)
            print(f"  {rows[0]['prompt'][:60]!r}: " + ", ".join(_fmt(r) for r in rows), file=sys.stderr)
    return written


def _fmt(row: dict) -> str:
    score = "-" if row["score"] is None else f"{row['score']:.1f}"
    return f"{row['model']}={score}"


def summarize(rows: list[dict], *, subagents: bool | None = None) -> str:
    by: dict[str, list[float]] = {}
    usage: dict[str, float] = {}
    for r in rows:
        if subagents is not None and bool(r.get("subagent")) != subagents:
            continue
        if r["score"] is not None:
            by.setdefault(r["model"], []).append(r["score"])
        usage[r["model"]] = usage.get(r["model"], 0.0) + (r.get("usage_usd") or 0.0)
    matches: dict[str, list[bool]] = {}
    for r in rows:
        if r.get("matches_best") is not None and (subagents is None or bool(r.get("subagent")) == subagents):
            matches.setdefault(r["model"], []).append(r["matches_best"])
    lines = [f"{'model':20s} {'graded':>6s} {'mean score':>10s} {'accepted (>=0.7)':>16s} "
             f"{'matches best':>12s} {'plan usage $':>12s}"]
    for model, scores in sorted(by.items()):
        m = matches.get(model)
        lines.append(f"{model:20s} {len(scores):6d} {sum(scores) / len(scores):10.2f} "
                     f"{sum(s >= 0.7 for s in scores) / len(scores):16.0%} "
                     f"{(f'{sum(m) / len(m):.0%}' if m else '-'):>12s} {usage.get(model, 0.0):12.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aar-gateway-grade", description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=10, help="max ungraded turns to grade (each costs plan usage)")
    ap.add_argument("--tiers", default=",".join(DEFAULT_TIERS))
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per CLI call")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary", action="store_true", help="only summarize what is already graded")
    ap.add_argument("--turns", help="turns to grade (default ~/.aar-gateway/turns.jsonl); "
                                    "e.g. a prompt pool from evaluation.gateway.pool")
    ap.add_argument("--out", help="graded rows (default ~/.aar-gateway/graded.jsonl, which the gateway reads)")
    ap.add_argument("--from-history", action="store_true",
                    help="first add your past Claude Code messages (~/.claude/projects) to the turns to grade")
    ap.add_argument("--from-subagents", action="store_true",
                    help="first add past subagent tasks that can still be replayed (their folder and "
                         "input files exist) to the turns to grade")
    ap.add_argument("--subagents", choices=("only", "exclude"),
                    help="grade and summarize only subagent tasks, or only your own messages")
    args = ap.parse_args(argv)
    if args.from_history:
        from .history import history_turns, mark_self_contained, merge_into

        have = {t["prompt_hash"] for t in read_jsonl(_home() / "turns.jsonl")}
        fresh = [r for r in history_turns() if r["prompt_hash"] not in have]
        added = merge_into(_home() / "turns.jsonl", mark_self_contained(fresh))
        print(f"added {added} past message(s) from Claude Code's history", file=sys.stderr)
    if args.from_subagents:
        from .history import merge_into
        from .subagent_baseline import subagent_turns

        added = merge_into(_home() / "turns.jsonl", subagent_turns())
        print(f"added {added} past subagent task(s)", file=sys.stderr)
    only = {"only": True, "exclude": False}.get(args.subagents)
    turns_file = Path(args.turns).expanduser() if args.turns else None
    out_file = Path(args.out).expanduser() if args.out else _home() / "graded.jsonl"
    if not args.summary:
        grade(args.n, tiers=tuple(t.strip() for t in args.tiers.split(",") if t.strip()),
              judge_model=args.judge, workers=args.workers, timeout=args.timeout, seed=args.seed,
              subagents=only, turns_file=turns_file, out_file=out_file)
    print(summarize(read_jsonl(out_file), subagents=only))
    return 0


if __name__ == "__main__":
    sys.exit(main())
