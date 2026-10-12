"""Build the held-out multi-turn conversation set for the agentic latency bench.

Reads Chatbot Arena 55k (``data/raw/arena/arena-human-preference-55k``), keeps
conversations with at least ``--min-turns`` user turns whose query id is in the
``--split`` split (default ``test``: the router was trained on Arena, so only
held-out conversations are fair to route), samples ``--n`` of them stratified
by turn count, and writes one JSON record per line:

    {"conversation_id", "query_id", "turns": [{"user", "assistant"}, ...],
     "model_a", "model_b", "winner"}

``assistant`` is ``response_a``'s turn -- a replayed history, so turn k can be
routed with the real turns 1..k-1 in front of it.

The query id is computed exactly as the training loader does
(``make_query_id(Source.ARENA, <dataset dir>, render_prompt(prompt))``), so
split membership matches the trained router's.

    python scripts/agentic/build_multiturn_set.py
    python scripts/agentic/build_multiturn_set.py --n 50 --split validation
"""

from __future__ import annotations

import ast
import json
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

from training.cli import base_parser, get_config, info, resolve

DATASET_DIR = "arena-human-preference-55k"


def parse_turns(raw: object) -> Optional[list[Optional[str]]]:
    """A ``str(list)`` / JSON-array field -> list of turns (``None`` kept for a
    missing assistant turn); ``None`` if it isn't a list of turns."""
    if isinstance(raw, (list, tuple)):
        return list(raw)
    s = str(raw)
    try:
        parsed = json.loads(s)
    except ValueError:
        with warnings.catch_warnings():          # escape-sequence noise from old dumps
            warnings.simplefilter("ignore", SyntaxWarning)
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                return None
    if isinstance(parsed, list) and all(t is None or isinstance(t, str) for t in parsed):
        return parsed
    return None


def load_conversations(raw_dir: Path) -> Iterator[dict]:
    """Every Arena battle as ``{id, prompt_raw, users, assistants, model_a, model_b, winner}``."""
    from datasets import load_from_disk

    from training.data.normalize import sanitize_text

    ds = load_from_disk(str(raw_dir))
    if hasattr(ds, "keys"):
        ds = ds[list(ds.keys())[0]]
    for ex in ds:
        users = parse_turns(ex["prompt"])
        assistants = parse_turns(ex["response_a"])
        if not users or assistants is None:
            continue
        winner = ("a" if ex.get("winner_model_a") else
                  "b" if ex.get("winner_model_b") else "tie")
        yield {
            "id": str(ex["id"]),
            "prompt_raw": ex["prompt"],
            "users": [sanitize_text(u or "") for u in users],
            "assistants": [sanitize_text(a or "") for a in assistants],
            "model_a": ex["model_a"],
            "model_b": ex["model_b"],
            "winner": winner,
        }


def held_out_query_ids(cfg, split: str) -> set[str]:
    path = cfg.path("splits") / f"{split}.json"
    return set(json.loads(path.read_text(encoding="utf-8"))["query_ids"])


def select(convs: Iterator[dict], ids: set[str], *, n: int, min_turns: int,
           seed: int = 0) -> list[dict]:
    """Conversations in ``ids`` with ``>= min_turns`` user turns, every user
    turn non-empty, sampled up to ``n`` -- stratified by turn count (2, 3, 4+),
    equal share per stratum, leftovers filled from the larger strata."""
    from training.data.normalize import make_query_id, render_prompt
    from training.data.schemas import Source

    strata: dict[int, list[dict]] = defaultdict(list)
    for c in convs:
        k = len(c["users"])
        if k < min_turns or not all(u.strip() for u in c["users"]):
            continue
        qid = make_query_id(Source.ARENA, DATASET_DIR, render_prompt(c["prompt_raw"]))
        if qid not in ids:
            continue
        turns = [{"user": u, "assistant": c["assistants"][i] if i < len(c["assistants"]) else ""}
                 for i, u in enumerate(c["users"])]
        strata[min(k, 4)].append({
            "conversation_id": f"arena55k:{c['id']}",
            "query_id": qid,
            "turns": turns,
            "model_a": c["model_a"],
            "model_b": c["model_b"],
            "winner": c["winner"],
        })

    rng = random.Random(seed)
    for rows in strata.values():
        rng.shuffle(rows)
    keys = sorted(strata)
    share = n // max(len(keys), 1)
    picked = [r for k in keys for r in strata[k][:share]]
    rest = [r for k in keys for r in strata[k][share:]]
    rng.shuffle(rest)
    picked += rest[: max(0, n - len(picked))]
    return sorted(picked, key=lambda r: r["conversation_id"])


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--min-turns", type=int, default=2)
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/eval/arena_multiturn_{split}.jsonl")
    args = p.parse_args()
    cfg = get_config(args)

    raw_dir = cfg.resolve(cfg.sources.arena.local_dir)
    rows = select(load_conversations(raw_dir), held_out_query_ids(cfg, args.split),
                  n=args.n, min_turns=args.min_turns, seed=args.seed)
    out = resolve(args.out.format(split=args.split))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_turns: dict[int, int] = defaultdict(int)
    for r in rows:
        by_turns[min(len(r["turns"]), 4)] += 1
    info(f"wrote {len(rows)} conversations to {out} (by turns, 4 = 4+: {dict(sorted(by_turns.items()))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
