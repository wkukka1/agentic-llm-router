"""A multi-turn pool for ``aar-gateway-grade``: turns taken from the middle of
coding conversations, each with its earlier conversation packed in front.

    python -m evaluation.gateway.multiturn own --n 25        # your ~/.claude/projects
    python -m evaluation.gateway.multiturn swe-chat --n 25   # SALT-NLP/SWE-chat (gated: hf auth login)
    python -m evaluation.gateway.multiturn wildchat --n 25   # allenai/WildChat-1M, coding chats
    python -m evaluation.gateway.multiturn convcode --n 25   # ConvCodeWorld/convcodebench

A row's ``prompt`` is what the tiers answer and the judge reads: the last few
exchanges (text only, capped at ``CONTEXT_CHARS``) then the message.
``route_text`` is the message alone -- the gateway routes on the last human
message (``interceptor.human_text``), so offline routers must see only that.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from aar_gateway.grading.history import _TEMP_PROJECTS, PROJECTS_DIR, merge_into
from aar_gateway.translate import REMINDER, text_of

from evaluation.gateway.pool import POOL_DIR, _row

OUT_DIR = POOL_DIR / "multiturn"
CONTEXT_CHARS = 3000
EXCHANGES = 4  # earlier user+assistant pairs kept
MIN_CHARS, MAX_CHARS = 40, 4000
PER_SESSION = 2

# a conversation: [(role, text)], role "user" | "assistant"; a user entry is
# None when it isn't a typed message (a tool result) -- never a turn, never context
Conversation = list[tuple[str, str | None]]
_IDE = re.compile(r"<(ide_\w+)>.*?</\1>", re.S)  # the editor's opened-file / selection notes


def with_context(prior: Conversation, message: str, limit: int = CONTEXT_CHARS) -> str:
    """``message`` after the last ``EXCHANGES`` exchanges of ``prior``,
    the oldest text cut first so the block fits in ``limit`` characters."""
    lines = [f"[{role}] {text.strip()}" for role, text in prior if text and text.strip()]
    lines = lines[-2 * EXCHANGES:]
    block = "\n\n".join(lines)[-limit:]
    if not block:
        return message
    return f"<earlier conversation>\n{block}\n</earlier conversation>\n\n{message}"


def conversation_turns(conv: Conversation, *, source: str, group: str) -> list[dict]:
    """A row per user message that has an earlier user message before it."""
    rows, seen_user = [], False
    for i, (role, text) in enumerate(conv):
        if role != "user" or text is None:
            continue
        if seen_user and MIN_CHARS <= len(text) <= MAX_CHARS:
            row = _row(with_context(conv[:i], text), source=source, group=group, cwd=None)
            rows.append(row | {"route_text": text})
        seen_user = True
    return rows


def claude_code_conversation(lines: Iterable[dict]) -> Conversation:
    """A Claude Code session transcript (``~/.claude/projects/*/*.jsonl``) as
    a conversation: typed messages and the assistant's text, side-chains dropped."""
    conv: Conversation = []
    for rec in lines:
        if rec.get("isSidechain") or rec.get("isMeta"):
            continue
        content = (rec.get("message") or {}).get("content")
        if rec.get("type") == "assistant":
            text = text_of(content, sep="\n").strip()
            if text:
                conv.append(("assistant", text))
        elif rec.get("type") == "user":
            blocks = content if isinstance(content, list) else []
            origin = (rec.get("origin") or {}).get("kind")
            # older transcripts have no ``origin``: a typed message is then one with no tool result
            typed = origin == "human" if origin else not any(b.get("type") == "tool_result" for b in blocks)
            raw = text_of(content, sep="\n").lstrip()
            text = _IDE.sub("", REMINDER.sub("", raw)).strip()
            if typed and text and not raw.startswith(("/", "<command-", "<local-command", "<task-notification")):
                conv.append(("user", text))
            else:
                conv.append(("user", None))
    return conv


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def own_rows(projects_dir: Path = PROJECTS_DIR) -> list[dict]:
    rows = []
    for path in sorted(projects_dir.glob("*/*.jsonl")):
        if not path.parent.name.startswith(_TEMP_PROJECTS):
            rows += conversation_turns(claude_code_conversation(_read_jsonl(path)), source="own", group=path.stem)
    return rows


def swe_chat_rows(n_sessions: int, seed: int, raw: Path) -> list[dict]:
    """Claude Code sessions from SWE-chat; downloads only the transcripts sampled."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo = "SALT-NLP/SWE-chat"
    get = lambda f: Path(hf_hub_download(repo, f, repo_type="dataset", local_dir=raw))  # noqa: E731
    logs = pd.read_parquet(get("session_logs.parquet"))
    sessions = pd.read_parquet(get("sessions.parquet"))
    agent_col = next(c for c in ("agent", "agent_name", "tool", "source") if c in sessions.columns)
    cc = set(sessions.loc[sessions[agent_col].astype(str).str.contains("claude", case=False), "session_id"])
    paths = sorted(p for s, p in zip(logs["session_id"], logs["transcript_path"], strict=True)
                   if s in cc and isinstance(p, str) and p.endswith(".jsonl"))
    random.Random(seed).shuffle(paths)
    rows = []
    for p in paths[:n_sessions]:
        rows += conversation_turns(claude_code_conversation(_read_jsonl(get(p))), source="swe-chat",
                                   group=Path(p).stem)
    return rows


def openai_conversation(messages: Iterable[dict]) -> Conversation:
    conv: Conversation = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        text = content if isinstance(content, str) else text_of(content, sep="\n")
        if role == "assistant":
            if text and text.strip():
                conv.append(("assistant", text))
        elif role == "user":
            conv.append(("user", REMINDER.sub("", text).strip() or None))
        elif role == "tool":
            conv.append(("user", None))
    return conv


# shortcut: a keyword test for "is about code"; swap for an LLM label if non-coding chats slip in
_CODE = ("```", "def ", "function", "error", "code", "python", "javascript", "sql", "class ", "import ")


def wildchat_rows(raw: Path) -> list[dict]:
    """Multi-turn English coding chats from one WildChat-1M shard (real ChatGPT users).
    (novita/agentic_code_dataset_22 was tried first: its sessions hold ~1 human message
    each, the rest tool loops -- nothing to route.)"""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("allenai/WildChat-1M", "data/train-00000-of-00014.parquet",
                           repo_type="dataset", local_dir=raw)
    df = pd.read_parquet(path, columns=["conversation_hash", "conversation", "turn", "language", "toxic"])
    df = df[(df["turn"] >= 2) & (df["language"] == "English") & ~df["toxic"]]
    rows = []
    for h, conv in zip(df["conversation_hash"], df["conversation"], strict=True):
        first = next((m["content"] for m in conv if m["role"] == "user"), "") or ""
        if any(k in first.lower() for k in _CODE):
            rows += conversation_turns(openai_conversation(conv), source="wildchat", group=h)
    return rows


CONVCODE_FIX = "Your code fails. Feedback:\n\n{feedback}\n\nFix the function and give the complete corrected code."


def convcode_rows(raw: Path) -> list[dict]:
    """ConvCodeBench (BigCodeBench + feedback): the task, the model's first
    attempt, then its failing feedback as the next user turn."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("ConvCodeWorld/convcodebench", "data/train-00000-of-00001-86cafbcfcce85026.parquet",
                           repo_type="dataset", local_dir=raw)
    it = pd.read_parquet(path).iloc[0]["CF_EF_FULL_SEF"]["ITER=1"]
    rows = []
    for tid, code, ex, verbal, label in zip(it["task_id"], it["previous_code"], it["execution_feedback"],
                                            it["verbal_feedback"], it["label"], strict=True):
        stub, sep, attempt = code.partition("\n\n    pass\n")  # previous_code = task stub + the attempt
        if label != "fail" or not sep:
            continue
        feedback = f"{(ex or '')[:1200]}\n\n{verbal or ''}".strip()
        conv: Conversation = [("user", f"Implement this function:\n\n{stub}{sep}"), ("assistant", attempt),
                              ("user", CONVCODE_FIX.format(feedback=feedback))]
        rows += conversation_turns(conv, source="convcode", group=tid)
    return rows


def sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """``n`` rows, at most ``PER_SESSION`` from one conversation, seeded."""
    rows = sorted(rows, key=lambda r: r["prompt_hash"])
    random.Random(seed).shuffle(rows)
    per: dict[str, int] = {}
    out = []
    for r in rows:
        if len(out) < n and per.get(r["group"], 0) < PER_SESSION:
            per[r["group"]] = per.get(r["group"], 0) + 1
            out.append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaluation.gateway.multiturn", description=__doc__.split("\n\n")[0])
    ap.add_argument("source", choices=("own", "swe-chat", "wildchat", "convcode"))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    raw = out_dir / "raw"
    rows = {"own": lambda: own_rows(),
            "swe-chat": lambda: swe_chat_rows(4 * args.n, args.seed, raw),
            "wildchat": lambda: wildchat_rows(raw),
            "convcode": lambda: convcode_rows(raw)}[args.source]()
    picked = sample(rows, args.n, args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    added = merge_into(out_dir / "turns.jsonl", picked)
    print(f"{len(rows)} candidate turn(s); added {added} {args.source} turn(s) to {out_dir / 'turns.jsonl'}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
