"""The user's past Claude Code messages, from Claude Code's own local session
transcripts (``~/.claude/projects/*/*.jsonl``) -- grading material from day
one, instead of waiting for the gateway to capture new turns.

Only what the person typed: transcript lines marked ``origin.kind ==
"human"``, outside subagent side-chains. Dropped: tool results, Claude Code's
``<system-reminder>`` text, slash commands and local-command output, task
notifications, sessions run in temporary folders (scripts, the grader's own
replays), duplicates (a continued session repeats its history), and replies
too short to grade ("continue", "yes", "#").
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Optional

PROJECTS_DIR = Path.home() / ".claude" / "projects"
MIN_CHARS = 20
_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
_TEMP_PROJECTS = ("-private-tmp", "-private-var-folders", "-tmp-")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content or []
                     if isinstance(b, dict) and b.get("type") == "text")


def _typed(record: dict) -> Optional[str]:
    if record.get("type") != "user" or record.get("isSidechain") or record.get("isMeta"):
        return None
    if (record.get("origin") or {}).get("kind") != "human":
        return None
    text = _REMINDER.sub("", _text((record.get("message") or {}).get("content"))).strip()
    if len(text) < MIN_CHARS or text.startswith(("/", "<command-", "<local-command", "<task-notification")):
        return None
    return text


def history_turns(projects_dir: Path = PROJECTS_DIR) -> list[dict]:
    """Rows in the gateway's ``turns.jsonl`` shape, oldest first, one per
    distinct message."""
    out: dict[str, dict] = {}
    for transcript in sorted(projects_dir.glob("*/*.jsonl")):
        if transcript.parent.name.startswith(_TEMP_PROJECTS):
            continue
        for line in transcript.open(encoding="utf-8", errors="replace"):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            text = _typed(record)
            if text is None:
                continue
            h = hashlib.sha1(text.encode()).hexdigest()[:16]
            out.setdefault(h, {"ts": record.get("timestamp"), "prompt_hash": h, "prompt": text,
                               "cwd": record.get("cwd"), "source": "history"})
    return sorted(out.values(), key=lambda r: r.get("ts") or "")


def merge_into(turns_file: Path, rows: Iterable[dict]) -> int:
    """Append the rows ``turns_file`` doesn't already have; returns how many."""
    have = set()
    if turns_file.exists():
        have = {json.loads(line)["prompt_hash"] for line in turns_file.read_text(encoding="utf-8").splitlines()
                if line.strip()}
    new = [r for r in rows if r["prompt_hash"] not in have]
    turns_file.parent.mkdir(parents=True, exist_ok=True)
    with turns_file.open("a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    return len(new)


SELF_CONTAINED_PROMPT = """Below are numbered messages a developer typed to an AI coding assistant, each
taken out of its conversation. For each, decide: could a capable assistant understand what is being
asked and give a useful answer from this message alone (plus read access to the developer's
repository), without the earlier conversation? Follow-ups like "do it", "okay test it out", "yes",
or ones that refer to something only said earlier ("that", "the second option", "what you
proposed") are NOT self-contained.

Reply with ONLY a JSON array of the numbers of the self-contained messages, e.g. [1, 4, 7]."""


def mark_self_contained(rows: list[dict], model: str = "claude-haiku-5-5", batch: int = 40) -> list[dict]:
    """Adds ``self_contained`` (bool) to each row, asking a cheap model in
    batches -- grading a message nobody could answer out of context would
    only add noise. A batch whose reply can't be parsed is left unmarked."""
    from router.llm.adapters.claude_subscription import ClaudeSubscriptionAdapter

    llm = ClaudeSubscriptionAdapter(model, timeout=600)
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        listing = "\n\n".join(f"[{i + 1}] {r['prompt'][:1500]}" for i, r in enumerate(chunk))
        reply = llm.complete(f"{SELF_CONTAINED_PROMPT}\n\n{listing}",
                             extra_args=("--permission-mode", "plan", "--strict-mcp-config")).content
        found = re.search(r"\[[\d,\s]*\]", reply or "")
        if not found:
            continue
        keep = set(json.loads(found.group(0)))
        for i, r in enumerate(chunk):
            r["self_contained"] = (i + 1) in keep
    return rows
