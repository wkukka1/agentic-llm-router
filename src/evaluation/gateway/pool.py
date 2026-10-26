"""A prompt pool for routing data beyond one user's traffic: real coding
requests, in the gateway's ``turns.jsonl`` shape, for ``aar-gateway-grade
--turns <pool>/turns.jsonl --out <pool>/graded.jsonl`` to replay on each
Claude tier and judge.

    python -m evaluation.gateway.pool arena --n 120
    python -m evaluation.gateway.pool swe --n 80
    python -m evaluation.gateway.pool swe-smith --n 25 --pool-dir ~/.cache/aar/routing-pool/swe-smith

Three sources:

* ``arena`` -- single-turn coding prompts people sent to LMArena in 2025
  (``lmarena-ai/arena-human-preference-140k``, CC-BY-4.0, ``is_code``
  rows, English). Self-contained: replayed in an empty folder.
* ``swe`` -- SWE-bench Verified issues (``princeton-nlp/SWE-bench_Verified``),
  each replayed inside its repository checked out at the issue's base
  commit, asking for the root cause and a fix. The fix the maintainers
  merged travels as ``reference``, so the judge grades against a known-good
  answer instead of its own opinion.
* ``swe-smith`` -- the SWE-smith tasks Claude worked in
  ``thoughtworks/agentic-coding-trajectories`` (only its first message is
  typed by a person; the rest is a tool loop, so the task is the prompt).
  Each is replayed on the files of its task branch of the ``swesmith/``
  mirror, the bug in place, without the branch's history; the reference is the bug's patch reversed. Claude 3.7 Sonnet's
  resolve rate on the task travels as ``claude_resolved``.

Every row carries ``group`` (the repository, or ``arena``) so train and test
splits can be made by group: a router tested on repositories it trained on
looks better than it is.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import subprocess
import sys
import tarfile
from pathlib import Path

from training.data.download import fetch as _download

POOL_DIR = Path.home() / ".cache" / "aar" / "routing-pool"
ARENA_URL = ("https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k/resolve/main/"
             "data/train-0000{shard}-of-00007.parquet")
SWE_URL = ("https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/"
           "refs%2Fconvert%2Fparquet/default/test/0000.parquet")
MIN_CHARS, MAX_CHARS = 40, 4000

SWE_TASK = """You are working in this repository, checked out at the commit where the issue below was reported.
Investigate it in the code: find the root cause, then give the fix as a unified diff with a short
explanation of why it is correct. Do not run the test suite.

<issue>
{issue}
</issue>"""


def _row(prompt: str, **extra) -> dict:
    return {"prompt_hash": hashlib.sha1(prompt.encode()).hexdigest()[:16], "prompt": prompt, **extra}


def _user_text(message: dict) -> str | None:
    parts = message.get("content")
    if parts is None:
        return None
    if any(p.get("type") != "text" for p in parts):  # images etc.: not replayable as text
        return None
    return "\n".join(p.get("text") or "" for p in parts).strip()


def arena_rows(n: int, seed: int = 0, pool_dir: Path = POOL_DIR, shards: int = 1) -> list[dict]:
    """``n`` distinct single-turn English coding prompts from LMArena."""
    import pandas as pd

    frames = []
    for shard in range(shards):
        path = _download(ARENA_URL.format(shard=shard), pool_dir / "raw" / f"arena-{shard}.parquet")
        frames.append(pd.read_parquet(path, columns=["is_code", "language", "conversation_a", "conv_metadata"]))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["is_code"] & (df["language"] == "en")]
    seen: dict[str, dict] = {}
    for conv, meta in zip(df["conversation_a"], df["conv_metadata"], strict=True):
        if (meta or {}).get("turns") != 1:
            continue
        users = [m for m in conv if m.get("role") == "user"]
        text = _user_text(users[0]) if len(users) == 1 else None
        if not text or not MIN_CHARS <= len(text) <= MAX_CHARS:
            continue
        row = _row(text, source="arena", group="arena", cwd=None)
        seen.setdefault(row["prompt_hash"], row)
    rows = sorted(seen.values(), key=lambda r: r["prompt_hash"])
    random.Random(seed).shuffle(rows)
    return rows[:n]


def _git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _clone(repo: str, pool_dir: Path) -> Path:
    clone = pool_dir / "repos" / repo.replace("/", "__")
    if not clone.exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--quiet", "--no-checkout", f"https://github.com/{repo}.git", str(clone))
    return clone


def _worktree(repo: str, commit: str, instance_id: str, pool_dir: Path) -> Path:
    """The repository at ``commit``, in a folder of its own (git worktrees
    share one clone's objects)."""
    clone = _clone(repo, pool_dir)
    tree = pool_dir / "worktrees" / instance_id
    if not tree.exists():
        _git("worktree", "add", "--detach", "--quiet", str(tree), commit, cwd=clone)
    return tree


def _snapshot(repo: str, ref: str, instance_id: str, pool_dir: Path) -> Path:
    """The files at ``ref`` as a fresh one-commit repository. A SWE-smith task
    branch carries its bug as a "Bug Patch" commit: found in the pilot, Haiku
    read it back with ``git show`` and answered ``git revert``."""
    tree = pool_dir / "snapshots" / instance_id
    if not tree.exists():
        archive = subprocess.run(["git", "archive", ref], cwd=_clone(repo, pool_dir), check=True,
                                 capture_output=True).stdout
        tree.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(tree, filter="data")
        _git("init", "--quiet", cwd=tree)
        _git("add", "-A", cwd=tree)
        _git("-c", "user.name=aar", "-c", "user.email=aar@localhost", "commit", "--quiet", "-m", "snapshot",
             cwd=tree)
    return tree


def _spread(recs: list[dict], key: str, n: int, seed: int) -> list[dict]:
    """``n`` records taken round-robin over ``key``, so every repository is represented."""
    rng = random.Random(seed)
    by_key: dict[str, list[dict]] = {}
    for rec in recs:
        by_key.setdefault(rec[key], []).append(rec)
    for group in by_key.values():
        rng.shuffle(group)
    picked: list[dict] = []
    while len(picked) < n and any(by_key.values()):
        for group in by_key.values():
            if group and len(picked) < n:
                picked.append(group.pop())
    return picked


def swe_rows(n: int, seed: int = 0, pool_dir: Path = POOL_DIR,
             repos: list[str] | None = None) -> list[dict]:
    """``n`` SWE-bench Verified issues, spread across repositories, each
    checked out at its base commit."""
    import pandas as pd

    df = pd.read_parquet(_download(SWE_URL, pool_dir / "raw" / "swe-bench-verified.parquet"))
    if repos:
        df = df[df["repo"].isin(repos)]
    picked = _spread(df.to_dict("records"), "repo", n, seed)
    rows = []
    for i, rec in enumerate(picked):
        print(f"  [{i + 1}/{len(picked)}] {rec['instance_id']}", file=sys.stderr)
        try:
            tree = _worktree(rec["repo"], rec["base_commit"], rec["instance_id"], pool_dir)
        except subprocess.CalledProcessError as exc:
            print(f"    skipped: {exc.stderr.decode(errors='replace')[:200]}", file=sys.stderr)
            continue
        rows.append(_row(SWE_TASK.format(issue=rec["problem_statement"].strip()), source="swe-bench-verified",
                         group=rec["repo"], cwd=str(tree), reference=rec["patch"],
                         instance_id=rec["instance_id"], difficulty=rec.get("difficulty")))
    return rows


_HUNK = re.compile(r"^@@ -(\S+) \+(\S+) @@")


def reverse_patch(patch: str) -> str:
    """``patch`` undone: added and removed lines, hunk ranges and file headers swapped."""
    lines = patch.splitlines(keepends=True)
    out = []
    for i, line in enumerate(lines):
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            out += ["--- " + lines[i + 1][4:], "+++ " + line[4:]]
        elif line.startswith("+++ ") and out and out[-1].startswith("+++ "):
            continue  # emitted with its "--- " line
        elif m := _HUNK.match(line):
            out.append(f"@@ -{m[2]} +{m[1]} @@{line[m.end():]}")
        elif line.startswith("new file mode"):
            out.append("deleted" + line[len("new"):])
        elif line.startswith("deleted file mode"):
            out.append("new" + line[len("deleted"):])
        elif line[:1] in ("+", "-"):
            out.append(("-" if line[0] == "+" else "+") + line[1:])
        else:
            out.append(line)
    return "".join(out)


TRAJECTORIES = "thoughtworks/agentic-coding-trajectories"
SWE_SMITH = "SWE-bench/SWE-smith"


def _upstream(mirror: str) -> str:
    """``swesmith/pandas-dev__pandas.95280573`` -> ``pandas-dev/pandas``."""
    return mirror.split("/", 1)[1].rsplit(".", 1)[0].replace("__", "/", 1)


def swe_smith_rows(n: int, seed: int = 0, pool_dir: Path = POOL_DIR,
                   repos: list[str] | None = None) -> list[dict]:
    """``n`` SWE-smith tasks that Claude attempted in the trajectory corpus,
    spread across repositories, each on its task branch."""
    import pandas as pd
    from huggingface_hub import hf_hub_download, snapshot_download

    raw = pool_dir / "raw"
    sessions = pd.read_parquet(
        hf_hub_download(TRAJECTORIES, "sessions.parquet", repo_type="dataset", local_dir=raw / "thoughtworks-act"),
        columns=["recorded_model", "ground_truth_meta_json"])
    resolved: dict[str, list[bool]] = {}
    for model, meta in zip(sessions["recorded_model"], sessions["ground_truth_meta_json"], strict=True):
        if str(model).startswith("claude"):
            meta = json.loads(meta)
            resolved.setdefault(meta["instance_id"], []).append(bool(meta.get("resolved")))
    smith = snapshot_download(SWE_SMITH, repo_type="dataset", allow_patterns="data/*.parquet",
                              local_dir=raw / "swe-smith")
    df = pd.concat(pd.read_parquet(p, columns=["instance_id", "patch", "repo", "problem_statement"])
                   for p in sorted(Path(smith, "data").glob("*.parquet")))
    df = df[df["instance_id"].isin(resolved.keys()) & (df["problem_statement"].str.strip() != "")]
    df = df.assign(upstream=df["repo"].map(_upstream))
    if repos:
        df = df[df["upstream"].isin(repos)]
    picked = _spread(df.to_dict("records"), "upstream", n, seed)
    rows = []
    for i, rec in enumerate(picked):
        iid = rec["instance_id"]
        print(f"  [{i + 1}/{len(picked)}] {iid}", file=sys.stderr)
        try:
            tree = _snapshot(rec["repo"], f"origin/{iid}", iid, pool_dir)
        except subprocess.CalledProcessError as exc:
            print(f"    skipped: {exc.stderr.decode(errors='replace')[:200]}", file=sys.stderr)
            continue
        runs = resolved[iid]
        rows.append(_row(SWE_TASK.format(issue=rec["problem_statement"].strip()), source="swe-smith",
                         group=rec["upstream"], cwd=str(tree), reference=reverse_patch(rec["patch"]),
                         instance_id=iid, claude_resolved=round(sum(runs) / len(runs), 3)))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaluation.gateway.pool", description=__doc__.split("\n\n")[0])
    ap.add_argument("source", choices=("arena", "swe", "swe-smith"))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-dir", default=str(POOL_DIR))
    ap.add_argument("--repos", help="swe: comma-separated owner/name filter")
    args = ap.parse_args(argv)
    pool_dir = Path(args.pool_dir).expanduser()
    repos = args.repos.split(",") if args.repos else None
    if args.source == "arena":
        rows = arena_rows(args.n, args.seed, pool_dir)
    elif args.source == "swe":
        rows = swe_rows(args.n, args.seed, pool_dir, repos)
    else:
        rows = swe_smith_rows(args.n, args.seed, pool_dir, repos)
    from aar_gateway.grading.history import merge_into

    added = merge_into(pool_dir / "turns.jsonl", rows)
    print(f"added {added} {args.source} prompt(s) to {pool_dir / 'turns.jsonl'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
