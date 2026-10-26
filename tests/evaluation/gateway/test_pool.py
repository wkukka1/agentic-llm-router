"""The routing-data prompt pool: which LMArena rows become prompts, and how
the judge uses a reference solution and labels "matches the best"."""

from __future__ import annotations

import pandas as pd

from evaluation.gateway import pool
from aar_gateway.grading.grade import judge_prompt, parse_verdicts


def msg(role, text, kind="text"):
    return {"role": role, "content": [{"type": kind, "text": text}]}


def test_arena_keeps_single_turn_english_text_coding_prompts_once(tmp_path, monkeypatch):
    good = "How do I reverse a linked list in Python without recursion?"
    rows = [
        (True, "en", [msg("user", good), msg("assistant", "...")], {"turns": 1}),
        (True, "en", [msg("user", good), msg("assistant", "...")], {"turns": 1}),       # duplicate vote
        (False, "en", [msg("user", "Write me a poem about the sea, please")], {"turns": 1}),  # not code
        (True, "de", [msg("user", "Wie sortiere ich eine Liste in Python?")], {"turns": 1}),
        (True, "en", [msg("user", "ok")], {"turns": 1}),                                   # too short
        (True, "en", [msg("user", good + " (two)"), msg("assistant", "a"), msg("user", "more")], {"turns": 2}),
        (True, "en", [msg("user", "What is in this screenshot of my code?", kind="image")], {"turns": 1}),
    ]
    path = tmp_path / "raw" / "arena-0.parquet"
    path.parent.mkdir()
    pd.DataFrame(rows, columns=["is_code", "language", "conversation_a", "conv_metadata"]).to_parquet(path)
    monkeypatch.setattr(pool, "_download", lambda url, dest: path)
    (row,) = pool.arena_rows(10, pool_dir=tmp_path)
    assert (row["prompt"], row["source"], row["group"], row["cwd"]) == (good, "arena", "arena", None)


def test_judge_gets_the_reference_and_reports_matches_best():
    text = judge_prompt("fix the bug", {"A": "patch a", "B": "patch b"}, reference="--- a/x.py\n+++ b/x.py")
    assert "=== Reference solution ===" in text and "maintainers" in text
    assert "Reference solution" not in judge_prompt("fix the bug", {"A": "x"})
    verdict = parse_verdicts('{"A": {"score": 9, "matches_best": true, "reason": "r"}, '
                             '"B": {"score": 4, "matches_best": "maybe", "reason": "q"}}', ["A", "B"])
    assert verdict == {"A": (0.9, "r", True), "B": (0.4, "q", None)}


BUG = """diff --git a/pkg/col.py b/pkg/col.py
index c27a9d8..ccf989f 100644
--- a/pkg/col.py
+++ b/pkg/col.py
@@ -10,3 +10,3 @@ class Column:
     @property
-    def null_count(self) -> int:
-        return int(self._col.isna().sum())
+    def null_count(self):
+        return self._col.isna().sum()
"""


def test_reverse_patch_turns_a_bug_into_its_fix():
    fix = pool.reverse_patch(BUG)
    assert "-    def null_count(self):\n" in fix and "+        return int(self._col.isna().sum())\n" in fix
    assert "--- b/pkg/col.py\n+++ a/pkg/col.py\n@@ -10,3 +10,3 @@ class Column:\n" in fix
    assert pool.reverse_patch(fix) == BUG
    created = "--- /dev/null\n+++ b/new.py\nnew file mode 100644\n@@ -0,0 +1 @@\n+x = 1\n"
    assert pool.reverse_patch(created) == "--- b/new.py\n+++ /dev/null\ndeleted file mode 100644\n@@ -1 +0,0 @@\n-x = 1\n"


def test_swe_smith_takes_claude_tasks_on_their_branch_with_the_fix_as_reference(tmp_path, monkeypatch):
    import json
    import huggingface_hub

    meta = lambda iid, ok: json.dumps({"instance_id": iid, "resolved": ok})  # noqa: E731
    raw = tmp_path / "raw"
    (raw / "swe-smith" / "data").mkdir(parents=True)
    pd.DataFrame({"recorded_model": ["claude-3-7-sonnet-20250219", "claude-3-7-sonnet-20250219", "unknown"],
                  "ground_truth_meta_json": [meta("o__r.abc.t1", True), meta("o__r.abc.t1", False),
                                             meta("o__r.abc.t2", True)]}).to_parquet(raw / "sessions.parquet")
    pd.DataFrame({"instance_id": ["o__r.abc.t1", "o__r.abc.t2"], "patch": [BUG, BUG],
                  "repo": ["swesmith/o__r.abc"] * 2,
                  "problem_statement": ["null_count is not an int", "other"]}
                 ).to_parquet(raw / "swe-smith" / "data" / "train-0.parquet")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(raw / "sessions.parquet"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(raw / "swe-smith"))
    trees = []
    monkeypatch.setattr(pool, "_snapshot", lambda repo, ref, iid, d: trees.append((repo, ref)) or d / iid)
    (row,) = pool.swe_smith_rows(5, pool_dir=tmp_path)
    assert trees == [("swesmith/o__r.abc", "origin/o__r.abc.t1")]  # t2 was never run by Claude
    assert (row["source"], row["group"], row["claude_resolved"]) == ("swe-smith", "o/r", 0.5)
    assert "null_count is not an int" in row["prompt"] and row["reference"] == pool.reverse_patch(BUG)


def test_snapshot_has_the_task_files_but_not_the_bug_commit(tmp_path):
    import subprocess

    git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=clone,  # noqa: E731
                                    check=True, capture_output=True, text=True).stdout
    clone = tmp_path / "repos" / "swesmith__o__r.abc"
    clone.mkdir(parents=True)
    git("init", "--quiet")
    (clone / "col.py").write_text("return int(x)\n")
    git("add", "-A"), git("commit", "-qm", "Initial commit")
    git("checkout", "-qb", "task")
    (clone / "col.py").write_text("return x\n")
    git("commit", "-qam", "Bug Patch")
    tree = pool._snapshot("swesmith/o__r.abc", "task", "o__r.abc.t1", tmp_path)
    assert (tree / "col.py").read_text() == "return x\n"
    log = subprocess.run(["git", "log", "--format=%s"], cwd=tree, capture_output=True, text=True).stdout
    assert log.split() == ["snapshot"]
