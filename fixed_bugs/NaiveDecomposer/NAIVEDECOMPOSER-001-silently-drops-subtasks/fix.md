# NAIVEDECOMPOSER-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** date not recorded, committed in `cfb7007`
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

Short parts are merged into a neighbour and parts past `max_parts` are folded into the last one, in both `NaiveDecomposer` and `LLMDecomposer._parse` (shared `_fold_tail`). Tests: `test_naive_decomposer_*` and `test_llm_decomposer_folds_lines_past_max_parts_into_the_last` in `tests/router/agentic/test_agentic_router.py`.

## Fix suggested when the bug was filed

Merge a too-short part into its neighbour instead of dropping it, and when there are more than
`max_parts` either fall back to `[prompt]` or fold the tail into the last part. `LLMDecomposer._parse`
truncates at `max_parts=8` the same way.

## Regression tests

`tests/router/agentic/test_agentic_router.py:118` `test_naive_decomposer_splits_and_then`

```python
def test_naive_decomposer_splits_and_then():
    parts = NaiveDecomposer()("Explain quicksort and then implement it in Rust "
                              "and then analyse the complexity")
    assert len(parts) == 3
    assert parts[0].lower().startswith("explain quicksort")
```

`tests/router/agentic/test_agentic_router.py:125` `test_naive_decomposer_declines_single_task`

```python
def test_naive_decomposer_declines_single_task():
    assert NaiveDecomposer()("What is 2 + 2?") == ["What is 2 + 2?"]
```

`tests/router/agentic/test_agentic_router.py:129` `test_naive_decomposer_keeps_a_short_leading_instruction`

```python
def test_naive_decomposer_keeps_a_short_leading_instruction():
    """NAIVEDECOMPOSER-001: "Fix bug" is under min_part_chars but is a real sub-task."""
    assert NaiveDecomposer()("Fix bug. Add tests for the module. Update the docs.") == [
        "Fix bug. Add tests for the module.",
        "Update the docs.",
    ]
```

`tests/router/agentic/test_agentic_router.py:137` `test_naive_decomposer_keeps_a_short_trailing_part`

```python
def test_naive_decomposer_keeps_a_short_trailing_part():
    assert NaiveDecomposer()("Explain quicksort in detail. Write it in Rust. Thanks.") == [
        "Explain quicksort in detail.",
        "Write it in Rust. Thanks.",
    ]
```

`tests/router/agentic/test_agentic_router.py:144` `test_naive_decomposer_declines_when_every_part_is_short`

```python
def test_naive_decomposer_declines_when_every_part_is_short():
    assert NaiveDecomposer()("Do A. Do B.") == ["Do A. Do B."]
```

`tests/router/agentic/test_agentic_router.py:148` `test_naive_decomposer_folds_parts_past_max_parts_into_the_last`

```python
def test_naive_decomposer_folds_parts_past_max_parts_into_the_last():
    """NAIVEDECOMPOSER-001: parts beyond max_parts used to be cut off."""
    prompt = "Please handle every item\n" + "\n".join(f"{i}. Handle item number {i} carefully" for i in range(1, 9))
    parts = NaiveDecomposer()(prompt)
    assert len(parts) == 6
    assert parts[:5] == ["Please handle every item."] + [f"Handle item number {i} carefully." for i in range(1, 5)]
    assert parts[5] == " ".join(f"Handle item number {i} carefully." for i in range(5, 9))
```

`tests/router/agentic/test_agentic_router.py:157` `test_naive_decomposer_max_parts_one_declines_rather_than_truncating`

```python
def test_naive_decomposer_max_parts_one_declines_rather_than_truncating():
    assert NaiveDecomposer(max_parts=1)("Explain quicksort. Then write it in Rust.") == [
        "Explain quicksort. Then write it in Rust."
    ]
```

`tests/router/agentic/test_agentic_router.py:163` `test_llm_decomposer_folds_lines_past_max_parts_into_the_last`

```python
def test_llm_decomposer_folds_lines_past_max_parts_into_the_last():
    """NAIVEDECOMPOSER-001: LLMDecomposer._parse truncated the same way."""
    class Chat:
        def invoke(self, _prompt):
            return "1. first task\n2. second task\n3. third task\n4. fourth task"

    assert LLMDecomposer(Chat(), max_parts=2)("anything") == [
        "first task",
        "second task third task fourth task",
    ]
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/agentic/decompose.py`

```diff
--- a/src/router/agentic/decompose.py
+++ b/src/router/agentic/decompose.py
@@ -44,27 +44,50 @@ _SPLIT_RE = re.compile(
     re.IGNORECASE,
 )
 
-# "1. ", "2) ", "- ", "* " line prefixes an LLM adds despite being told not to
-_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)]|#+)\s*")
+# "1. ", "2) ", "- ", "* " line prefixes an LLM adds despite being told not to. A numbered
+# marker needs trailing whitespace, else it would eat the start of "3.14 times 2".
+_LINE_PREFIX_RE = re.compile(r"^\s*(?:(?:[-*•]+|#+)\s*|\d+[.)]\s+)")
+
+
+def _fold_tail(parts: list[str], max_parts: int) -> list[str]:
+    """Cap ``parts`` at ``max_parts`` by folding the overflow into the last kept
+    part -- truncating would silently drop the later sub-tasks."""
+    if len(parts) <= max_parts:
+        return parts
+    keep = max(max_parts, 1)
+    return parts[: keep - 1] + [" ".join(parts[keep - 1:])]
 
 
 class NaiveDecomposer:
-    """Regex split on list markers / "and then" / question boundaries. No LLM."""
+    """Regex split on list markers / "and then" / question boundaries. No LLM.
+
+    No part of the request is dropped: a fragment shorter than ``min_part_chars``
+    is merged into its neighbour, and parts beyond ``max_parts`` are folded into
+    the last one."""
 
     def __init__(self, *, max_parts: int = 6, min_part_chars: int = 8):
         self.max_parts = max_parts
         self.min_part_chars = min_part_chars
 
     def __call__(self, prompt: str) -> list[str]:
-        raw = _SPLIT_RE.split(prompt.strip())
-        parts = [p.strip(" \t\n-*.") for p in raw if p and len(p.strip()) >= self.min_part_chars]
-        # de-dupe while keeping order
         seen, out = set(), []
-        for p in parts:
-            if p.lower() not in seen:
-                seen.add(p.lower())
-                out.append(p if p.endswith(("?", ".", "!")) else p + ".")
-        return out[: self.max_parts] if len(out) > 1 else [prompt]
+        lead = ""          # short fragments before the first full part, awaiting a neighbour
+        for frag in _SPLIT_RE.split(prompt.strip()):
+            p = frag.strip(" \t\n-*.")
+            if not p or p.lower() in seen:      # empty, or a repeat of an earlier part
+                continue
+            seen.add(p.lower())
+            text = p if p.endswith(("?", ".", "!")) else p + "."
+            if len(p) < self.min_part_chars:
+                if out:
+                    out[-1] += " " + text
+                else:
+                    lead = f"{lead} {text}".strip()
+            else:
+                out.append(f"{lead} {text}".strip())
+                lead = ""
+        out = _fold_tail(out, self.max_parts)
+        return out if len(out) > 1 else [prompt]
 
 
 class LLMDecomposer:
@@ -110,7 +138,7 @@ class LLMDecomposer:
             if len(ln) < 4 or ln.endswith(":"):
                 continue
             parts.append(ln)
-        return parts[: self.max_parts]
+        return _fold_tail(parts, self.max_parts)
 
 
 def concat_synthesizer(original: str, pairs: Sequence[tuple[str, str]]) -> str:
```
