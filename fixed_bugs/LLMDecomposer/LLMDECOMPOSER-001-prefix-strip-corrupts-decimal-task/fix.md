# LLMDECOMPOSER-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, committed in `cfb7007`
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`_LINE_PREFIX_RE`'s numeric-marker alternative now requires trailing whitespace, covered by `tests/router/agentic/test_agentic_router.py::test_llm_decomposer_keeps_leading_decimal_number`; re-verified 2026-09-23 against current source and the `_repros/repro_agentic.py` repro

## Fix suggested when the bug was filed

Require whitespace after the marker for the numeric form: `\d+[.)]\s+`.

## Regression tests

`tests/router/agentic/test_agentic_router.py:185` `test_llm_decomposer_keeps_leading_decimal_number`

```python
def test_llm_decomposer_keeps_leading_decimal_number():
    """LLMDECOMPOSER-001: a task starting with a decimal is not a list marker."""
    reply = "3.14 times 2 equals what?\n2.5 GHz is how many MHz?\n1.2.3 is which version?"
    assert LLMDecomposer(_ReplyChat(reply))("q") == [
        "3.14 times 2 equals what?",
        "2.5 GHz is how many MHz?",
        "1.2.3 is which version?",
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
```
