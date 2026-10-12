# MOD-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-25, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

(both imports corrected, `load_handlabelled` restored from `ea4d209^`, smoke tests in `tests/prompt_decomposition/test_data.py`). Was reproduced (`known_bugs/_repros/repro_prompt_decomposition_training.py`, section 0; `scan_stale_imports.py`)

## Fix suggested when the bug was filed

Import `from training.prompt_decomposition.data import sources` in both builders,
restore `load_handlabelled` (a plain read of `HANDLABELLED_PATH` into `Example`s
with `capability=row.domain`; the repro's shim is the 3-line version), and add a
smoke test that builds both variants into a `tmp_path` from a tiny fixture
parquet. Fix the `router ...` command names in the messages.

## Regression tests

All of `tests/prompt_decomposition/test_data.py` (new module).

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/training/prompt_decomposition/data/dataset.py`

```diff
--- a/src/training/prompt_decomposition/data/dataset.py
+++ b/src/training/prompt_decomposition/data/dataset.py
@@ -152,7 +152,7 @@ def load_splits(variant: str = "domain_v3", out_dir: Path = PROCESSED_DIR) -> di
     target = out_dir / variant
     missing = [n for n in SPLIT_NAMES if not (target / f"{n}.parquet").exists()]
     if missing:
-        raise FileNotFoundError(f"missing splits {missing} under {target}; run `router build-data`")
+        raise FileNotFoundError(f"missing splits {missing} under {target}; run `prompt-decomposition build-real` or `build-task`")
     return {name: pd.read_parquet(target / f"{name}.parquet") for name in SPLIT_NAMES}
 
 
@@ -214,7 +214,7 @@ def build_task_dataset(
     by default: measured against the random sample they move macro-F1 +0.025 and
     top-1 -0.013, neither of which clears a paired bootstrap at n=1,000.
     """
-    from decompose.classifiers.prompt_decomposition import sources
+    from training.prompt_decomposition.data import sources
 
     # `to_frame` writes the label into `domain` (the canonical field on
     # Example); copy it to `task` so a split file carries an unambiguous name
@@ -265,7 +265,7 @@ def build_real_only_dataset(
     The evaluation rows are the frozen set, matched by prompt text, so results
     stay comparable across runs and across changes to the label pool.
     """
-    from decompose.classifiers.prompt_decomposition import sources
+    from training.prompt_decomposition.data import sources
 
     hand = dedupe(to_frame(sources.load_handlabelled()))
     if merge_domains:
```

`src/training/prompt_decomposition/data/sources.py`

```diff
--- a/src/training/prompt_decomposition/data/sources.py
+++ b/src/training/prompt_decomposition/data/sources.py
@@ -38,6 +38,21 @@ GENERATED_TASKS_PATH = "data/synthetic/generated_tasks.parquet"
 SYNTHETIC_DIR = "data/synthetic"
 
 
+def load_handlabelled(path: str = HANDLABELLED_PATH) -> list[Example]:
+    """Real prompts read and labelled by hand -- the anchor of the whole set.
+
+    Small but irreplaceable: it is the only source that pairs genuine user
+    traffic with a label from every domain, including `personal_life` and
+    `meta_other`, which no benchmark contains at all.
+    """
+    frame = pd.read_parquet(path)
+    return [
+        Example(prompt=r.prompt, source="handlabelled", subset="lmarena",
+                capability=r.domain, meta={"arena_id": r.arena_id})
+        for r in frame.itertuples()
+    ]
+
+
 def load_real_tasks(path: str = REAL_TASKS_PATH) -> list[Example]:
     """1,000 real prompts hand-labelled with task type.
 
```
