# DOMAINCLASSIFIER-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

Regression test `tests/router/test_review_fixes.py::test_default_factory_domain_head_uses_merged_taxonomy`

`default_prompt_decomposer` now calls
`DomainClassifier(paths["domain"], merge_domains=True)`, matching the
merged-taxonomy usage already shown in `prompt_heads.py`'s own module
docstring example.

## Fix suggested when the bug was filed

Pass `merge_domains=True` explicitly at the `DomainClassifier(...)` call
site in `default_prompt_decomposer`, or confirm the unmerged default is
actually intended and update `docs/prompt_decomposition.md` to match.

## Regression tests

`tests/router/test_review_fixes.py:426` `test_default_factory_domain_head_uses_merged_taxonomy`

```python
def test_default_factory_domain_head_uses_merged_taxonomy(monkeypatch):
    """DOMAINCLASSIFIER-001: default_prompt_decomposer must build the domain
    head with merge_domains=True, not the unmerged 10-class default."""
    from decompose.decomposer import default_prompt_decomposer

    calls = []

    class _SpyDomainClassifier:
        name = "domain"

        def __init__(self, run_dir, **kwargs):
            calls.append(kwargs)

        def classify(self, input):
            return []

    monkeypatch.setattr("decompose.classifiers.prompt_heads.DomainClassifier", _SpyDomainClassifier)
    default_prompt_decomposer({"domain": "unused"})
    assert calls == [{"merge_domains": True}]
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/decompose/decomposer.py`

```diff
--- a/src/decompose/decomposer.py
+++ b/src/decompose/decomposer.py
@@ -81,7 +81,7 @@ def default_prompt_decomposer(
         )
     classifiers: list[Classifier] = [SurfaceClassifier()]
     if "domain" in paths:
-        classifiers.append(DomainClassifier(paths["domain"]))
+        classifiers.append(DomainClassifier(paths["domain"], merge_domains=True))
     if "task" in paths:
         classifiers.append(TaskClassifier(paths["task"]))
     if "length" in paths:
```
