# Code review: decompose

**Files reviewed:** 4 (139 lines of diff-relevant code): `src/decompose/decomposer.py`, `src/decompose/classifiers/base.py`, `src/decompose/classifiers/prompt_heads.py`, `src/decompose/__init__.py` · **Context-only:** `src/decompose/signals.py`, `src/decompose/conversation.py`, `src/decompose/embedders/{base,bert,sentence_transformer}.py`, `src/decompose/classifiers/prompt_decomposition/**` (heads/base.py, heads/domain.py, heads/task.py, heads/length.py, heads/attributes.py, composite.py, `__init__.py`), `src/router/router.py`, `docs/prompt_decomposition.md`

**Summary:** 5 findings: 1 correctness, 3 design, 2 performance (one overlaps design) · 4 confirmed, 1 plausible

**Prior-finding recurrence check:** Signal-name collision — **not reintroduced**: the five heads emit disjoint signal names (`surface`, `domain`/`domain_shortlist`, `task`, `expected_tokens`/`length_bucket`, `gates`/`criteria`); `PromptSignals.add` still raises on a genuine duplicate (tested). Eager-loading-as-a-property-side-effect — **not reintroduced**: model loading happens in `__init__` (an obvious, expected place to pay that cost), not behind an innocuous attribute read like `.dimension`.

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| DC-01 | `src/decompose/decomposer.py:75-84` | correctness | medium | confirmed |
| DC-02 | `src/decompose/decomposer.py:76-84` | performance | medium | confirmed |
| DC-03 | `src/decompose/decomposer.py:78` | design | medium | plausible |
| DC-04 | `src/decompose/decomposer.py:1-16`, `src/router/router.py:56` | design | low | confirmed |
| DC-05 | `src/decompose/classifiers/prompt_heads.py:136-144` | performance | low | confirmed |

---

## src/decompose/decomposer.py

### DC-01: An unrecognized/misspelled `artifact_paths` key is silently dropped, not rejected `:75-84`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
75:    paths = dict(artifact_paths or {})
76:    classifiers: list[Classifier] = [SurfaceClassifier()]
77:    if "domain" in paths:
78:        classifiers.append(DomainClassifier(paths["domain"]))
79:    if "task" in paths:
80:        classifiers.append(TaskClassifier(paths["task"]))
81:    if "length" in paths:
82:        classifiers.append(LengthClassifier(paths["length"]))
83:    if "attributes" in paths:
84:        classifiers.append(AttributeClassifier(paths["attributes"]))
```

**What breaks:** `default_prompt_decomposer({"domains": "artifacts/v7/PROD_ensemble"})` (plural, or any other typo of `"domain"`/`"task"`/`"length"`/`"attributes"`) matches none of the four `if` checks. No exception, no warning — the function returns a `PromptDecomposer` that looks fully constructed (`isinstance(decomposer, PromptDecomposer)` passes, per the factory's own test) but silently contains only `SurfaceClassifier()`. A caller who believes they wired up the domain head gets no `domain` signal in `PromptSignals` and no indication why. The four checked keys are also never enumerated anywhere (no `KNOWN_KEYS` set, no `set(paths) - {"domain","task","length","attributes"}` check), so there's nothing else in the function that would catch this.
**Evidence:** Read the full function body; no `else`/`raise`/logging path exists for keys that don't match one of the four literals.
**Direction:** Validate `paths.keys()` against `{"domain", "task", "length", "attributes"}` up front and raise (or at least log) on anything else, e.g. `if unknown := set(paths) - {"domain","task","length","attributes"}: raise ValueError(f"unknown artifact_paths keys: {unknown}")`.

---

### DC-02: Wiring domain/length/attribute heads through `default_prompt_decomposer` drops the shared-embedding optimization `RouterHead` already implements, doubling encoder passes `:76-84`
**Category:** performance · **Severity:** medium · **Confidence:** confirmed

```python
76:    classifiers: list[Classifier] = [SurfaceClassifier()]
77:    if "domain" in paths:
78:        classifiers.append(DomainClassifier(paths["domain"]))
...
81:    if "length" in paths:
82:        classifiers.append(LengthClassifier(paths["length"]))
83:    if "attributes" in paths:
84:        classifiers.append(AttributeClassifier(paths["attributes"]))
```

Each appended classifier's `.classify()` calls its own head's `.predict(prompt)` → `.predict_batch([prompt])` independently (`prompt_heads.py:93-99, 146-155, 176-182`), with no shared embedding cache across classifiers.

**What breaks:** `src/decompose/classifiers/prompt_decomposition/composite.py` already solves exactly this problem: `RouterHead._shared_embeddings` (`composite.py:201-220`) encodes a prompt once per *distinct encoder model* rather than once per head, specifically because "the length and attribute heads are linear probes; when they sit on the same encoder, running it twice doubles the only expensive part of the call for nothing" (its own docstring). `default_prompt_decomposer` reimplements the same head-bundling job as `RouterHead` (angle D: reuse) but as independent per-head `Classifier` adapters with no such cache, so it reintroduces the exact inefficiency `RouterHead` was built to avoid. This isn't hypothetical: `prompt_heads.py`'s own module-docstring wiring example (`:10-16`) pairs `LengthClassifier("artifacts/length/multi_turn__bge-small-en-v1.5__surface_embedding")` with `AttributeClassifier("artifacts/attributes/bge-small-en-v1.5__surface_embedding")` — the *same* `bge-small-en-v1.5` encoder — so following the documented example through `default_prompt_decomposer({"length": ..., "attributes": ...})` pays for two full encoder passes over the same prompt where one would do. `docs/prompt_decomposition.md`'s own "Known limits" section already flags this class of cost ("Latency ~130 ms ... two heads running independently. They do not share encoder passes; that is the first thing to change if serving cost matters more than the last point of accuracy") — this diff had the chance to not reintroduce it and did anyway.
**Evidence:** Traced `LengthClassifier.classify`/`AttributeClassifier.classify` → `LengthHead.predict`/`AttributeHead.predict` → `predict_batch([prompt])` with no `embeddings=` argument ever passed; compared against `RouterHead.predict_batch` which explicitly threads a shared `pool[encoder_model]` into both heads' `predict_batch`.
**Direction:** Either (a) have `default_prompt_decomposer` build on `RouterHead` (wrap its output as `Signal`s from one `CompositeClassifier`) instead of reassembling the individual heads, or (b) give `PromptDecomposer.extract_signals` an embedding cache keyed by encoder name that `_HeadClassifier` subclasses can consult, mirroring `_shared_embeddings`.

---

### DC-03: `default_prompt_decomposer` never passes `merge_domains=True`, so its domain output silently diverges from the documented production configuration `:78`
**Category:** design · **Severity:** medium · **Confidence:** plausible

```python
77:    if "domain" in paths:
78:        classifiers.append(DomainClassifier(paths["domain"]))
```

**What breaks:** `DomainClassifier.__init__(self, run_dir, **kwargs)` forwards `**kwargs` to `DomainHead(run_dir, **kwargs)`, so `merge_domains` *is* plumbable through the classifier — but `default_prompt_decomposer` never supplies it, so the factory always builds with `DomainHead`'s default `merge_domains=False`. Per `docs/prompt_decomposition.md` ("Post-hoc merging, never coarse training" section) and `prompt_heads.py`'s own wiring example (`DomainClassifier("artifacts/v7/PROD_ensemble", merge_domains=True)`, `:12` — note the `PROD_` prefix), the shipped/production configuration merges 10 classes to 8 (0.763 top-1 vs. 0.741 unmerged, per the doc's results table). A caller using `default_prompt_decomposer({"domain": "artifacts/v7/PROD_ensemble"})` — following the module's own example almost verbatim — silently gets the *weaker*, unmerged 10-class output instead, with no signal that a production-shaped call produced a non-production result.
**Evidence:** `DomainHead.__init__` default `merge_domains: bool = False` (`heads/domain.py:50`); `default_prompt_decomposer` passes no kwargs at all to `DomainClassifier`; doc and example both frame merging as what "production"/`PROD_` runs use.
**Direction:** Either default `default_prompt_decomposer` to `merge_domains=True` (documenting the override for callers who want raw 10-class output), or extend `artifact_paths` to accept a `(path, kwargs)` pair / add a separate `domain_kwargs` parameter so this isn't silently unreachable through the factory.

---

### DC-04: `default_prompt_decomposer` is not wired into `Router` or any other production entry point, despite the module docstring's framing `:1-16`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
4:  Wired into :class:`~router.router.Router` (real, working glue) but
...
10: Five real :class:`Classifier` implementations exist --
11: :mod:`decompose.classifiers.prompt_heads` -- and :func:`default_prompt_decomposer`
12: below builds a live decomposer from whichever of them have a trained artifact
13: on hand. ``PromptDecomposer()`` with no arguments still constructs an empty,
14: always-safe decomposer (no artifact paths to fail on) -- that remains the
15: default a caller gets without opting into the factory.
```

**What breaks:** Nothing at runtime, but the framing invites a reviewer/future maintainer to believe the five trained heads are one `default_prompt_decomposer(...)` call away from being live in `Router`. In fact `src/router/router.py:56` still does `self.prompt_decomposer = prompt_decomposer or PromptDecomposer()` — the always-empty constructor, unchanged by this diff — and a repo-wide grep shows `default_prompt_decomposer` is referenced only from its own module, `prompt_heads.py`'s docstring example, and `tests/decompose/test_decomposer_wiring.py`. No config, bootstrap script, or `Router` call site invokes it. The task's own framing ("this is where prompt_heads.py classifiers got wired into PromptDecomposer") describes the factory function existing, not the heads actually being wired into the live `Router` path.
**Evidence:** `grep -rn "default_prompt_decomposer"` across the repo (src, tests, configs) returns only `src/decompose/decomposer.py`, `src/decompose/classifiers/prompt_heads.py` (docstring), and `tests/decompose/test_decomposer_wiring.py`.
**Direction:** Either wire a call to `default_prompt_decomposer` into `Router`'s construction path (behind config for artifact paths), or soften the docstring so it doesn't read as though the five heads are live in production.

---

## src/decompose/classifiers/prompt_heads.py

### DC-05: `LengthClassifier.__init__` reads `length.json` twice — once inside `LengthHead.__init__`, once again for `variant` `:136-144`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
136:    def __init__(self, run_dir: str | Path) -> None:
137:        super().__init__(run_dir)
138:        import json
139:
140:        from decompose.classifiers.prompt_decomposition import LengthHead
141:
142:        self.head = LengthHead(run_dir)
143:        meta = json.loads((self.run_dir / "length.json").read_text(encoding="utf-8"))
144:        self.variant = meta.get("variant", "single_turn")
```

**What breaks:** `LengthHead.__init__` (`heads/length.py:51-57`) already does `meta = json.loads((self.run_dir / "length.json").read_text(...))` to populate `feature_set`/`encoder_model`, then discards `meta`. `LengthClassifier.__init__` re-opens and re-parses the identical file immediately afterward purely to read `variant`, which `LengthHead` doesn't expose. It's construction-time-only (not per-request) so the cost is negligible, but it's two filesystem reads and two JSON parses of the same file for no reason beyond `LengthHead` not surfacing a field it already loaded.
**Evidence:** Compared `heads/length.py:51-57` against `prompt_heads.py:142-144` — both read `self.run_dir / "length.json"` independently.
**Direction:** Either have `LengthHead` store `self.variant` (or the raw `meta` dict) on itself so `LengthClassifier` can read `self.head.variant`, or have `LengthClassifier` read the file once and pass the parsed dict/variant into `LengthHead` if it's refactored to accept it.
