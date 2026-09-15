# Code review: decompose

**Files reviewed:** 10 (276 lines): `__init__.py`, `conversation.py`, `decomposer.py`, `signals.py`, `classifiers/__init__.py`, `classifiers/base.py`, `embedders/__init__.py`, `embedders/base.py`, `embedders/bert.py`, `embedders/sentence_transformer.py` · **Context-only:** `src/router/embeddings/encoder.py`
**Summary:** 3 findings: 0 correctness, 2 design, 1 performance · 3 confirmed, 0 suspected

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| DC-01 | `src/decompose/signals.py:26-27` | design | low | confirmed |
| DC-02 | `src/decompose/embedders/bert.py:33-35` | design | low | confirmed |
| DC-03 | `src/decompose/embedders/bert.py:37-38` | performance | low | confirmed |

---

## src/decompose/signals.py

### DC-01: Same-named signals from different classifiers silently overwrite each other `:26-27`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
26:    def add(self, signal: Signal) -> None:
27:        self.signals[signal.name] = signal
```

**What breaks:** `PromptDecomposer.extract_signals` (`decomposer.py:39-41`) passes every classifier's output through `add`. If two classifiers emit a signal with the same `name` (two `"language"` detectors, or a `CompositeClassifier` whose children overlap), the last one wins. Its `confidence` and `produced_by` replace the earlier signal without a trace. This is latent, because no concrete `Classifier` exists yet.
**Evidence:** Read `decomposer.py:37-44` and `classifiers/base.py:51-55`. `CompositeClassifier.classify` concatenates child signals without de-duplicating, so collisions reach `add`.
**Direction:** Key by `(produced_by, name)`, keep a list per name, or raise on a collision.

---

## src/decompose/embedders/bert.py (same pattern in sentence_transformer.py)

### DC-02: Reading `.dimension` loads the whole encoder model `:33-35`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
33:    @property
34:    def dimension(self) -> int:
35:        return self._encoder.dim
```

**What breaks:** `Embedder.dimension` is a plain class attribute (`embedders/base.py:20`), which reads like cheap metadata. Both concrete embedders override it with a property that calls `TextEncoder.dim`. That property touches `self.model` (`router/embeddings/encoder.py:106-112`), which downloads BERT or MiniLM and loads it onto the device. A caller that checks dimensions while wiring a pipeline pays for a full model load. The same property is at `sentence_transformer.py:27-29`.
**Evidence:** Traced `dimension` → `TextEncoder.dim` → `TextEncoder.model` → `_load_bert` / `_load_sentence_transformer`.
**Direction:** Read the hidden size from `transformers.AutoConfig` (cheap), or document the property as expensive.

### DC-03: Each request encodes a batch of one and converts it to a Python list `:37-38`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
37:    def embed(self, text: str) -> list[float]:
38:        return self._encoder.encode([text])[0].tolist()
```

**What breaks:** Every request runs a one-element batch through `TextEncoder.iter_encode`. On each call that path re-seeds every global RNG and turns on deterministic-algorithms mode (RT-06 in `router.md`). The 384- or 768-float vector then becomes a Python `list` stored in `PromptSignals.embeddings`, and any numpy or torch consumer has to convert it back. The same code is at `sentence_transformer.py:31-32`.
**Evidence:** `encoder.py:128` calls `_seed_everything` inside `iter_encode`, which `encode` calls.
**Direction:** Keep an `np.ndarray` in `PromptSignals`, and add `embed_batch` for callers with several prompts.

---

## src/decompose/decomposer.py, conversation.py, classifiers/base.py, `__init__` files

No findings.

---

## Checked and ruled out
- **`decompose.embedders.bert` / `sentence_transformer` importing `router.embeddings.encoder`:** these are sanctioned import-linter exemptions. `lint-imports` reports "Decompose stays dependency-free KEPT (2 ignored imports)".
- **`SimpleClassifier` adds nothing over `Classifier`:** documented scaffolding, not dead code.
- **`CompositeClassifier.__init__` not calling `super().__init__()`:** `Classifier` defines no `__init__`.
- **`ClassificationInput.metadata` mutable default:** it uses `field(default_factory=dict)`.
