# TEXTENCODER-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-10-02, uncommitted
- **Diff base:** `95c4f04` to the working tree on 2026-10-02

## What changed

([encoder.py](../../../src/router/embeddings/encoder.py); regression test `test_encode_leaves_caller_rngs_untouched` in `tests/router/embeddings/test_embeddings.py`)

`iter_encode` no longer reseeds anything. Inference is eval mode under `no_grad` with no dropout, so
the vectors never depended on RNG state; the per-call reseed only pinned every caller's `random` /
numpy / torch streams to the seed-42 state on the serving path (and cost about 0.5 ms per request,
mostly `torch.cuda.manual_seed_all`).

The one-time seed at model load is kept for a reproducible load, but now runs inside a new
`_scoped_seed(seed)` context manager: it saves the `random` and numpy states, forks the torch CPU and
CUDA generators with `torch.random.fork_rng`, seeds, loads the model, and restores everything on
exit. Loading the encoder lazily on the first routed request therefore leaves no trace on the
application's generators either.

Checked that the vectors are unchanged: 16 live `bert-base-uncased` encodes against the stored `irt`
query store differ by at most 2.1e-6 (GPU float noise). The repro now prints
`draws after the first encode identical -> False`.

## Fix suggested when the bug was filed

Drop the reseed from `iter_encode` (keep the one-time seed around model load, ideally scoped:
save/restore the RNG states with `random.getstate()` / `np.random.get_state()` /
`torch.random.fork_rng()`).

## Regression tests

```python
def test_encode_leaves_caller_rngs_untouched():
    """TEXTENCODER-001: encoding (serving path) must not reseed the
    process-global ``random`` / numpy / torch generators -- neither per call
    nor via the one-time lazy model load."""
    enc = _StubBertEncoder(EncoderConfig(backend="bert"))
    random.seed(1234), np.random.seed(1234), torch.manual_seed(1234)
    expected = [(random.random(), float(np.random.rand()), float(torch.rand(1))) for _ in range(3)]

    random.seed(1234), np.random.seed(1234), torch.manual_seed(1234)
    draws = []
    for _ in range(3):
        draws.append((random.random(), float(np.random.rand()), float(torch.rand(1))))
        enc.encode(["a request"])
    assert draws == expected
```

## Diff

```diff
diff --git a/src/router/embeddings/encoder.py b/src/router/embeddings/encoder.py
index cecbcf9..6f03c1a 100644
--- a/src/router/embeddings/encoder.py
+++ b/src/router/embeddings/encoder.py
@@ -11,5 +11,5 @@ Two backends, selected per *pathway* in ``configs/phase0.yaml -> embedding``:
   can be placed from its profile alone).
 
-Inference is deterministic (eval mode, ``torch.no_grad``, fixed seeds, no
+Inference is deterministic (eval mode, ``torch.no_grad``, seeded model load, no
 dropout). Phase 0 does **not** fine-tune anything here.
 
@@ -20,4 +20,5 @@ a float32 ``.npy`` matrix + a parquet id index + a JSON manifest.
 from __future__ import annotations
 
+import contextlib
 import hashlib
 import json
@@ -88,4 +89,30 @@ class EncoderConfig:
 
 
+@contextlib.contextmanager
+def _scoped_seed(seed: int):
+    """Seed ``random`` / numpy / torch (CPU + CUDA) for the body, then restore
+    the caller's generator states -- so a one-time seeded model load leaves no
+    trace on the application's RNG streams."""
+    import random
+
+    py_state, np_state = random.getstate(), np.random.get_state()
+    try:
+        import torch
+    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
+        torch = None
+    try:
+        if torch is None:
+            _seed_everything(seed, deterministic_algorithms=False)
+            yield
+        else:
+            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
+            with torch.random.fork_rng(devices=devices):
+                _seed_everything(seed, deterministic_algorithms=False)
+                yield
+    finally:
+        random.setstate(py_state)
+        np.random.set_state(np_state)
+
+
 class TextEncoder:
     def __init__(self, cfg: EncoderConfig):
@@ -117,11 +144,13 @@ class TextEncoder:
     def model(self):
         if self._model is None:
-            # no deterministic-algorithms switch: that is process-wide and this
-            # also runs on the serving path next to training / GPU jobs
-            _seed_everything(self.cfg.seed, deterministic_algorithms=False)
-            self._model = (
-                self._load_bert() if self.cfg.backend == "bert"
-                else self._load_sentence_transformer()
-            )
+            # seeded for a reproducible load, but inside a saved/restored RNG
+            # scope: this also runs on the serving path, and the process-global
+            # generators belong to the application (TEXTENCODER-001). No
+            # deterministic-algorithms switch either -- that is process-wide.
+            with _scoped_seed(self.cfg.seed):
+                self._model = (
+                    self._load_bert() if self.cfg.backend == "bert"
+                    else self._load_sentence_transformer()
+                )
         return self._model
 
@@ -147,6 +176,8 @@ class TextEncoder:
         is the absolute row index (``_start_offset`` + local offset), so callers
         can resume a partial build.
+        No reseed here: inference is eval-mode + ``no_grad`` with no dropout, so
+        the vectors don't depend on RNG state, and reseeding the process-global
+        generators per call pinned every caller's random stream (TEXTENCODER-001).
         """
-        _seed_everything(self.cfg.seed, deterministic_algorithms=False)
         bs = int(batch_size or self.cfg.batch_size)
         texts = [("" if t is None else str(t)) for t in texts]
```
