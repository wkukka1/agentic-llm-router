# Code review: router.embeddings

**Files reviewed:** 2 (527 lines): `__init__.py`, `encoder.py` · **Context-only:** `src/router/determinism.py`, `src/training/data/embeddings.py`, `src/training/models/profiles.py`, `configs/phase0.yaml`, `configs/irt_router.yaml`
**Summary:** 7 findings: 6 correctness, 1 design, 0 performance · 7 confirmed, 0 suspected

Claims marked "probe" were checked by running `build_store` with a stub encoder (scratchpad `probe_store.py`, `.venv/Scripts/python.exe`).

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| RE-01 | `src/router/embeddings/encoder.py:439-443` | correctness | medium | confirmed |
| RE-02 | `src/router/embeddings/encoder.py:457-492` | correctness | medium | confirmed |
| RE-03 | `src/router/embeddings/encoder.py:63-67` | correctness | low | confirmed |
| RE-04 | `src/router/embeddings/encoder.py:197-202` | correctness | low | confirmed |
| RE-05 | `src/router/embeddings/encoder.py:212` | correctness | low | confirmed |
| RE-06 | `src/router/embeddings/encoder.py:154-156` | correctness | low | confirmed |
| RE-07 | `src/router/embeddings/encoder.py:289-298` | design | low | confirmed |

---

## src/router/embeddings/encoder.py

### RE-01: `build_store` returns a stale store after the encoder config or the texts change `:439-443`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
440:    if not force and EmbeddingStore.exists(d):
441:        man = json.loads(EmbeddingStore._resolve(d, EmbeddingStore.MANIFEST).read_text("utf-8"))
442:        if man.get("ids_fingerprint") in (id_fp, None) and man.get("count") == n:
443:            return EmbeddingStore.load(d)
```

**What breaks:** The "already finished" shortcut compares only the id list. It ignores `cfg_fp`, which is computed two lines earlier (`:433`), and nothing fingerprints the texts. Three situations return the old vectors while the build appears to succeed:
- **Encoder change:** the directory name `<kind>__<pathway>` stays the same after a pathway's `model_name`, `pooling`, `normalize` or `max_seq_length` changes.
- **Text change:** re-rendered profile texts (see TM-01) or re-normalised query text are never re-encoded.
- **Legacy manifest:** a manifest without `ids_fingerprint` matches any id list of the same length (`in (id_fp, None)`).

**Evidence:** Probe: built with `model-A` (vectors 1.0), then called again with `model-B` (vectors 2.0). The call returned `[1.0, 1.0]`, and the manifest still said `model-A`. `tests/router/embeddings/test_embeddings.py` covers resume but not a config change.
**Direction:** Require `man["config_fingerprint"] == cfg_fp` and a matching texts fingerprint before returning, and treat a missing fingerprint as stale.

### RE-02: An interrupted rebuild leaves a loadable store with new ids, zero vectors and the old manifest `:457-492`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
458:    mm = np.lib.format.open_memmap(vpath, mode=mode, dtype=np.float32, shape=(n, dim))
461:    pd.DataFrame({id_field: ids, "row": range(n)}).to_parquet(d / EmbeddingStore.IDS, index=False)
492:    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
```

**What breaks:** Rebuilding into a directory that already holds a complete store (different ids, or `force=True`) overwrites `vectors.npy` (`w+`) and `ids.parquet` at once. The old `manifest.json` still says `"complete": true` until the final write. If the run dies in between (OOM, Ctrl-C), `EmbeddingStore.exists` returns `True` and `load` succeeds: new ids, a matrix of zeros past `rows_done`, and a manifest describing the previous build. `TrainingData.query_embeddings` then serves that store silently. Nothing in the repo writes `"complete": false`, so the guard at `:327-330` can never fire.
**Evidence:** Probe: a complete store for `[a, b]`, then a rebuild for `[c, d, e, f]` killed after one row. Result: `exists=True`, `ids=['c','d','e','f']`, `vectors=[5.0, 0.0, 0.0, 0.0]`, `manifest count=2`. A grep for `"complete"` across `src scripts` finds only writers of `True`.
**Direction:** Before opening the memmap in `w+`, write a `"complete": false` manifest or delete the old one.

### RE-03: Resume is rejected after a device or batch-size change `:63-67`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
63:    def fingerprint(self) -> str:
64:        payload = json.dumps(
65:            {k: v for k, v in self.__dict__.items() if k != "name"}, sort_keys=True
```

**What breaks:** The fingerprint includes `device`, `batch_size` and `seed`. A long CPU build that is killed and resumed with `device: cuda` (the dev machine has an RTX 3050 Ti) or a different batch size prints "fingerprint mismatch, restarting" (`:454-455`) and re-encodes from row 0.
**Evidence:** Read `:446-455`. `device` defaults to the literal `"auto"`, so only an explicit device change triggers this.
**Direction:** Fingerprint only the fields that change vectors (`backend`, `model_name`, `normalize`, `max_seq_length`, bert `pooling`). Record the others in the manifest for provenance.

### RE-04: Per-pathway `batch_size` and `device` are silently ignored `:197-202`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
198:    shared = {
199:        "batch_size": int(e.get("batch_size", 64)),
200:        "device": e.get("device", "auto"),
```

**What breaks:** `load_encoder` reads these keys only from the top-level `embedding:` block. A `pathways.irt.batch_size: 8`, which is natural for BERT on a 4 GB GPU next to a MiniLM pathway, is ignored and 64 is used. That can OOM. No config sets per-pathway values today, so this is latent.
**Evidence:** `configs/phase0.yaml:77-89` and `configs/irt_router.yaml:88-99` set them top-level only.
**Direction:** Use `p.get("batch_size", shared["batch_size"])`, and the same for `device`.

### RE-05: A `backend: bert` pathway without `model_name` loads MiniLM, not `bert-base-uncased` `:212`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
211:            backend=p.get("backend", "sentence_transformer"),
212:            model_name=p.get("model_name", EncoderConfig.model_name),
```

**What breaks:** The fallback is the dataclass default, `sentence-transformers/all-MiniLM-L6-v2`, loaded through `AutoModel` with manual pooling. The module docstring (`:7`) says the bert backend defaults to `bert-base-uncased`. A bert pathway that omits `model_name` gets a 384-d model, and its stores won't line up with a paper-faithful 768-d run. The same fallback is at `:222`.
**Evidence:** Read `:48-58` and `:209-227`.
**Direction:** Choose the default per backend.

### RE-06: `encode` crashes on numpy or pandas input `:154-156`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
154:    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
155:        if not texts:
```

**What breaks:** `not texts` on a `pd.Series` or `np.ndarray` raises `ValueError: The truth value ... is ambiguous`, although `iter_encode` accepts both types. Every in-repo caller passes a `list` (`router/routing/routers.py:118-131`, `scripts/data/encode_irt_router_pathway.py:50,67`), so this is latent.
**Evidence:** Grep of the `.encode(` callers.
**Direction:** `if len(texts) == 0:`.

### RE-07: The two store constructors disagree on id types `:289-298`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
289:    def build(cls, ids: Sequence[str], texts: Sequence[str], encoder: TextEncoder,
298:        return cls(ids, matrix, cls._manifest(encoder, ids, id_field, dim), id_field)
```

**What breaks:** `EmbeddingStore.build` keeps ids as passed, while `build_store` stringifies them (`:431`). The same integer id list therefore works for lookups with one constructor and raises `KeyError` with the other. RD-02 shows how this becomes a silently empty dataset.
**Evidence:** Read both constructors.
**Direction:** Normalise ids to `str` in `EmbeddingStore.__init__`.

---

## src/router/embeddings/__init__.py

No findings.

---

## Checked and ruled out
- **Global RNG reseeding and deterministic-algorithms mode on every encode (`:99`, `:128`):** already filed as RT-06 in `router.md`.
- **Mean-pooling divide-by-zero on an all-pad row:** `clamp(min=1e-9)` at `:175` prevents it.
- **`open_memmap(mode="r+", shape=...)` on resume:** `r+` reads shape and dtype from the header, and the dim is checked in the progress file (`:450`).
- **`None` turned into `""` in `iter_encode`:** callers pass parquet string columns.
- **Repeated `EmbeddingStore.load` on every routing call:** filed as RR-12 in `router.routing.md`.
