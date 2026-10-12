# MOD-001: `build_store` will not resume a partial build after a device / batch-size change

- **Status:** reproduced (`known_bugs/_repros/repro_build_store_resume.py`)
- **Severity:** low-med
- **Where:** `build_store` ([encoder.py:497-507](../../../src/router/embeddings/encoder.py#L497-L507)); contract stated at `EncoderConfig.vector_fingerprint` ([encoder.py:76-78](../../../src/router/embeddings/encoder.py#L76-L78))
- **Found:** 2026-09-24

## What

`vector_fingerprint()` is documented as the fingerprint that "decides whether a
finished store is still current **and whether a partial build can be
resumed**, so a device or batch-size change doesn't force a re-encode". The
*finished-store* check does use it ([encoder.py:490-492](../../../src/router/embeddings/encoder.py#L490-L492)),
but the *resume* check compares `prog.get("config_fingerprint") == cfg_fp`
([encoder.py:500](../../../src/router/embeddings/encoder.py#L500)). `cfg_fp` is
`EncoderConfig.fingerprint()`, which hashes every field except `name`,
including `device` and `batch_size`. `_write_progress` only records
`config_fingerprint`, not `vector_fingerprint`.

## Failure scenario

Encode a large query store on CPU, get interrupted, then re-run on the GPU
(`device: auto` now resolves to `cuda`; this dev box recently gained a CUDA
torch build) or with a different `batch_size`. The vectors would be identical,
but the progress file's `config_fingerprint` differs, so `build_store` prints
`fingerprint mismatch, restarting` and re-encodes from row 0.

Repro output (40 rows, batch 4, killed after 5 batches on `cpu`, resumed on `cuda`):

```
[build_store] fingerprint mismatch, restarting
batches encoded in run 2: 10 (10 total; 5 if it had resumed)
```

## Suggested fix

Record `vector_fingerprint` in `_write_progress` and compare that in the resume
gate (keep `config_fingerprint` as provenance only). Old progress files that
lack the key can fall back to the current comparison.
