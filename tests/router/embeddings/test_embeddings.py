from __future__ import annotations

import numpy as np
import pytest

import json

from router.config import load_config
from router.embeddings import EmbeddingStore, available_pathways, build_store, load_encoder
from router.embeddings.encoder import EncoderConfig, TextEncoder


@pytest.fixture(scope="module")
def encoder():
    enc = load_encoder(load_config(), pathway="retrieval")
    try:
        _ = enc.dim
    except Exception as exc:  # model not cached / offline
        pytest.skip(f"encoder unavailable: {exc}")
    return enc


@pytest.fixture(scope="module")
def bert_encoder():
    enc = load_encoder(load_config(), pathway="irt")
    if enc.cfg.backend != "bert":
        pytest.skip("irt pathway is not a bert backend")
    try:
        _ = enc.dim
    except Exception as exc:
        pytest.skip(f"bert encoder unavailable: {exc}")
    return enc


def test_pathways_configured():
    pw = available_pathways(load_config())
    assert "retrieval" in pw and "irt" in pw


def test_bert_backend_pools_and_shapes(bert_encoder):
    emb = bert_encoder.encode(["a short query", "another one about code"])
    assert emb.shape == (2, bert_encoder.dim)
    assert bert_encoder.dim == 768
    assert bert_encoder.cfg.pooling in ("mean", "cls")


def test_bert_backend_deterministic(bert_encoder):
    a = bert_encoder.encode(["reproducible?", "yes"])
    b = bert_encoder.encode(["reproducible?", "yes"])
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_bert_store_uses_model_id_field(tmp_path, bert_encoder):
    store = EmbeddingStore.build(["gpt-4-1106-preview", "llama-2-70b-chat"],
                                 ["strong model", "open model"], bert_encoder,
                                 id_field="model_id", show_progress=False)
    store.save(tmp_path)
    loaded = EmbeddingStore.load(tmp_path)
    assert loaded.id_field == "model_id"
    assert "gpt-4-1106-preview" in loaded
    assert loaded.get("gpt-4-1106-preview").shape == (bert_encoder.dim,)


TEXTS = ["what is the capital of france?", "solve for x: 2x + 3 = 9",
         "write a python function to reverse a list", "who wrote hamlet?"]


def test_output_dimensions(encoder):
    emb = encoder.encode(TEXTS)
    assert emb.shape == (len(TEXTS), encoder.dim)
    assert emb.dtype == np.float32


def test_deterministic_inference(encoder):
    a = encoder.encode(TEXTS)
    b = encoder.encode(TEXTS)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_normalization(encoder):
    emb = encoder.encode(TEXTS)
    norms = np.linalg.norm(emb, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-4)


def test_order_preserved(encoder):
    single = np.vstack([encoder.encode([t]) for t in TEXTS])
    batched = encoder.encode(TEXTS)
    np.testing.assert_allclose(single, batched, atol=1e-4)


def test_empty_input(encoder):
    assert encoder.encode([]).shape == (0, encoder.dim)


def test_store_roundtrip(tmp_path, encoder):
    ids = [f"q{i}" for i in range(len(TEXTS))]
    store = EmbeddingStore.build(ids, TEXTS, encoder, show_progress=False)
    store.save(tmp_path)
    loaded = EmbeddingStore.load(tmp_path)
    assert loaded.ids == ids
    np.testing.assert_allclose(loaded.matrix, store.matrix)
    np.testing.assert_allclose(loaded.get("q2"), store.get("q2"))
    assert loaded.manifest["dim"] == encoder.dim


def test_store_subset(tmp_path, encoder):
    ids = [f"q{i}" for i in range(len(TEXTS))]
    store = EmbeddingStore.build(ids, TEXTS, encoder, show_progress=False)
    sub = store.subset(["q1", "q3", "missing"])
    assert sub.ids == ["q1", "q3"]
    assert sub.matrix.shape == (2, encoder.dim)


def test_config_fingerprint_changes_with_model():
    a = EncoderConfig(model_name="m1").fingerprint()
    b = EncoderConfig(model_name="m2").fingerprint()
    assert a != b


# --------------------------------------------------------------------------- #
# streaming / resumable build                                                 #
# --------------------------------------------------------------------------- #
def test_build_store_writes_new_layout(tmp_path, encoder):
    ids = [f"q{i}" for i in range(20)]
    store = build_store(tmp_path, ids, [f"text {i}" for i in ids], encoder,
                        id_field="query_id", flush_every=2, show_progress=False)
    assert (tmp_path / "vectors.npy").exists()
    assert (tmp_path / "ids.parquet").exists()
    assert (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "_progress.json").exists()      # cleaned on completion
    assert store.manifest["complete"] is True
    assert len(store) == 20


def test_build_store_resumes_from_checkpoint(tmp_path, encoder):
    ids = [f"q{i}" for i in range(30)]
    texts = [f"query about topic {i}" for i in range(30)]
    full = build_store(tmp_path, ids, texts, encoder, id_field="query_id",
                       show_progress=False)
    ref = np.asarray(full.matrix).copy()
    full.close()

    # simulate a crash after 12 rows: drop manifest, write partial progress,
    # corrupt the tail of vectors.npy
    (tmp_path / "manifest.json").unlink()
    mm = np.lib.format.open_memmap(tmp_path / "vectors.npy", mode="r+")
    mm[12:] = -999.0
    mm.flush()
    dim = mm.shape[1]
    del mm
    (tmp_path / "_progress.json").write_text(json.dumps({
        "rows_done": 12, "count": 30, "dim": dim,
        "batch_size": encoder.cfg.batch_size,
        "ids_fingerprint": full.manifest["ids_fingerprint"],
        "config_fingerprint": encoder.cfg.fingerprint(),
    }))

    resumed = build_store(tmp_path, ids, texts, encoder, id_field="query_id",
                          show_progress=False)
    np.testing.assert_allclose(np.asarray(resumed.matrix), ref, atol=1e-4)
    assert resumed.manifest["complete"] is True
    resumed.close()


def test_build_store_skips_when_already_complete(tmp_path, encoder):
    ids = [f"q{i}" for i in range(10)]
    texts = [f"t{i}" for i in ids]
    build_store(tmp_path, ids, texts, encoder, id_field="query_id", show_progress=False)
    mtime = (tmp_path / "vectors.npy").stat().st_mtime_ns
    build_store(tmp_path, ids, texts, encoder, id_field="query_id", show_progress=False)
    assert (tmp_path / "vectors.npy").stat().st_mtime_ns == mtime   # not rewritten


def test_store_load_is_memmapped(tmp_path, encoder):
    ids = [f"q{i}" for i in range(8)]
    build_store(tmp_path, ids, [f"t{i}" for i in ids], encoder,
                id_field="query_id", show_progress=False)
    store = EmbeddingStore.load(tmp_path, mmap=True)
    assert isinstance(store.matrix, np.memmap)
    store.close()
