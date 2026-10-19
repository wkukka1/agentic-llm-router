"""A killed build resumes when only execution settings (batch size, device) changed."""

from __future__ import annotations

import json

import numpy as np

from router.embeddings import build_store
from router.embeddings.encoder import EncoderConfig, TextEncoder

DIM = 4


class _Fake(TextEncoder):
    """Row i of the matrix is i, so which rows were re-encoded is visible."""

    def __init__(self, **cfg):
        super().__init__(EncoderConfig(name="t", **cfg))

    @property
    def dim(self):
        return DIM

    def iter_encode(self, texts, batch_size=None, show_progress=False, _start_offset=0):
        bs = int(batch_size or self.cfg.batch_size)
        for start in range(0, len(texts), bs):
            rows = np.arange(_start_offset + start, _start_offset + min(start + bs, len(texts)), dtype=np.float32)
            yield _start_offset + start, np.repeat(rows[:, None], DIM, axis=1)


def _interrupt(tmp_path, first: _Fake, n: int, done: int, ids, texts):
    """Leave what a killed build leaves: sentinel rows done, progress written by `first`."""
    store = build_store(tmp_path, ids, texts, first, show_progress=False)
    store.close()
    (tmp_path / "manifest.json").unlink()
    mm = np.lib.format.open_memmap(tmp_path / "vectors.npy", mode="r+")
    mm[:done] = -1.0  # rows a resume must keep, not re-encode
    mm.flush()
    del mm
    (tmp_path / "_progress.json").write_text(json.dumps({
        "rows_done": done, "count": n, "dim": DIM, "batch_size": first.cfg.batch_size,
        "ids_fingerprint": store.manifest["ids_fingerprint"],
        "texts_fingerprint": store.manifest["texts_fingerprint"],
        "config_fingerprint": first.cfg.fingerprint(),
        "vector_fingerprint": first.cfg.vector_fingerprint(),
    }))


def test_a_changed_batch_size_or_device_still_resumes(tmp_path):
    ids, texts = [f"q{i}" for i in range(30)], [f"t{i}" for i in range(30)]
    _interrupt(tmp_path, _Fake(batch_size=4, device="cpu"), 30, 12, ids, texts)
    resumed = build_store(tmp_path, ids, texts, _Fake(batch_size=8, device="cuda"), show_progress=False)
    m = np.asarray(resumed.matrix)
    assert (m[:12] == -1.0).all()                       # kept: resumed, not rebuilt
    assert (m[12:, 0] == np.arange(12, 30)).all()       # the rest encoded


def test_a_changed_vector_setting_restarts(tmp_path):
    ids, texts = [f"q{i}" for i in range(30)], [f"t{i}" for i in range(30)]
    _interrupt(tmp_path, _Fake(batch_size=4), 30, 12, ids, texts)
    rebuilt = build_store(tmp_path, ids, texts, _Fake(batch_size=4, normalize=False), show_progress=False)
    assert (np.asarray(rebuilt.matrix)[:, 0] == np.arange(30)).all()
