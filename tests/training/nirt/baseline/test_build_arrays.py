"""``training.nirt.baseline.data.build_arrays`` -- relevance-dim staleness guard.

Regression for a crash where a checkpoint trained against an earlier taxonomy
(fewer clusters) was fed ``r_q`` from a since re-clustered, wider relevance
store: the mismatch surfaced as a bare matmul shape error deep inside
``DiscriminationHead.forward`` instead of a diagnosable message.
"""

from __future__ import annotations

import numpy as np
import pytest

from router.config import Config
from training.nirt.baseline.data import build_arrays


class _FakeGathered:
    def __init__(self, n=8, q_dim=6):
        self.query_ids = [f"q{i}" for i in range(n)]
        self.model_ids = [f"m{i % 2}" for i in range(n)]
        self.dropped = 0
        self._q_dim = q_dim

    def gather(self):
        rng = np.random.default_rng(0)
        n = len(self.query_ids)
        return {
            "query_embedding": rng.standard_normal((n, self._q_dim)).astype(np.float32),
            "model_embedding": rng.standard_normal((n, 4)).astype(np.float32),
            "target": rng.integers(0, 2, n).astype(np.float64),
            "cost": np.zeros(n, dtype=np.float32),
        }


class _FakeData:
    def __init__(self, **kw):
        self._ds = _FakeGathered(**kw)

    def nirt_dataset(self, split, pathway):
        return self._ds


class _FakeRelevanceStore:
    """A relevance store with a fixed ``dim`` and no rows (every query misses)."""

    def __init__(self, dim):
        self.dim = dim

    def __contains__(self, query_id):
        return False

    def row_of(self, query_id):
        return -1

    @property
    def matrix(self):
        return np.zeros((0, self.dim), dtype=np.float32)


def test_build_arrays_rejects_stale_relevance_dim(monkeypatch):
    """A re-clustered taxonomy (wider r_q) must not silently feed a checkpoint's
    fixed-width ``w_r`` layer -- it should fail with a clear message."""
    monkeypatch.setattr("training.taxonomy.load_relevance",
                        lambda cfg, pathway=None: _FakeRelevanceStore(154))

    with pytest.raises(ValueError, match=r"relevance store at dim 154.*relevance_dim \(39\)"):
        build_arrays(Config({}), split="test", data=_FakeData(), use_relevance=True,
                     use_warmup=False, relevance_dim=39)


def test_build_arrays_accepts_matching_relevance_dim(monkeypatch):
    monkeypatch.setattr("training.taxonomy.load_relevance",
                        lambda cfg, pathway=None: _FakeRelevanceStore(39))

    arr = build_arrays(Config({}), split="test", data=_FakeData(), use_relevance=True,
                       use_warmup=False, relevance_dim=39)
    assert arr.relevance_dim == 39


def test_build_arrays_skips_check_without_expected_dim(monkeypatch):
    """``relevance_dim=None`` (the training path, before a checkpoint exists) keeps
    trusting whatever the live taxonomy produces."""
    monkeypatch.setattr("training.taxonomy.load_relevance",
                        lambda cfg, pathway=None: _FakeRelevanceStore(154))

    arr = build_arrays(Config({}), split="test", data=_FakeData(), use_relevance=True,
                       use_warmup=False)
    assert arr.relevance_dim == 154
