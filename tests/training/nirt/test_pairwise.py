from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import make_store as _store

torch = pytest.importorskip("torch")

from router.nirt.model import NIRTModel
from training.nirt.pairwise import build_pairwise_arrays, pairwise_logit_diff, pairwise_loss


class _StubPairwiseData:
    """Just enough of ``TrainingData`` for ``build_pairwise_arrays``."""

    def __init__(self, battles: pd.DataFrame, q_store, m_store):
        self._battles = battles
        self._q = q_store
        self._m = m_store

    def pairwise(self, source=None, split=None, models="all"):
        df = self._battles
        return df[df["source"] == source].copy() if source else df.copy()

    def query_embeddings(self, pathway=None):
        return self._q

    def profile_embeddings(self, pathway=None):
        return self._m


@pytest.fixture
def battles():
    return pd.DataFrame({
        "query_id": ["q0", "q1", "q2", "q3"],
        "model_a": ["ma", "mb", "ma", "missing-model"],
        "model_b": ["mb", "mc", "mc", "mb"],
        "score_a": [1.0, 0.0, 0.5, 1.0],
        "source": ["chatbot_arena"] * 4,
    })


@pytest.fixture
def stores():
    rng = np.random.default_rng(0)
    q_store = _store(["q0", "q1", "q2"], rng.standard_normal((3, 10)), "query_id")   # q3 missing
    m_store = _store(["ma", "mb", "mc"], rng.standard_normal((3, 6)), "model_id")    # missing-model absent
    return q_store, m_store


def test_build_pairwise_arrays_joins_and_drops(battles, stores):
    q_store, m_store = stores
    data = _StubPairwiseData(battles, q_store, m_store)
    arr = build_pairwise_arrays(data, source="chatbot_arena", pathway="irt")
    assert len(arr) == 3                     # the q3/missing-model row is dropped
    assert arr.n_dropped == 1
    assert arr.e_q.shape == (3, 10)
    assert arr.e_a.shape == (3, 6) and arr.e_b.shape == (3, 6)
    np.testing.assert_array_equal(arr.e_q[0], q_store.get("q0"))
    np.testing.assert_array_equal(arr.e_a[0], m_store.get("ma"))
    assert arr.y.tolist() == [1.0, 0.0, 0.5]


def test_build_pairwise_arrays_missing_store_raises(battles):
    data = _StubPairwiseData(battles, None, None)
    with pytest.raises(FileNotFoundError):
        build_pairwise_arrays(data, pathway="irt")


def test_build_pairwise_arrays_empty_when_nothing_joins(stores):
    q_store, m_store = stores
    empty = pd.DataFrame({"query_id": ["nope"], "model_a": ["ma"], "model_b": ["mb"],
                          "score_a": [1.0], "source": ["chatbot_arena"]})
    arr = build_pairwise_arrays(_StubPairwiseData(empty, q_store, m_store), pathway="irt")
    assert len(arr) == 0 and arr.n_dropped == 1


def test_pairwise_logit_diff_matches_manual_bilinear():
    torch.manual_seed(0)
    m = NIRTModel(query_dim=8, dim=3, model_params="projected", profile_dim=5)
    e_q = torch.randn(4, 8)
    e_a = torch.randn(4, 5)
    e_b = torch.randn(4, 5)
    diff = pairwise_logit_diff(m, e_q, e_a, e_b)
    z_a = m(e_q, e_a)
    z_b = m(e_q, e_b)
    assert torch.allclose(diff, z_a - z_b, atol=1e-5)


def test_pairwise_logit_diff_uses_shared_centering_reference():
    """XA-04: a_a/a_b come from two separate forward calls, so a
    center_discrimination model must centre both on ONE shared reference --
    not each on its own batch, which would bias z_a - z_b by
    -(r_a - r_b) . theta. The fixed diff must match a manual computation
    that shares one reference, and differ from the old batch-separate one."""
    torch.manual_seed(0)
    m = NIRTModel(query_dim=8, dim=3, model_params="projected", profile_dim=5,
                  center_discrimination=True)
    e_q = torch.randn(4, 8)
    e_a = torch.randn(4, 5)
    e_b = torch.randn(4, 5)

    fixed_diff = pairwise_logit_diff(m, e_q, e_a, e_b)
    assert m._a_ref is None   # the context manager restores the previous (unset) reference

    theta = m.latent_query(e_q)

    # old, buggy behaviour: a_a and a_b each centred on their OWN batch mean
    m.set_discrimination_reference(None)
    a_a_bad, b_a_bad = m.model_parameters(e_a)
    a_b_bad, b_b_bad = m.model_parameters(e_b)
    bad_diff = ((a_a_bad * theta).sum(-1) - b_a_bad) - ((a_b_bad * theta).sum(-1) - b_b_bad)

    # fixed behaviour: both centred on the mean discrimination over e_a and e_b combined
    m.set_discrimination_reference(torch.cat([e_a, e_b], dim=0))
    a_a_good, b_a_good = m.model_parameters(e_a)
    a_b_good, b_b_good = m.model_parameters(e_b)
    good_diff = ((a_a_good * theta).sum(-1) - b_a_good) - ((a_b_good * theta).sum(-1) - b_b_good)
    m.set_discrimination_reference(None)

    assert torch.allclose(fixed_diff, good_diff, atol=1e-5)
    assert not torch.allclose(fixed_diff, bad_diff, atol=1e-4)


def test_pairwise_logit_diff_rejects_free_model():
    m = NIRTModel(query_dim=8, dim=3, n_models=2, model_params="free")
    with pytest.raises(ValueError, match="projected"):
        pairwise_logit_diff(m, torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8))


def test_pairwise_loss_decreases_when_fit_directly():
    """A tiny gradient-descent sanity check: the auxiliary loss is differentiable
    and improves the model's ability to separate a clear winner from a clear loser."""
    torch.manual_seed(0)
    m = NIRTModel(query_dim=6, dim=2, model_params="projected", profile_dim=4)
    e_q = torch.randn(64, 6)
    e_a = torch.randn(64, 4)
    e_b = torch.randn(64, 4)
    y = torch.ones(64)   # model_a always wins in this toy set
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = pairwise_loss(m, e_q, e_a, e_b, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0]
