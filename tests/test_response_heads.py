"""Shared ResponseHead interface + factory + BaselineNIRT wiring."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from router.nirt.baseline import BaselineNIRT
from router.nirt.response_head import RESPONSE_MODELS, build_response_head


@pytest.mark.parametrize("name", RESPONSE_MODELS)
def test_factory_and_common_api(name):
    head = build_response_head(name, {}, feature_dim=20, hidden=8)
    z = torch.randn(16)
    feats = torch.randn(16, 20)
    out = head(z, None if name == "bernoulli" else feats)
    y = torch.rand(16).clamp(1e-3, 1 - 1e-3)

    nll = head.nll(y, out)
    assert nll.shape == (16,) and torch.isfinite(nll).all()
    assert head.loss(y, out).ndim == 0
    for fn in (head.mean, head.variance, head.stddev):
        v = fn(out)
        assert v.shape == (16,) and torch.isfinite(v).all()
    p = head.as_probability(out)
    assert ((p >= 0) & (p <= 1)).all()
    lo, hi = head.interval(out, 0.9)
    assert (hi >= lo).all()
    assert head.lower_bound(out, 0.9).shape == (16,)


def test_bernoulli_is_parameter_free():
    assert sum(p.numel() for p in build_response_head("bernoulli", feature_dim=4).parameters()) == 0


def test_bernoulli_nll_equals_bce():
    import torch.nn.functional as F

    head = build_response_head("bernoulli", feature_dim=4)
    z, y = torch.randn(32), (torch.rand(32) > 0.5).float()
    out = head(z)
    torch.testing.assert_close(head.nll(y, out),
                               F.binary_cross_entropy_with_logits(z, y, reduction="none"))


@pytest.mark.parametrize("name", RESPONSE_MODELS)
def test_baseline_nirt_accepts_head(name):
    m = BaselineNIRT(query_dim=16, dim=4, n_models=5, model_params="free", query_hidden=8,
                     response_model=name)
    out = m(torch.randn(12, 16), torch.randint(0, 5, (12,)))
    assert out.response is not None
    y = torch.rand(12).clamp(1e-3, 1 - 1e-3)
    assert torch.isfinite(m.response_head.loss(y, out.response))
    assert m.predict_proba(torch.randn(3, 16), torch.zeros(3, dtype=torch.long)).shape == (3,)


def test_default_head_is_bernoulli_and_unchanged():
    m = BaselineNIRT(query_dim=8, dim=2, n_models=3, model_params="free", query_hidden=4)
    assert m.response_model == "bernoulli"
    e_q, midx = torch.randn(5, 8), torch.zeros(5, dtype=torch.long)
    out = m(e_q, midx)
    torch.testing.assert_close(out.proba(), torch.sigmoid(out.logit))
    # features not built for bernoulli
    assert out.response["logit"] is out.logit
