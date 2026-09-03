"""Phase 1 BaselineNIRT: shapes, IRT equation, ablation switches, identifiability."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from router.nirt.baseline import BaselineNIRT, build_baseline_model
from router.nirt.components import DiscriminationHead, InteractionLayer, WarmupBlender


def _model(**kw):
    base = dict(query_dim=16, dim=4, n_models=5, relevance_dim=6, profile_dim=10,
                model_params="free", query_hidden=8)
    base.update(kw)
    return BaselineNIRT(**base)


def test_output_shapes_and_ranges():
    m = _model()
    e_q = torch.randn(32, 16)
    midx = torch.randint(0, 5, (32,))
    r_q = torch.softmax(torch.randn(32, 6), dim=-1)
    out = m(e_q, midx, r_q)
    assert out.logit.shape == (32,)
    assert out.base_irt.shape == (32,)
    assert out.a_q.shape == (32, 4)
    assert out.b_q.shape == (32,)
    assert out.theta_m.shape == (32, 4)
    assert torch.isfinite(out.logit).all()
    p = out.proba()
    assert ((p >= 0) & (p <= 1)).all()
    assert (out.a_q >= 0).all()          # softplus / gate keep discrimination >= 0


def test_theta_table_shape():
    m = _model(dim=3, n_models=7)
    assert m.theta_table().shape == (7, 3)


def test_irt_equation_matches_base_when_interaction_off():
    m = _model(use_interaction=False, use_relevance=False)
    e_q = torch.randn(10, 16)
    midx = torch.arange(5).repeat(2)
    out = m(e_q, midx)
    manual = (out.a_q * out.theta_m).sum(-1) - out.b_q
    torch.testing.assert_close(out.logit, manual)
    torch.testing.assert_close(out.base_irt, manual)


def test_interaction_starts_as_pure_irt():
    """gamma initialised at 0 -> enabling the interaction layer is a no-op start."""
    m = _model(use_interaction=True, use_relevance=False)
    e_q = torch.randn(8, 16)
    out = m(e_q, torch.zeros(8, dtype=torch.long))
    torch.testing.assert_close(out.logit, out.base_irt)


def test_relevance_gate_is_identityish_at_init():
    m = _model(use_relevance=True)
    e_q = torch.randn(8, 16)
    r_q = torch.softmax(torch.randn(8, 6), dim=-1)
    with_rel = m(e_q, torch.zeros(8, dtype=torch.long), r_q)
    assert with_rel.relevance_gate is not None
    assert (with_rel.relevance_gate > 0.7).float().mean() > 0.9   # near pass-through


def test_ablation_switches_from_config():
    m = build_baseline_model(
        {"theta_dim": 2, "model_params": "free",
         "ablation": {"use_relevance": False, "use_interaction": False, "use_warmup": False}},
        n_models=3, query_dim=8, relevance_dim=5,
    )
    assert not m.use_relevance and not m.use_interaction and not m.use_warmup
    assert not m.disc_head.use_relevance


def test_projected_theta_cold_start_path():
    m = _model(model_params="projected", profile_dim=10)
    e_q = torch.randn(6, 16)
    e_m = torch.randn(6, 10)          # unseen-model profile embeddings
    out = m(e_q, e_m)
    assert out.theta_m.shape == (6, 4)
    assert ((out.theta_m >= 0) & (out.theta_m <= 1)).all()   # bound_ability sigmoid


def test_warmup_blender_passthrough_and_blend():
    wb = WarmupBlender(enabled=True, alpha=0.25)
    e = torch.randn(4, 5)
    nbr = torch.randn(4, 5)
    torch.testing.assert_close(wb(e, None), e)               # no neighbour mean -> passthrough
    torch.testing.assert_close(wb(e, nbr), 0.75 * e + 0.25 * nbr)
    nbr0 = nbr.clone(); nbr0[1] = 0
    assert torch.allclose(wb(e, nbr0)[1], e[1])              # zero row falls back to e_q


def test_recenter_of_theta_mean_penalty_shrinks_mean():
    from router.nirt.losses import RegConfig, regularization

    theta = torch.randn(9, 3) + 5.0
    reg = regularization(RegConfig(theta_center_l2=1.0, theta_l2=0.0),
                         theta_all=theta, a_q=torch.rand(4, 3), b_q=torch.randn(4))
    assert reg.item() == pytest.approx((theta.mean(0) ** 2).sum().item(), rel=1e-4)
