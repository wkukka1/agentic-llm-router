"""``router.nirt.model`` structure: forward shapes, discrimination constraint,
vector difficulty, query-head capacity knobs, the non-additive interaction
residual at init, and the ``IRTRouterModel`` (model-latent) orientation.

Anything that calls ``fit`` lives in ``test_nirt_train``; the classical baselines
and routing report live in ``test_nirt_routing``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from router.nirt.model import IRTRouterModel, NIRTModel, _query_head, build_model


# --------------------------------------------------------------------------- #
# NIRTModel forward                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["projected", "free"])
@pytest.mark.parametrize("K", [1, 4])
def test_forward_shapes_and_range(mode, K):
    B = 32
    model = NIRTModel(query_dim=10, dim=K, n_models=5, model_params=mode,
                      query_hidden=8, profile_dim=7)
    e_q = torch.randn(B, 10)
    ref = torch.randn(B, 7) if mode == "projected" else torch.randint(0, 5, (B,))
    logit = model(e_q, ref)
    assert logit.shape == (B,)
    proba = model.predict_proba(e_q, ref)
    assert proba.shape == (B,)
    assert torch.all((proba >= 0) & (proba <= 1))


def test_constrain_discrimination_positive():
    model = NIRTModel(query_dim=4, dim=1, n_models=3, model_params="projected",
                      query_hidden=None, constrain_discrimination=True, profile_dim=4)
    a, _ = model.model_parameters(torch.randn(16, 4))
    assert torch.all(a > 0)


def test_from_config_auto_constraint():
    m1 = NIRTModel.from_config({"dim": 1}, n_models=2)
    m4 = NIRTModel.from_config({"dim": 4}, n_models=2)
    assert m1.constrain_discrimination is True
    assert m4.constrain_discrimination is False


# --------------------------------------------------------------------------- #
# multidimensional difficulty (b_m in R^K)                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["projected", "free"])
def test_vector_difficulty_shapes(mode):
    B, K = 20, 8
    m = NIRTModel(query_dim=10, dim=K, n_models=5, model_params=mode,
                  query_hidden=16, profile_dim=7, difficulty="vector")
    e_q = torch.randn(B, 10)
    ref = torch.randn(B, 7) if mode == "projected" else torch.randint(0, 5, (B,))
    a, b = m.model_parameters(ref)
    assert a.shape == (B, K) and b.shape == (B, K)          # b is now a K-vector
    assert m(e_q, ref).shape == (B,)
    p = m.predict_proba(e_q, ref)
    assert torch.all((p >= 0) & (p <= 1))


def test_scalar_difficulty_unchanged():
    """difficulty='scalar' is the default and bit-identical to before."""
    torch.manual_seed(0)
    m_default = NIRTModel(query_dim=8, dim=4, n_models=3, model_params="projected",
                          query_hidden=None, profile_dim=6)
    torch.manual_seed(0)
    m_explicit = NIRTModel(query_dim=8, dim=4, n_models=3, model_params="projected",
                           query_hidden=None, profile_dim=6, difficulty="scalar")
    e_q, ref = torch.randn(12, 8), torch.randn(12, 6)
    assert torch.allclose(m_default(e_q, ref), m_explicit(e_q, ref))
    _, b = m_default.model_parameters(ref)
    assert b.shape == (12,)


def test_difficulty_from_config_and_validation():
    assert NIRTModel.from_config({"dim": 4}, n_models=2).difficulty == "scalar"
    assert NIRTModel.from_config({"dim": 4, "difficulty": "vector"}, n_models=2).difficulty == "vector"
    assert build_model({"orientation": "query_latent", "dim": 4, "difficulty": "vector"},
                       n_models=3).difficulty == "vector"
    with pytest.raises(ValueError, match="difficulty"):
        NIRTModel(query_dim=4, dim=2, n_models=2, difficulty="bogus")


# --------------------------------------------------------------------------- #
# P2a: query-head capacity knobs                                               #
# --------------------------------------------------------------------------- #
def test_query_head_default_identical():
    """The default P2a knobs must reproduce the legacy Linear->ReLU->Linear head
    exactly -- same module type and same state_dict keys -- so old checkpoints
    load under strict=True."""
    legacy = _query_head(16, 4, 8)
    withkw = _query_head(16, 4, 8, layers=1, norm="none", activation="relu",
                         dropout=0.0, residual=False)
    assert isinstance(legacy, torch.nn.Sequential) and isinstance(withkw, torch.nn.Sequential)
    assert list(legacy.state_dict()) == list(withkw.state_dict()) == ["0.weight", "0.bias", "2.weight", "2.bias"]

    torch.manual_seed(0)
    m_plain = NIRTModel(query_dim=12, dim=3, n_models=4, model_params="projected",
                        query_hidden=16, profile_dim=6)
    torch.manual_seed(0)
    m_defaults = NIRTModel.from_config(
        {"dim": 3, "model_params": "projected", "query_hidden": 16,
         "constrain_discrimination": True,
         "query_head_layers": 1, "query_head_norm": "none", "query_head_activation": "relu",
         "query_head_dropout": 0.0, "query_head_residual": False},
        n_models=4, query_dim=12, profile_dim=6,
    )
    assert list(m_plain.state_dict()) == list(m_defaults.state_dict())
    e_q, ref = torch.randn(8, 12), torch.randn(8, 6)
    assert torch.allclose(m_plain(e_q, ref), m_defaults(e_q, ref))
    assert not any("interaction" in k for k in m_plain.state_dict())


@pytest.mark.parametrize("norm", ["none", "layernorm"])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
@pytest.mark.parametrize("layers,residual", [(1, False), (2, False), (3, True)])
def test_query_head_knob_shapes(norm, activation, layers, residual):
    m = NIRTModel.from_config(
        {"dim": 4, "model_params": "projected", "query_hidden": 16,
         "query_head_layers": layers, "query_head_norm": norm,
         "query_head_activation": activation, "query_head_dropout": 0.1,
         "query_head_residual": residual},
        n_models=5, query_dim=10, profile_dim=7,
    )
    m.eval()
    e_q, ref = torch.randn(9, 10), torch.randn(9, 7)
    assert m(e_q, ref).shape == (9,)
    p = m.predict_proba(e_q, ref)
    assert torch.all((p >= 0) & (p <= 1))


def test_model_hidden_default_and_mlp():
    """model_hidden=None keeps bare Linear a_m/b_m heads (legacy state_dict);
    an int makes them a 1-hidden-layer MLP."""
    m_lin = NIRTModel.from_config(
        {"dim": 4, "model_params": "projected", "query_hidden": None}, n_models=3,
        query_dim=10, profile_dim=8,
    )
    assert isinstance(m_lin.a_head, torch.nn.Linear)
    assert [k for k in m_lin.state_dict() if k.startswith("a_head")] == ["a_head.weight", "a_head.bias"]

    m_mlp = NIRTModel.from_config(
        {"dim": 4, "model_params": "projected", "query_hidden": None, "model_hidden": 16},
        n_models=3, query_dim=10, profile_dim=8,
    )
    assert isinstance(m_mlp.a_head, torch.nn.Sequential)
    m_mlp.eval()
    e_q, ref = torch.randn(6, 10), torch.randn(6, 8)
    assert m_mlp(e_q, ref).shape == (6,)
    assert torch.all((m_mlp.predict_proba(e_q, ref) >= 0) & (m_mlp.predict_proba(e_q, ref) <= 1))


def test_query_head_config_validation():
    with pytest.raises(ValueError, match="query_head_norm"):
        NIRTModel.from_config({"dim": 2, "query_head_norm": "batchnorm"}, n_models=2)
    with pytest.raises(ValueError, match="query_head_activation"):
        NIRTModel.from_config({"dim": 2, "query_head_activation": "silu"}, n_models=2)


# --------------------------------------------------------------------------- #
# P2c: non-additive interaction residual (structure only)                      #
# --------------------------------------------------------------------------- #
def test_interaction_off_identical():
    torch.manual_seed(0)
    base = NIRTModel(query_dim=10, dim=3, n_models=4, model_params="projected",
                     query_hidden=8, profile_dim=6)
    assert not any("interaction" in k for k in base.state_dict())


def test_interaction_noop_at_init():
    """gamma starts at 0 so an interaction model's logit == the bilinear base at init."""
    torch.manual_seed(0)
    plain = NIRTModel(query_dim=10, dim=3, n_models=4, model_params="projected",
                      query_hidden=8, profile_dim=6)
    torch.manual_seed(0)
    inter = NIRTModel(query_dim=10, dim=3, n_models=4, model_params="projected",
                      query_hidden=8, profile_dim=6, interaction=True)
    e_q, ref = torch.randn(7, 10), torch.randn(7, 6)
    assert float(inter.interaction_gamma.detach()) == 0.0
    assert torch.allclose(plain(e_q, ref), inter(e_q, ref))


# --------------------------------------------------------------------------- #
# P7b: centering a_m / whitening theta_q (routing-collapse fixes)             #
# --------------------------------------------------------------------------- #
def test_center_discrimination_default_off_no_state_change():
    """center_discrimination has no learnable parameters -> off/on state_dict
    keys are identical (only the forward math differs when enabled)."""
    torch.manual_seed(0)
    off = NIRTModel(query_dim=6, dim=3, n_models=5, model_params="free", query_hidden=None)
    torch.manual_seed(0)
    on = NIRTModel(query_dim=6, dim=3, n_models=5, model_params="free", query_hidden=None,
                   center_discrimination=True)
    assert list(off.state_dict()) == list(on.state_dict())


def test_center_discrimination_lossless_for_ranking_free():
    """Centering a_m subtracts the SAME constant from every candidate at a given
    query -> pairwise logit differences (hence argmax) are unchanged."""
    torch.manual_seed(0)
    off = NIRTModel(query_dim=6, dim=3, n_models=5, model_params="free", query_hidden=None)
    torch.manual_seed(0)
    on = NIRTModel(query_dim=6, dim=3, n_models=5, model_params="free", query_hidden=None,
                  center_discrimination=True)
    e_q = torch.randn(1, 6).repeat(5, 1)   # one query, all 5 candidate models
    ref = torch.arange(5)
    logit_off, logit_on = off(e_q, ref), on(e_q, ref)
    diff_off = logit_off[:, None] - logit_off[None, :]
    diff_on = logit_on[:, None] - logit_on[None, :]
    assert torch.allclose(diff_off, diff_on, atol=1e-5)
    assert torch.equal(logit_off.argmax(), logit_on.argmax())


def test_center_discrimination_zeroes_pool_mean_free():
    torch.manual_seed(0)
    m = NIRTModel(query_dim=6, dim=3, n_models=6, model_params="free", query_hidden=None,
                 center_discrimination=True)
    a, _ = m.model_parameters(torch.arange(6))   # covers the full pool
    assert torch.allclose(a.mean(0), torch.zeros(3), atol=1e-5)


def test_query_head_batchnorm_default_off_no_state():
    m = NIRTModel(query_dim=8, dim=3, n_models=4, model_params="projected",
                 query_hidden=None, profile_dim=6)
    assert not any("query_bn" in k for k in m.state_dict())


def test_query_head_batchnorm_whitens_theta():
    torch.manual_seed(0)
    m = NIRTModel(query_dim=8, dim=4, n_models=4, model_params="projected",
                 query_hidden=None, profile_dim=6, query_head_batchnorm=True)
    m.train()
    theta = m.latent_query(torch.randn(256, 8))
    assert torch.allclose(theta.mean(0), torch.zeros(4), atol=1e-5)
    assert torch.allclose(theta.var(0, unbiased=False), torch.ones(4), atol=1e-3)


def test_center_and_batchnorm_from_config():
    m = NIRTModel.from_config(
        {"dim": 2, "model_params": "free", "center_discrimination": True,
         "query_head_batchnorm": True},
        n_models=3,
    )
    assert m.center_discrimination is True
    assert m.query_head_batchnorm is True
    assert isinstance(m.query_bn, torch.nn.BatchNorm1d)
    # defaults stay False
    d = NIRTModel.from_config({"dim": 2}, n_models=3)
    assert d.center_discrimination is False
    assert d.query_head_batchnorm is False


# --------------------------------------------------------------------------- #
# orientation: IRTRouterModel (model_latent) + build_model                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["projected", "free"])
@pytest.mark.parametrize("K", [1, 4])
def test_irtrouter_forward_shapes_and_range(mode, K):
    B = 24
    m = IRTRouterModel(query_dim=10, dim=K, n_models=5, model_params=mode,
                       query_hidden=8, profile_dim=7)
    e_q = torch.randn(B, 10)
    ref = torch.randn(B, 7) if mode == "projected" else torch.randint(0, 5, (B,))
    assert m(e_q, ref).shape == (B,)
    p = m.predict_proba(e_q, ref)
    assert torch.all((p >= 0) & (p <= 1))
    assert m.orientation == "model_latent"


def test_irtrouter_bounded_ability_and_softplus():
    m = IRTRouterModel(query_dim=6, dim=1, n_models=3, model_params="projected",
                       query_hidden=None, constrain_discrimination=True,
                       bound_ability=True, profile_dim=6)
    theta = m.latent_ability(torch.randn(12, 6))
    a, _ = m.query_parameters(torch.randn(12, 6))
    assert torch.all((theta >= 0) & (theta <= 1))
    assert torch.all(a > 0)


def test_build_model_orientation():
    assert build_model({"orientation": "query_latent", "dim": 2}, n_models=3).orientation == "query_latent"
    assert build_model({"orientation": "model_latent", "dim": 2}, n_models=3).orientation == "model_latent"
    assert build_model({"dim": 1}, n_models=3).orientation == "query_latent"   # default
    with pytest.raises(ValueError, match="orientation"):
        build_model({"orientation": "nonsense"}, n_models=3)
