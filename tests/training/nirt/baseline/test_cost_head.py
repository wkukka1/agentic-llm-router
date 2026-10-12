"""Expected-cost probe head on ``BaselineNIRT`` (detached, quality path bit-identical)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from helpers import baseline_cfg

torch = pytest.importorskip("torch")

from router.nirt.components import CostHead
from training.nirt.baseline.checkpoint import load_checkpoint, save_checkpoint
from training.nirt.baseline.data import batched_forward, to_tensors
from training.nirt.baseline.model import build_baseline_model
from training.nirt.baseline.synthetic import make_synthetic, make_synthetic_continuous, to_arrays
from training.nirt.baseline.train import _clip_grads, fit, train_log_cost_mean

K = 2


def _with_cost(arr, *, seed=0, nan_frac=0.0):
    """log-cost = per-model level + query-length proxy (first e_q coord) + noise."""
    rng = np.random.default_rng(seed)
    level = np.linspace(-9.0, -5.0, int(arr.midx.max()) + 1)
    log_c = level[arr.midx] + 0.8 * arr.e_q[:, 0] + 0.05 * rng.standard_normal(len(arr))
    cost = np.exp(log_c).astype(np.float32)
    if nan_frac:
        cost[rng.random(len(cost)) < nan_frac] = np.nan
    return replace(arr, cost=cost)


def _arrays(head="zoib", seed=1, nan_frac=0.0):
    syn = (make_synthetic(n_queries=300, n_models=6, K=K, seed=seed) if head == "bernoulli"
           else make_synthetic_continuous(head, n_queries=300, n_models=6, K=K, seed=seed))
    tr, va = to_arrays(syn, seed=seed)
    return _with_cost(tr, seed=seed, nan_frac=nan_frac), _with_cost(va, seed=seed + 1, nan_frac=nan_frac)


def _cfg(head="zoib", cost=False, epochs=2, **train):
    cfg = baseline_cfg(response=head, k=K, epochs=epochs, patience=50, **train)
    if cost:
        cfg["cost"] = {"enabled": True, "weight": 1.0, "hidden": 16}
    return cfg


def _same_grad(a, b):
    return (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))


def _quality_state(model):
    cost_ids = {id(p) for p in model.cost_parameters()}
    named = dict(model.named_parameters())
    return {k: v.detach().clone() for k, v in named.items() if id(v) not in cost_ids}


# --------------------------------------------------------------------------- #
# CostHead                                                                    #
# --------------------------------------------------------------------------- #
def test_cost_head_nll_is_a_true_gaussian_nll():
    head = CostHead(4, None, init_mu=-7.0)
    y, mu, ls = torch.tensor([-6.0]), torch.tensor([-7.0]), torch.tensor([0.3])
    ref = -torch.distributions.Normal(mu, ls.exp()).log_prob(y)
    assert torch.allclose(head.nll(y, mu, ls), ref, atol=1e-6)


def test_cost_head_init_bias_and_lognormal_mean():
    head = CostHead(4, 8, init_mu=-7.5)
    mu, ls = head(torch.zeros(3, 4))
    assert torch.allclose(mu, torch.full((3,), -7.5), atol=1e-5)
    assert (ls >= -5).all() and (ls <= 3).all()
    exp = head.expected_cost(mu, ls)
    assert torch.allclose(exp, torch.exp(mu + 0.5 * torch.exp(2 * ls)))


def test_enabling_cost_head_without_init_mu_raises():
    with pytest.raises(ValueError, match="cost_init_mu"):
        build_baseline_model({"theta_dim": K, "use_cost_head": True}, n_models=3, query_dim=8)


# --------------------------------------------------------------------------- #
# bit identity of the quality path                                           #
# --------------------------------------------------------------------------- #
def _pair(head="zoib"):
    base = {"theta_dim": K, "model_params": "free", "query_hidden": 16, "use_length_head": False,
            "response_model": head, "ablation": {"use_relevance": False, "use_interaction": True}}
    torch.manual_seed(0)
    off = build_baseline_model(base, n_models=6, query_dim=32)
    torch.manual_seed(0)
    on = build_baseline_model({**base, "use_cost_head": True, "cost_init_mu": -7.0, "cost_hidden": 16},
                              n_models=6, query_dim=32)
    return off, on


@pytest.mark.parametrize("head", ["bernoulli", "zoib"])
def test_quality_init_forward_loss_and_grads_identical(head):
    off, on = _pair(head)
    q_off, q_on = _quality_state(off), _quality_state(on)
    assert q_off.keys() == q_on.keys()
    for k in q_off:
        assert torch.equal(q_off[k], q_on[k]), k

    tr, _ = _arrays(head)
    t = to_tensors(tr, model_params="free", use_relevance=False, use_warmup=False)
    e_q, ref, y = t["e_q"][:256], t["ref"][:256], t["y_soft"][:256]
    o_off, o_on = off(e_q, ref), on(e_q, ref)
    assert torch.equal(o_off.logit, o_on.logit)
    for k, v in o_off.response.params.items():
        assert torch.equal(v, o_on.response.params[k]), k
    l_off = off.response_head.loss(y, o_off.response)
    l_on = on.response_head.loss(y, o_on.response)
    assert torch.equal(l_off, l_on)

    log_c = t["log_cost"][:256]
    cost_loss = on.cost_head.nll(log_c, o_on.cost_mu, o_on.cost_log_sigma).mean()
    l_off.backward()
    (l_on + 10.0 * cost_loss).backward()     # a large cost term must not touch quality grads
    for (n, p_off), p_on in zip(off.named_parameters(), on.quality_parameters()):
        assert _same_grad(p_off.grad, p_on.grad), n
    _clip_grads(off, 0.01)
    _clip_grads(on, 0.01)
    for (n, p_off), p_on in zip(off.named_parameters(), on.quality_parameters()):
        assert _same_grad(p_off.grad, p_on.grad), f"after clip: {n}"


def test_cost_loss_has_no_path_to_quality_params():
    _, on = _pair()
    tr, _ = _arrays()
    t = to_tensors(tr, model_params="free", use_relevance=False, use_warmup=False)
    o = on(t["e_q"][:64], t["ref"][:64])
    loss = on.cost_head.nll(t["log_cost"][:64], o.cost_mu, o.cost_log_sigma).mean()
    grads = torch.autograd.grad(loss, on.quality_parameters(), allow_unused=True)
    assert all(g is None or not g.abs().any() for g in grads)


@pytest.mark.parametrize("head", ["bernoulli", "zoib"])
def test_fit_quality_params_identical_after_each_epoch(head, tmp_path):
    arrays = _arrays(head)
    runs = {}
    for cost in (False, True):
        runs[cost] = [fit(_cfg(head, cost=cost, epochs=e, grad_clip=0.5), arrays=arrays,
                          save=False, verbose=False) for e in (1, 2)]
    for e in range(2):
        q_off = _quality_state(runs[False][e].model)
        q_on = _quality_state(runs[True][e].model)
        for k in q_off:
            assert torch.equal(q_off[k], q_on[k]), f"epoch {e + 1}: {k}"
    # saved checkpoint quality tensors too
    m_on = runs[True][1]
    save_checkpoint(tmp_path, model=m_on.model, config=m_on.config, model_index=m_on.model_index,
                    query_dim=32, profile_dim=1, relevance_dim=0)
    loaded, _ = load_checkpoint(tmp_path)
    q_loaded, q_off = _quality_state(loaded), _quality_state(runs[False][1].model)
    for k in q_off:
        assert torch.equal(q_off[k], q_loaded[k]), f"checkpoint: {k}"
    quality = {k: v for k, v in runs[True][1].val_metrics.items() if not k.startswith("cost_")}
    assert quality == runs[False][1].val_metrics


# --------------------------------------------------------------------------- #
# training behaviour                                                          #
# --------------------------------------------------------------------------- #
def test_init_mu_is_train_split_mean_only():
    tr, va = _arrays()
    va = replace(va, cost=np.full(len(va), 1.0, np.float32))   # would shift the mean if leaked
    run = fit(_cfg(cost=True, epochs=1), arrays=(tr, va), save=False, verbose=False)
    assert run.config["model"]["cost_init_mu"] == pytest.approx(train_log_cost_mean(tr))
    assert train_log_cost_mean(tr) == pytest.approx(float(np.log(tr.cost).mean()), rel=1e-6)


def test_probe_recovers_synthetic_cost():
    from scipy.stats import spearmanr

    tr, va = _arrays()
    cfg = _cfg(cost=True, epochs=30, lr=3e-3, batch_size=64)
    cfg["cost"].update(hidden=64, probe_epochs=500, probe_patience=20, probe_lr=1e-2,
                       probe_batch_size=64)
    run = fit(cfg, arrays=(tr, va), save=False, verbose=False)
    pred = batched_forward(run.model, va, fields=("cost_mu",))["cost_mu"]
    assert spearmanr(pred, np.log(va.cost)).correlation > 0.9
    assert run.val_metrics["cost_spearman"] > 0.9
    assert {"val_cost_nll", "val_cost_mae_log", "val_cost_spearman"} <= set(run.history[-1])


def test_nan_costs_are_masked():
    tr, va = _arrays(nan_frac=0.3)
    run = fit(_cfg(cost=True, epochs=2), arrays=(tr, va), save=False, verbose=False)
    for p in run.model.cost_parameters():
        assert torch.isfinite(p).all()
    assert math.isfinite(run.val_metrics["cost_nll"])


def test_beta_nll_reweights_by_detached_sigma():
    head = CostHead(4, None, init_mu=0.0)
    y, mu = torch.tensor([1.0]), torch.tensor([0.0])
    ls = torch.tensor([0.7], requires_grad=True)
    assert torch.allclose(head.train_loss(y, mu, ls, beta=0.0), head.nll(y, mu, ls))
    w = head.train_loss(y, mu, ls, beta=0.5) / head.nll(y, mu, ls)
    assert torch.allclose(w, torch.exp(ls).detach())


def test_probe_phase_does_not_move_quality_params():
    tr, va = _arrays()
    base = fit(_cfg(cost=False, epochs=3), arrays=(tr, va), save=False, verbose=False)
    cfg = _cfg(cost=True, epochs=3)
    cfg["cost"].update(probe_epochs=20, probe_patience=20)
    run = fit(cfg, arrays=(tr, va), save=False, verbose=False)
    assert len(run.history[-1]["cost_probe"]) == 20
    q_off, q_on = _quality_state(base.model), _quality_state(run.model)
    for k in q_off:
        assert torch.equal(q_off[k], q_on[k]), k


def test_all_nan_batch_adds_zero_cost_loss():
    from training.nirt.baseline.train import _masked_cost_loss

    _, on = _pair()
    mu = torch.zeros(4, requires_grad=True)
    loss = _masked_cost_loss(on.cost_head, torch.full((4,), float("nan")), mu, torch.zeros(4))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(mu.grad).all()


def test_batched_forward_cost_fields():
    tr, va = _arrays()
    run = fit(_cfg(cost=True, epochs=1), arrays=(tr, va), save=False, verbose=False)
    f = batched_forward(run.model, va, fields=("cost_mu", "cost_log_sigma", "cost"))
    assert np.allclose(f["cost"], np.exp(f["cost_mu"] + 0.5 * np.exp(2 * f["cost_log_sigma"])),
                       rtol=1e-5)


def test_old_checkpoint_without_cost_keys_loads(tmp_path):
    arrays = _arrays()
    run = fit(_cfg(cost=False, epochs=1), arrays=arrays, save=False, verbose=False)
    save_checkpoint(tmp_path, model=run.model, config=run.config, model_index=run.model_index,
                    query_dim=32, profile_dim=1, relevance_dim=0)
    model, blob = load_checkpoint(tmp_path)
    assert not any("cost" in k for k in blob["model_cfg"])
    assert model.cost_head is None and model.cost_parameters() == []


def test_projected_cold_model_cost_is_finite():
    rng = np.random.default_rng(0)
    tr, va = _arrays()
    prof = rng.standard_normal((6, 8)).astype(np.float32)
    tr = replace(tr, e_m=prof[tr.midx])
    va = replace(va, e_m=prof[va.midx])
    held = tr.midx == 5                              # model 5 unseen in train
    tr = replace(tr, **{f: getattr(tr, f)[~held] for f in
                        ("e_q", "midx", "y", "y_soft", "query_ids", "model_ids", "e_m", "cost")})
    cfg = _cfg(cost=True, epochs=3)
    cfg["model"]["model_params"] = "projected"
    run = fit(cfg, arrays=(tr, va), save=False, verbose=False)
    cold = batched_forward(run.model, va, fields=("cost",))["cost"][va.midx == 5]
    assert len(cold) and np.isfinite(cold).all()


# --------------------------------------------------------------------------- #
# residual mode: log C_hat = train log C(m) + learned per-query correction    #
# --------------------------------------------------------------------------- #
def _residual_cfg(grad_clip=0.0, **cost):
    cfg = _cfg(cost=True, epochs=2, grad_clip=grad_clip)
    cfg["cost"].update(residual=True, **cost)
    return cfg


def test_residual_offsets_come_from_train_split_only():
    from training.nirt.baseline.data import cost_offset_table

    tr, va = _arrays()
    va = replace(va, cost=np.full(len(va), 123.0, np.float32))      # would leak if used
    run = fit(_residual_cfg(probe_epochs=0), arrays=(tr, va), save=False, verbose=False)
    table = run.config["model"]["cost_offsets"]
    assert table == cost_offset_table(tr.model_ids, tr.cost)
    lc = np.log(tr.cost)
    for m in np.unique(tr.model_ids):
        assert table["per_model"][str(m)] == pytest.approx(lc[tr.model_ids == m].mean())
    assert abs(run.config["model"]["cost_init_mu"]) < 1e-6        # residuals are mean-zero


def test_residual_head_starts_exactly_at_the_per_model_baseline():
    tr, va = _arrays()
    cfg = _residual_cfg(probe_epochs=0)
    cfg["cost"]["weight"] = 0.0                                    # nothing trains the head
    run = fit(cfg, arrays=(tr, va), save=False, verbose=False)
    va = attach(va, run)
    mu = batched_forward(run.model, va, fields=("cost_mu",))["cost_mu"]
    assert np.allclose(mu, va.cost_offset, atol=1e-5)


def attach(arr, run):
    from training.nirt.baseline.data import attach_cost_offset

    return attach_cost_offset(arr, run.config["model"])


def test_residual_unseen_model_gets_global_default():
    from training.nirt.baseline.data import attach_cost_offset

    tr, _ = _arrays()
    cfg = {"cost_residual": True, "cost_offsets": {"per_model": {"sm0": -1.0}, "default": -7.0}}
    off = attach_cost_offset(replace(tr, model_ids=np.array(["sm0", "ghost"])), cfg).cost_offset
    assert off.tolist() == [-1.0, -7.0]


def test_residual_forward_without_offset_raises():
    tr, va = _arrays()
    run = fit(_residual_cfg(probe_epochs=0), arrays=(tr, va), save=False, verbose=False)
    with pytest.raises(ValueError, match="cost_offset"):
        batched_forward(run.model, va, fields=("cost",))


def test_residual_quality_path_still_bit_identical():
    arrays = _arrays()
    off = fit(_cfg(cost=False, epochs=2, grad_clip=0.5), arrays=arrays, save=False, verbose=False)
    on = fit(_residual_cfg(grad_clip=0.5, probe_epochs=3), arrays=arrays, save=False,
             verbose=False)
    q_off, q_on = _quality_state(off.model), _quality_state(on.model)
    for k in q_off:
        assert torch.equal(q_off[k], q_on[k]), k


def test_residual_probe_keeps_model_ordering_and_learns_query_signal():
    from scipy.stats import spearmanr

    tr, va = _arrays()
    cfg = _cfg(cost=True, epochs=30, lr=3e-3, batch_size=64)
    cfg["cost"].update(residual=True, hidden=64, probe_epochs=300, probe_patience=20,
                       probe_lr=1e-2, probe_batch_size=64)
    run = fit(cfg, arrays=(tr, va), save=False, verbose=False)
    va = attach(va, run)
    mu = batched_forward(run.model, va, fields=("cost_mu",))["cost_mu"]
    true = np.log(va.cost)
    # beats the per-model baseline it starts from
    assert np.abs(mu - true).mean() < 0.5 * np.abs(va.cost_offset - true).mean()
    assert spearmanr(mu, true).correlation > 0.95


def test_residual_checkpoint_round_trip(tmp_path):
    tr, va = _arrays()
    run = fit(_residual_cfg(probe_epochs=2), arrays=(tr, va), save=False, verbose=False)
    save_checkpoint(tmp_path, model=run.model, config=run.config, model_index=run.model_index,
                    query_dim=32, profile_dim=1, relevance_dim=0)
    model, blob = load_checkpoint(tmp_path)
    assert model.cost_residual and blob["model_cfg"]["cost_offsets"] == run.config["model"]["cost_offsets"]
    a = batched_forward(run.model, attach(va, run), fields=("cost",))["cost"]
    from training.nirt.baseline.data import attach_cost_offset

    b = batched_forward(model, attach_cost_offset(va, blob["model_cfg"]), fields=("cost",))["cost"]
    assert np.array_equal(a, b)


def test_log_cost_floors_zero_but_keeps_missing_nan():
    """A genuine $0 (free model) becomes log(FREE_COST_FLOOR); NaN / negative stay
    NaN (unknown, never fabricated)."""
    from training.nirt.baseline.data import FREE_COST_FLOOR, log_cost

    lc = log_cost(np.array([0.0, 1e-4, np.nan, -1.0, np.inf]), 5)
    assert lc[0] == pytest.approx(math.log(FREE_COST_FLOOR))
    assert lc[1] == pytest.approx(math.log(1e-4))
    assert np.isnan(lc[2:]).all()
    assert lc[0] < lc[1]
