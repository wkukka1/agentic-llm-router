"""Output-length cost head (``cost.mode: length``): bucket classifier + token prices."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from helpers import baseline_cfg

torch = pytest.importorskip("torch")

from training.data.tokens import fit_token_prices
from training.nirt.baseline.checkpoint import load_checkpoint, save_checkpoint
from training.nirt.baseline.data import batched_forward
from training.nirt.baseline.length import (
    BUCKET_EDGES,
    attach_length_inputs,
    bucket_values,
    length_labels,
    length_prior_table,
)
from training.nirt.baseline.synthetic import make_synthetic_continuous, to_arrays
from training.nirt.baseline.train import fit

K = 2
P_IN = np.array([1, 2, 4, 8, 16, 32]) * 1e-7
P_OUT = 3 * P_IN
VERBOSE = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])      # log2 output length per model


def _tokens(arr, seed):
    rng = np.random.default_rng(seed)
    q = np.unique(arr.query_ids, return_inverse=True)[1]
    q_in = rng.integers(20, 400, q.max() + 1)
    task = (arr.e_q[:, 1] > 0).astype(float) * 4.0          # "long-answer task" from the query
    out = np.round(2 ** (VERBOSE[arr.midx] + task) * rng.uniform(0.8, 1.2, len(arr)))
    l_in = q_in[q].astype(np.float32)
    cost = (P_IN[arr.midx] * l_in + P_OUT[arr.midx] * out).astype(np.float32)
    return replace(arr, in_tokens=l_in, out_tokens=out.astype(np.float32), cost=cost)


def _arrays(seed=1, n_queries=300):
    syn = make_synthetic_continuous("zoib", n_queries=n_queries, n_models=6, K=K, seed=seed)
    tr, va = to_arrays(syn, seed=seed)
    return _tokens(tr, seed), _tokens(va, seed + 1)


def _cfg(cost=True, epochs=2, **cost_over):
    cfg = baseline_cfg(response="zoib", k=K, epochs=epochs, patience=50)
    if cost:
        cfg["cost"] = {"enabled": True, "mode": "length", "hidden": 16, **cost_over}
    return cfg


def _quality_state(model):
    cost_ids = {id(p) for p in model.cost_parameters()}
    return {k: v.detach().clone() for k, v in model.named_parameters() if id(v) not in cost_ids}


def _attach(arr, run):
    return attach_length_inputs(arr, run.config["model"])


# --------------------------------------------------------------------------- #
# tables                                                                      #
# --------------------------------------------------------------------------- #
def test_length_labels_and_bucket_values():
    lab = length_labels([0, 1, 2, 3, 4, 5, 2048, 2049, 10 ** 6, np.nan])
    assert lab.tolist() == [0, 1, 2, 3, 3, 4, 12, 13, 13, -1]
    vals = bucket_values([1, 1, 3, 4, 100])
    assert vals[1] == 1.0 and vals[3] == 3.5 and vals[0] == 0.0       # empty -> lower edge


def test_prior_table_is_smoothed_per_model_with_pooled_default():
    t = length_prior_table(["a", "a", "b", "z"], [0, 0, 1, -1], n_buckets=3, alpha=1.0)
    assert np.allclose(np.exp(t["per_model"]["a"]), [3 / 5, 1 / 5, 1 / 5])
    assert "z" not in t["per_model"]                                    # no labelled row
    assert np.allclose(np.exp(t["default"]), [3 / 6, 2 / 6, 1 / 6])


def test_fit_token_prices_recovers_exact_prices():
    rng = np.random.default_rng(0)
    rows = []
    for m in range(3):
        li, lo = rng.integers(1, 500, 200), rng.integers(1, 500, 200)
        rows.append(pd.DataFrame({"model_id": f"m{m}", "input_tokens": li, "output_tokens": lo,
                                  "cost": P_IN[m] * li + P_OUT[m] * lo}))
    fit = fit_token_prices(pd.concat(rows))
    for m in range(3):
        assert fit[f"m{m}"]["p_in"] == pytest.approx(P_IN[m], rel=1e-6)
        assert fit[f"m{m}"]["p_out"] == pytest.approx(P_OUT[m], rel=1e-6)
        assert fit[f"m{m}"]["r2"] == pytest.approx(1.0)


def test_unseen_model_gets_default_prior_and_nan_price():
    tr, _ = _arrays()
    mcfg = {"cost_mode": "length",
            "length_priors": {"per_model": {"sm0": [0.0] * 14}, "default": [-1.0] * 14},
            "token_prices": {"sm0": {"p_in": 1.0, "p_out": 2.0}}}
    a = attach_length_inputs(replace(tr, model_ids=np.array(["sm0", "ghost"])), mcfg)
    assert a.length_prior[1].tolist() == [-1.0] * 14
    assert a.token_price[0].tolist() == [1.0, 2.0] and np.isnan(a.token_price[1]).all()


# --------------------------------------------------------------------------- #
# model / training                                                            #
# --------------------------------------------------------------------------- #
def test_tables_come_from_train_split_only():
    tr, va = _arrays()
    va = replace(va, out_tokens=np.full(len(va), 5000, np.float32),
                 cost=np.full(len(va), 9.0, np.float32))
    run = fit(_cfg(probe_epochs=0), arrays=(tr, va), save=False, verbose=False)
    m = run.config["model"]
    assert m["length_buckets"]["values"] == bucket_values(tr.out_tokens)
    assert m["token_prices"]["sm3"]["p_in"] == pytest.approx(P_IN[3], rel=1e-4)
    assert m["token_prices"]["sm3"]["p_out"] == pytest.approx(P_OUT[3], rel=1e-4)


def test_head_starts_exactly_at_the_per_model_prior():
    tr, va = _arrays()
    run = fit(_cfg(probe_epochs=0, weight=0.0), arrays=(tr, va), save=False, verbose=False)
    va = _attach(va, run)
    logits = batched_forward(run.model, va, fields=("length_logits",))["length_logits"]
    assert np.allclose(logits, va.length_prior, atol=1e-6)


def test_quality_path_bit_identical_with_length_head():
    arrays = _arrays()
    off = fit(_cfg(cost=False, epochs=2), arrays=arrays, save=False, verbose=False)
    on = fit(_cfg(epochs=2, probe_epochs=3), arrays=arrays, save=False, verbose=False)
    q_off, q_on = _quality_state(off.model), _quality_state(on.model)
    assert q_off.keys() == q_on.keys()
    for k in q_off:
        assert torch.equal(q_off[k], q_on[k]), k


def test_length_head_learns_query_dependence_and_prices_cost():
    from scipy.stats import spearmanr

    tr, va = _arrays(n_queries=1500)     # the model x task interaction needs data to resolve
    cfg = _cfg(epochs=10, hidden=64, probe_epochs=300, probe_patience=20, probe_lr=3e-3,
               probe_batch_size=256)
    cfg["train"].update(batch_size=256, lr=3e-3)
    run = fit(cfg, arrays=(tr, va), save=False, verbose=False)
    vm = run.val_metrics
    va = _attach(va, run)
    f = batched_forward(run.model, va, fields=("cost", "expected_length"))
    # proper scoring rule: the query-aware head beats the per-model prior it starts from
    lab = length_labels(va.out_tokens)
    prior_ce = float(-(va.length_prior - np.log(np.exp(va.length_prior).sum(1, keepdims=True)))
                     [np.arange(len(lab)), lab].mean())
    assert vm["length_ce"] < prior_ce - 0.3
    assert spearmanr(f["cost"], va.cost).correlation > 0.95
    assert vm["cost_spearman"] > 0.95
    # E[cost] is exactly p_in * L_in + p_out * E[L_out]
    p = np.array([[run.config["model"]["token_prices"][m]["p_in"],
                   run.config["model"]["token_prices"][m]["p_out"]] for m in va.model_ids])
    assert np.allclose(f["cost"], p[:, 0] * va.in_tokens + p[:, 1] * f["expected_length"],
                       rtol=1e-5)


def test_forward_without_prior_raises():
    tr, va = _arrays()
    run = fit(_cfg(probe_epochs=0), arrays=(tr, va), save=False, verbose=False)
    with pytest.raises(ValueError, match="length_prior"):
        batched_forward(run.model, va, fields=("expected_length",))


def test_explicit_arrays_without_tokens_raise():
    tr, va = _arrays()
    tr = replace(tr, in_tokens=None, out_tokens=None)
    with pytest.raises(ValueError, match="in_tokens / out_tokens"):
        fit(_cfg(probe_epochs=0), arrays=(tr, va), save=False, verbose=False)


def test_prior_table_keys_are_plain_str():
    t = length_prior_table(np.array(["a", "b"]), [0, 1], n_buckets=2)
    assert all(type(k) is str for k in t["per_model"])


def test_checkpoint_round_trip(tmp_path):
    tr, va = _arrays()
    run = fit(_cfg(probe_epochs=2), arrays=(tr, va), save=False, verbose=False)
    save_checkpoint(tmp_path, model=run.model, config=run.config, model_index=run.model_index,
                    query_dim=32, profile_dim=1, relevance_dim=0)
    model, blob = load_checkpoint(tmp_path)
    assert model.cost_mode == "length"
    a = batched_forward(run.model, _attach(va, run), fields=("cost",))["cost"]
    b = batched_forward(model, attach_length_inputs(va, blob["model_cfg"]), fields=("cost",))["cost"]
    assert np.array_equal(a, b)
    assert len(BUCKET_EDGES) == model.cost_head.n_buckets
