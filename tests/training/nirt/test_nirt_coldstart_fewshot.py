"""P7f few-shot cold-start: ``training.nirt.coldstart.fit_probe`` (MAP fit of a
new model's (a_m*, b_m*) against a frozen theta_q, regularized toward a
zero-shot prior) and ``evaluation.nirt.coldstart_eval.lomo_eval`` (the
leave-one-model-out learning-curve harness built on top of it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from evaluation.nirt.coldstart_eval import lomo_eval
from training.nirt.coldstart import fit_probe


# --------------------------------------------------------------------------- #
# fit_probe                                                                    #
# --------------------------------------------------------------------------- #
def test_fit_probe_zero_probe_returns_prior_unchanged():
    prior_a = np.array([0.7, -0.3])
    prior_b = 0.2
    a, b = fit_probe(np.zeros((0, 2)), np.zeros(0), prior_a=prior_a, prior_b=prior_b)
    assert np.allclose(a, prior_a)
    assert b == pytest.approx(prior_b)


def test_fit_probe_recovers_true_params_with_enough_data_and_low_ridge():
    rng = np.random.default_rng(0)
    true_a, true_b = np.array([1.5, -0.8]), 0.3
    theta = rng.standard_normal((400, 2))
    p = 1 / (1 + np.exp(-(theta @ true_a - true_b)))
    y = (rng.random(400) < p).astype(np.float64)

    bad_prior_a, bad_prior_b = np.array([-2.0, 2.0]), -3.0   # deliberately wrong
    a, b = fit_probe(theta, y, prior_a=bad_prior_a, prior_b=bad_prior_b,
                     ridge=1e-3, epochs=500, lr=0.05)
    assert np.allclose(a, true_a, atol=0.3)
    assert abs(b - true_b) < 0.3


def test_fit_probe_high_ridge_stays_near_prior():
    rng = np.random.default_rng(1)
    theta = rng.standard_normal((50, 2))
    y = rng.integers(0, 2, 50).astype(np.float64)
    prior_a, prior_b = np.array([0.5, -0.5]), 0.1
    a, b = fit_probe(theta, y, prior_a=prior_a, prior_b=prior_b,
                     ridge=1e6, epochs=100, lr=0.05)
    assert np.allclose(a, prior_a, atol=1e-2)
    assert abs(b - prior_b) < 1e-2


def test_fit_probe_more_data_moves_closer_to_truth_than_less():
    """Monotonicity sanity check: a bigger probe set should pull the estimate
    closer to the true params than a tiny one, for the same bad prior."""
    rng = np.random.default_rng(2)
    true_a, true_b = np.array([1.0, -1.0]), 0.0
    theta = rng.standard_normal((300, 2))
    p = 1 / (1 + np.exp(-(theta @ true_a - true_b)))
    y = (rng.random(300) < p).astype(np.float64)
    bad_prior_a, bad_prior_b = np.array([-1.5, 1.5]), 2.0

    a_small, b_small = fit_probe(theta[:10], y[:10], prior_a=bad_prior_a, prior_b=bad_prior_b,
                                 ridge=0.1, epochs=300, lr=0.05)
    a_big, b_big = fit_probe(theta, y, prior_a=bad_prior_a, prior_b=bad_prior_b,
                             ridge=0.1, epochs=300, lr=0.05)
    err_small = np.linalg.norm(a_small - true_a) + abs(b_small - true_b)
    err_big = np.linalg.norm(a_big - true_a) + abs(b_big - true_b)
    assert err_big < err_small


# --------------------------------------------------------------------------- #
# lomo_eval                                                                    #
# --------------------------------------------------------------------------- #
def _linear_world(seed=0, n_q=300, K=2, true_a=(1.2, -0.9), true_b=0.2, model_id="new_model"):
    rng = np.random.default_rng(seed)
    qids = [f"q{i}" for i in range(n_q)]
    theta_q = {qid: rng.standard_normal(K) for qid in qids}
    true_a = np.asarray(true_a)
    rows = []
    for qid in qids:
        p = 1 / (1 + np.exp(-(theta_q[qid] @ true_a - true_b)))
        rows.append({"query_id": qid, "model_id": model_id, "target": float(rng.random() < p)})
    return theta_q, pd.DataFrame(rows)


def test_lomo_eval_more_probe_data_beats_a_bad_prior():
    theta_q, obs = _linear_world(seed=3)
    bad_prior = (np.array([-1.0, 1.0]), -1.0)   # deliberately poor
    res = lomo_eval(theta_q, obs, {"new_model": bad_prior},
                    k_values=[0, 100], n_repeats=3, ridge=1e-2, seed=0)
    bce_by_k = res.groupby("k")["bce"].mean()
    assert bce_by_k[100] < bce_by_k[0]


def test_lomo_eval_zero_k_is_deterministic_and_matches_prior_score():
    from training.nirt.metrics import prediction_metrics

    theta_q, obs = _linear_world(seed=4)
    prior_a, prior_b = np.array([0.3, -0.2]), 0.1
    res = lomo_eval(theta_q, obs, {"new_model": (prior_a, prior_b)},
                    k_values=[0], n_repeats=5, ridge=1.0, seed=0)
    assert len(res) == 1   # k=0 -> exactly 1 rep regardless of n_repeats
    y = obs["target"].to_numpy()
    theta_mat = np.stack([theta_q[q] for q in obs["query_id"]])
    p = 1 / (1 + np.exp(-(theta_mat @ prior_a - prior_b)))
    expected_bce = prediction_metrics(y, p)["bce"]
    assert res.iloc[0]["bce"] == pytest.approx(expected_bce)


def test_lomo_eval_skips_models_without_a_prior():
    theta_q, obs = _linear_world(seed=5, model_id="has_prior")
    theta_q2, obs2 = _linear_world(seed=6, model_id="no_prior")
    combined_theta = {**theta_q, **theta_q2}
    combined_obs = pd.concat([obs, obs2], ignore_index=True)
    res = lomo_eval(combined_theta, combined_obs, {"has_prior": (np.zeros(2), 0.0)},
                    k_values=[0], n_repeats=1, seed=0)
    assert set(res["model_id"]) == {"has_prior"}


def test_lomo_eval_output_columns():
    theta_q, obs = _linear_world(seed=7)
    res = lomo_eval(theta_q, obs, {"new_model": (np.zeros(2), 0.0)}, k_values=[0, 10], n_repeats=1)
    for col in ("model_id", "k", "repeat", "n_probe", "n_eval", "bce", "auc"):
        assert col in res.columns
