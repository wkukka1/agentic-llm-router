from __future__ import annotations

import numpy as np
import pandas as pd

from training.models.manifold import build_feature_matrix, fit_manifold, project_model


def _planted(m=40, f=10, rank=2, missing=0.3, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(m, rank))
    W = rng.normal(size=(f, rank))
    W[0] = [1.0, 0.0]                      # feature 0 (the "Elo" anchor) is axis 0
    X = Z @ W.T + 0.01 * rng.normal(size=(m, f))
    Xm = X.copy()
    mask = rng.random(X.shape) < missing
    mask[:, 0] = rng.random(m) < 0.15      # anchor mostly observed
    Xm[mask] = np.nan
    return X, Xm, mask


def test_recovers_planted_rank2_with_missing_cells():
    X, Xm, mask = _planted()
    man = fit_manifold(Xm, 2)
    Xs = (X - man["mean"]) / man["std"]
    recon = man["Z"] @ man["W"].T
    # held-out (missing) cells are predicted from the low-rank structure
    assert np.sqrt(np.mean((recon - Xs)[mask] ** 2)) < 0.15
    assert man["explained"] > 0.99


def test_axis0_is_the_anchor_feature():
    X, Xm, _ = _planted()
    man = fit_manifold(Xm, 2)
    elo_std = (X[:, 0] - man["mean"][0]) / man["std"][0]
    rated = ~np.isnan(Xm[:, 0])
    np.testing.assert_allclose(man["Z"][rated, 0], elo_std[rated], atol=1e-12)
    assert np.corrcoef(man["Z"][:, 0], elo_std)[0, 1] > 0.99      # imputed rows too


def test_project_model_matches_fitted_coordinates():
    _, Xm, _ = _planted()
    man = fit_manifold(Xm, 2, n_iter=2000, tol=1e-14)
    for i in (0, 7, 23):
        np.testing.assert_allclose(project_model(Xm[i], man), man["Z"][i], atol=1e-5)


def test_project_model_with_nothing_observed_is_the_pool_mean():
    _, Xm, _ = _planted()
    man = fit_manifold(Xm, 2)
    np.testing.assert_allclose(project_model(np.full(Xm.shape[1], np.nan), man), 0.0)


def test_build_feature_matrix_alias_and_missing():
    cfg = {
        "arena": {"anchor_category": "full", "categories": ["full", "math"]},
        "open_llm_leaderboard": {"columns": ["BBH", "MATH Lvl 5"],
                                  "overlaps": {"MATH Lvl 5": ["math"]}},
        "models": {"a": {"arena_key": "A", "hf_repo_id": "org/a"},
                   "a_cot": {"alias_of": "a"},
                   "b": {}},
    }
    elo = pd.DataFrame([{"category": "full", "arena_key": "A", "rating": 1200.0},
                        {"category": "math", "arena_key": "A", "rating": 1150.0}])
    olb = [{"fullname": "org/a", "BBH": 40.0, "MATH Lvl 5": 20.0}]
    ids, X, cols = build_feature_matrix(cfg, elo, olb)
    assert ids == ["a", "a_cot", "b"]
    assert cols == ["elo:full", "elo:math", "olb:BBH", "olb:MATH Lvl 5"]
    np.testing.assert_array_equal(X[0], X[1])
    assert np.isnan(X[2]).all()
    _, _, cols = build_feature_matrix(cfg, elo, olb, drop_overlapping=True)
    assert "olb:MATH Lvl 5" not in cols
