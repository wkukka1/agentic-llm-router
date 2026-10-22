"""``training.nirt.elo_pseudo``: Arena Elo -> synthetic quality labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.nirt.elo_pseudo import (
    assign_families,
    dataset_families,
    fit_elo_quality_map,
    pool_category_elo,
    pseudo_observations,
)


def _elo_rows():
    rows = []
    for key, base in [("a-1", 1000), ("b-1", 1100), ("c-1", 1200), ("d-1", 1300), ("e-1", 1250)]:
        rows.append(("full", key, float(base)))
        rows.append(("math", key, float(base + (50 if key == "a-1" else 0))))
    return pd.DataFrame(rows, columns=["category", "arena_key", "rating"])


def _elo():
    cfg = {"models": {"a": {"arena_key": "a-1"}, "b": {"arena_key": "b-1"},
                      "b_cot": {"alias_of": "b"}, "u": {}},
           "extra_models": {"c": {"arena_key": "c-1"}, "d": {"arena_key": "d-1"},
                            "e": {"arena_key": "e-1"}, "x": {"arena_key": "not-on-arena"}}}
    return pool_category_elo(cfg, _elo_rows())


def test_pool_category_elo_resolves_aliases_and_drops_unrated():
    elo = _elo()
    assert sorted(elo.index) == ["a", "b", "b_cot", "c", "d", "e"]
    assert elo.at["b_cot", "full"] == elo.at["b", "full"] == 1100.0
    assert elo.at["a", "math"] == 1050.0


def test_dataset_families():
    assert dataset_families({"math": ["gsm8k", "math"], "code": ["mbpp"]}) == {
        "gsm8k": "math", "math": "math", "mbpp": "code"}


def test_fit_map_recovers_positive_slope_and_falls_back():
    elo = _elo()
    # mean score rises with Elo in math; `code` has too few models -> fallback
    perf = pd.DataFrame({"math": {"a": 0.3, "b": 0.4, "c": 0.55, "d": 0.7, "e": 0.65},
                         "code": {"a": 0.2, "b": np.nan, "c": np.nan, "d": np.nan, "e": np.nan}})
    fmap = fit_elo_quality_map(perf, elo)
    assert fmap["math"]["category"] == "math" and fmap["math"]["beta"] > 0
    assert fmap["math"]["r2"] > 0.8
    assert fmap["code"].get("fallback") and fmap["code"]["category"] == "full"
    with pytest.raises(ValueError, match="too few"):
        fit_elo_quality_map(perf.iloc[:2], elo)


def test_assign_families_nearest_centroid():
    rng = np.random.default_rng(0)
    E_ref = np.vstack([rng.normal([5, 0], 0.1, (20, 2)), rng.normal([0, 5], 0.1, (20, 2))])
    fam = ["math"] * 20 + ["code"] * 20
    got = assign_families(np.array([[3.0, 0.2], [0.1, 2.0]]), E_ref, fam)
    assert got.tolist() == ["math", "code"]


def test_pseudo_observations_monotone_in_elo():
    elo = _elo()
    perf = pd.DataFrame({"math": {"a": 0.3, "b": 0.4, "c": 0.55, "d": 0.7, "e": 0.65}})
    fmap = fit_elo_quality_map(perf, elo)
    obs = pseudo_observations(["q0", "q1"], ["math", "other"], elo, fmap)
    assert len(obs) == 2 * len(elo)
    assert set(obs["model_id"]) == set(elo.index)          # unrated `u`, `x` get no rows
    q1 = obs[obs.query_id == "q1"].set_index("model_id")["target"]
    order = elo["full"].sort_values().index
    assert np.all(np.diff(q1.reindex(order).to_numpy()) >= 0)
    assert obs["target"].between(0, 1).all()


def test_pseudo_observations_models_per_query():
    elo = _elo()
    perf = pd.DataFrame({"math": {"a": 0.3, "b": 0.4, "c": 0.55, "d": 0.7, "e": 0.65}})
    obs = pseudo_observations([f"q{i}" for i in range(50)], ["math"] * 50, elo,
                              fit_elo_quality_map(perf, elo), models_per_query=3, seed=1)
    assert len(obs) == 150
    assert (obs.groupby("query_id")["model_id"].nunique() == 3).all()
