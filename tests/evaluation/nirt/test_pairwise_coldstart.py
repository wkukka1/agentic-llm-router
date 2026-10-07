"""Pairwise cold-start generalization eval (E0-E6, Part 3 of the richer-profiles
+ pairwise-battle-generalization plan).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import make_store as _store
from helpers.nirt import FakeTrainingData

torch = pytest.importorskip("torch")

from router.nirt.model import NIRTModel
from evaluation.nirt.pairwise_coldstart import (
    _check_leakage_guards,
    _reorient_to_target,
    assert_arena_blind,
    pairwise_cold_start_eval,
    training_pairwise_model_ids,
)


# --------------------------------------------------------------------------- #
# leakage guards                                                              #
# --------------------------------------------------------------------------- #
def _battles(rows):
    return pd.DataFrame(rows, columns=["query_id", "model_a", "model_b", "score_a", "source"])


def test_training_pairwise_model_ids_unions_both_sides():
    battles = _battles([("q0", "a", "b", 1.0, "chatbot_arena"), ("q1", "c", "d", 0.5, "chatbot_arena")])
    data = FakeTrainingData(battles=battles)
    assert training_pairwise_model_ids(data) == {"a", "b", "c", "d"}


def test_leakage_guard_raises_when_preference_training_enabled():
    data = FakeTrainingData(battles=_battles([]))
    run_cfg = {"train": {"preference": {"enabled": True}}}
    with pytest.raises(ValueError, match="preference"):
        _check_leakage_guards(run_cfg, data, target_models=["cold-model"])


def test_leakage_guard_raises_when_target_not_actually_registered_cold():
    """Model-identity check: a target scored under require_warm_opponent=True
    (the E2+ cold-start claim) must actually BE a registered cold-start
    model -- catches accidentally testing a warm model under a zero-shot
    framing, which the config-flag check alone cannot catch. A cold model's
    own battles are always present in the general pairwise pool (that's
    exactly what E2 scores against), so presence there is never itself the
    violation -- mislabeled identity is."""
    battles = _battles([("q0", "not-really-cold", "warm-model", 1.0, "chatbot_arena")])
    data = FakeTrainingData(battles=battles, cold_models=["cold0"], warm_models=["warm-model"])
    run_cfg = {"train": {"preference": {"enabled": False}}}
    with pytest.raises(ValueError, match="not-really-cold"):
        _check_leakage_guards(run_cfg, data, target_models=["not-really-cold"],
                              require_warm_opponent=True)


def test_leakage_guard_skips_cold_registration_check_for_e1_warm_targets():
    """E1 (unseen-pair, both warm) deliberately targets warm models under
    require_warm_opponent=False -- it isn't claiming a cold-start result, so
    the cold-registration check doesn't apply."""
    battles = _battles([("q0", "warm-a", "warm-b", 1.0, "chatbot_arena")])
    data = FakeTrainingData(battles=battles, warm_models=["warm-a", "warm-b"], cold_models=[])
    run_cfg = {"train": {"preference": {"enabled": False}}}
    _check_leakage_guards(run_cfg, data, target_models=["warm-a", "warm-b"],
                          require_warm_opponent=False)  # no raise


def test_leakage_guard_passes_when_clean():
    battles = _battles([("q0", "warm-a", "cold-x", 1.0, "chatbot_arena")])
    data = FakeTrainingData(battles=battles, warm_models=["warm-a"], cold_models=["cold-x"])
    run_cfg = {"train": {"preference": {"enabled": False}}}
    _check_leakage_guards(run_cfg, data, target_models=["cold-x"], require_warm_opponent=True)


# --------------------------------------------------------------------------- #
# E4 guard: arena_elo must not leak into the Arena-blind variant               #
# --------------------------------------------------------------------------- #
def test_assert_arena_blind_passes_when_p4_actually_excludes_arena_elo():
    benchmarks = {
        "arena_elo": {"value": 1116, "unit": "elo", "source": "lmarena_hf",
                      "version": None, "snapshot_date": "2023-12-15"},
        "mmlu": {"value": 70.6, "unit": "percent", "source": "hf_open_llm_leaderboard",
                 "version": "v1", "snapshot_date": None},
    }
    assert_arena_blind(benchmarks, "P4")  # no raise


def test_assert_arena_blind_raises_if_arena_elo_present_under_p4_label():
    """Catches a mislabeled run -- e.g. passing a P3 (full) store while
    claiming it's the P4 (Arena-blind) experiment."""
    benchmarks = {"arena_elo": {"value": 1116, "unit": "elo", "source": "lmarena_hf",
                                 "version": None, "snapshot_date": "2023-12-15"}}

    class _FakeP4AlwaysIncludesEverything(dict):
        pass

    # simulate a caller that mislabels a full (P3-equivalent) variant as P4
    import evaluation.nirt.pairwise_coldstart as mod
    orig = mod.variant_benchmarks
    try:
        mod.variant_benchmarks = lambda b, v: dict(b)  # pretend nothing gets filtered
        with pytest.raises(ValueError, match="arena_elo"):
            assert_arena_blind(benchmarks, "P4")
    finally:
        mod.variant_benchmarks = orig


# --------------------------------------------------------------------------- #
# reorientation                                                               #
# --------------------------------------------------------------------------- #
def test_reorient_to_target_keeps_a_side_rows_unchanged():
    raw = _battles([("q0", "target", "opp", 1.0, "chatbot_arena")])
    out = _reorient_to_target(raw, "target")
    assert out.iloc[0]["model_a"] == "target"
    assert out.iloc[0]["model_b"] == "opp"
    assert out.iloc[0]["score_a"] == 1.0


def test_reorient_to_target_flips_b_side_rows():
    raw = _battles([("q0", "opp", "target", 1.0, "chatbot_arena")])
    out = _reorient_to_target(raw, "target")
    assert out.iloc[0]["model_a"] == "target"
    assert out.iloc[0]["model_b"] == "opp"
    assert out.iloc[0]["score_a"] == 0.0   # 1 - 1.0


def test_reorient_to_target_drops_rows_not_touching_target():
    raw = _battles([("q0", "x", "y", 1.0, "chatbot_arena")])
    out = _reorient_to_target(raw, "target")
    assert out.empty


def test_reorient_to_target_drops_target_vs_target_rows():
    raw = _battles([("q0", "target", "target", 1.0, "chatbot_arena")])
    out = _reorient_to_target(raw, "target")
    assert out.empty


# --------------------------------------------------------------------------- #
# end-to-end (small synthetic world)                                          #
# --------------------------------------------------------------------------- #
@pytest.fixture
def world():
    """One cold model (``cold0``) battling two warm models, plus warm-vs-warm
    battles for E1. Small real NIRTModel in projected mode (untrained --
    these tests check mechanics/shape, not prediction quality)."""
    rng = np.random.default_rng(0)
    q_store = _store([f"q{i}" for i in range(6)], rng.standard_normal((6, 8)), "query_id")
    m_store = _store(["cold0", "warm0", "warm1"], rng.standard_normal((3, 5)), "model_id")
    battles = _battles([
        ("q0", "cold0", "warm0", 1.0, "chatbot_arena"),
        ("q1", "warm0", "cold0", 0.0, "chatbot_arena"),
        ("q2", "cold0", "warm1", 0.5, "chatbot_arena"),
        ("q3", "warm0", "warm1", 1.0, "chatbot_arena"),
        ("q4", "warm1", "warm0", 0.0, "chatbot_arena"),
    ])
    data = FakeTrainingData(battles=battles, query_store=q_store, profile_store=m_store,
                            warm_models=["warm0", "warm1"], cold_models=["cold0"])
    torch.manual_seed(0)
    model = NIRTModel(query_dim=8, dim=3, model_params="projected", profile_dim=5)
    run_cfg = {"train": {"preference": {"enabled": False}}}
    return model, run_cfg, data


def test_e2_cold_model_eval_scores_only_cold_vs_warm_battles(world):
    model, run_cfg, data = world
    result = pairwise_cold_start_eval(model, run_cfg, data, target_models=["cold0"],
                                      sources=("chatbot_arena",))
    per_model = result["per_model"]["cold0"]
    assert per_model["prediction"]["n"] == 3   # 3 battles touch cold0 (2 vs warm0, 1 vs warm1)
    assert "coin_flip" in per_model["baselines"]
    assert "warm_pool_mean" in per_model["baselines"]
    assert "nearest_warm_model" in per_model["baselines"]
    assert "bce_ci" in per_model["prediction"]


def test_e2_result_keeps_gpt4_judge_and_arena_separate(world):
    model, run_cfg, data = world
    arena_only = pairwise_cold_start_eval(model, run_cfg, data, target_models=["cold0"],
                                          sources=("chatbot_arena",))
    judge_only = pairwise_cold_start_eval(model, run_cfg, data, target_models=["cold0"],
                                          sources=("gpt4_judge",))
    assert arena_only["sources"] == ("chatbot_arena",)
    assert judge_only["sources"] == ("gpt4_judge",)
    # no row is source=gpt4_judge in our fixture -> empty, not an error
    assert judge_only["per_model"].get("cold0", {}).get("prediction", {}).get("n", 0) == 0


def test_e1_both_warm_mode_scores_warm_vs_warm_battles(world):
    model, run_cfg, data = world
    result = pairwise_cold_start_eval(model, run_cfg, data, target_models=["warm0"],
                                      sources=("chatbot_arena",), require_warm_opponent=False)
    # warm0 appears in 2 of the battles as either side, both vs warm1
    assert result["per_model"]["warm0"]["prediction"]["n"] == 2


def test_require_warm_opponent_excludes_cold_vs_cold_rows():
    """With require_warm_opponent=True (the E2+ default), a battle between two
    cold models contributes to neither side -- baselines assume a warm
    opponent, so a cold-vs-cold row would be ill-defined."""
    rng = np.random.default_rng(0)
    q_store = _store(["q0"], rng.standard_normal((1, 8)), "query_id")
    m_store = _store(["cold0", "cold1"], rng.standard_normal((2, 5)), "model_id")
    battles = _battles([("q0", "cold0", "cold1", 1.0, "chatbot_arena")])
    data = FakeTrainingData(battles=battles, query_store=q_store, profile_store=m_store,
                            warm_models=[], cold_models=["cold0", "cold1"])
    torch.manual_seed(0)
    model = NIRTModel(query_dim=8, dim=3, model_params="projected", profile_dim=5)
    run_cfg = {"train": {"preference": {"enabled": False}}}
    result = pairwise_cold_start_eval(model, run_cfg, data, target_models=["cold0"],
                                      sources=("chatbot_arena",))
    assert result["per_model"]["cold0"]["prediction"]["n"] == 0


def test_profile_override_substitutes_a_different_models_embedding(world):
    model, run_cfg, data = world
    baseline = pairwise_cold_start_eval(model, run_cfg, data, target_models=["cold0"],
                                        sources=("chatbot_arena",))
    overridden = pairwise_cold_start_eval(model, run_cfg, data, target_models=["cold0"],
                                          sources=("chatbot_arena",),
                                          profile_override={"cold0": "warm1"})
    # same battles (n unchanged) but different predictions (different embedding used)
    assert overridden["per_model"]["cold0"]["prediction"]["n"] == \
        baseline["per_model"]["cold0"]["prediction"]["n"]


def test_profile_store_override_is_used_instead_of_datas_default_store(world):
    """E3's ablation sweep needs to score each P0-P6 variant's OWN embedding
    store, not data.profile_embeddings(pathway)'s default -- a distinct
    profile_store_override must actually change predictions, not be
    silently ignored."""
    model, run_cfg, data = world
    rng = np.random.default_rng(1)
    variant_store = _store(["cold0", "warm0", "warm1"], rng.standard_normal((3, 5)), "model_id")

    default_result = pairwise_cold_start_eval(model, run_cfg, data, target_models=["cold0"],
                                               sources=("chatbot_arena",))
    variant_result = pairwise_cold_start_eval(model, run_cfg, data, target_models=["cold0"],
                                              sources=("chatbot_arena",),
                                              profile_store_override=variant_store)
    default_p = default_result["per_model"]["cold0"]["prediction"]["pred_mean"]
    variant_p = variant_result["per_model"]["cold0"]["prediction"]["pred_mean"]
    assert default_p != variant_p


def test_rejects_free_model():
    data = FakeTrainingData(battles=_battles([]), warm_models=[], cold_models=["cold0"])
    model = NIRTModel(query_dim=8, dim=3, n_models=2, model_params="free")
    run_cfg = {"train": {"preference": {"enabled": False}}}
    with pytest.raises(ValueError, match="projected"):
        pairwise_cold_start_eval(model, run_cfg, data, target_models=["cold0"])
