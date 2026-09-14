from __future__ import annotations

import pytest

from router.config import load_config
from training.data import schemas
from training.data.facade import load_training_data


def test_model_scopes(toy_training_data):
    assert "llama-2-70b-chat" in toy_training_data.cold_start_model_ids()
    assert "llama-2-70b-chat" not in toy_training_data.warm_model_ids()
    assert "gpt-4-1106-preview" in toy_training_data.warm_model_ids()


def test_correctness_excludes_pairwise(toy_training_data):
    c = toy_training_data.correctness(models="all")
    assert set(c["metric_type"]).issubset(schemas.MetricType.CORRECTNESS)
    assert schemas.MetricType.ARENA_PREFERENCE not in set(c["metric_type"])


def test_correctness_score_kinds(toy_training_data):
    c = toy_training_data.correctness(models="all", score_kind="raw")
    mc = c[c["metric_type"] == schemas.MetricType.MC_ACCURACY]
    # raw keeps 0.25; effective would rescale it to 0.0
    assert (mc["score_raw"] == 0.25).any()
    eff = toy_training_data.correctness(models="all", score_kind="effective")
    mc_eff = eff[(eff["metric_type"] == "mc_accuracy") & (eff["score_raw"] == 0.25)]
    assert mc_eff["score"].iloc[0] == pytest.approx(0.0)


def test_correctness_matrix_shape_and_split(toy_training_data):
    R = toy_training_data.correctness_matrix(split="train",
                                      combine_metrics=["accuracy", "mc_accuracy"],
                                      models="all")
    assert "gpt-4-1106-preview" in R.columns
    assert "routerbench:gsm8k:aaaa" in R.index
    # test split query must be absent
    assert "routerbench:mystery:cccc" not in R.index


def test_pairwise_reconstructs_triples(toy_training_data):
    p = toy_training_data.pairwise(models="all")
    assert len(p) == 1
    row = p.iloc[0]
    assert {row["model_a"], row["model_b"]} == {"gpt-4-1106-preview", "llama-2-70b-chat"}
    assert row["winner"] in ("a", "b", "tie")
    assert row["score_a"] in (0.0, 0.5, 1.0)


def test_pairwise_warm_filter_drops_cold_opponent(toy_training_data):
    # llama-2-70b-chat is cold-start -> warm-scoped pairwise excludes the battle
    assert toy_training_data.pairwise(models="warm").empty
    assert len(toy_training_data.pairwise(models="all")) == 1


def test_n_choices_map(toy_training_data):
    nc = toy_training_data.n_choices()
    assert nc.get("routerbench:mmlu:bbbb") == 4


def test_summary_keys(toy_training_data):
    s = toy_training_data.summary()
    for k in ("observations", "correctness_observations", "pairwise_observations",
              "models_warm", "models_cold_start", "train_queries"):
        assert k in s


# --- real data (skips if not built) --------------------------------------- #
def test_load_training_data_real():
    cfg = load_config()
    try:
        d = load_training_data(cfg)
    except FileNotFoundError:
        pytest.skip("processed tables / splits not built")
    s = d.summary()
    assert s["observations"] > 0
    # no query leakage reachable through the facade
    tr = set(d.split_query_ids("train"))
    te = set(d.split_query_ids("test"))
    assert tr.isdisjoint(te)
    # correctness never contains a pairwise metric
    assert d.correctness(split="train").pipe(
        lambda x: x["metric_type"].isin(schemas.MetricType.CORRECTNESS).all()
    )
