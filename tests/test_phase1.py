from __future__ import annotations

import pandas as pd
import pytest

from router.config import load_config
from router.data import schemas
from router.data.phase1 import Phase1Data, load_phase1
from router.data.response_matrix import build_tables


@pytest.fixture
def toy_phase1(toy_responses):
    class _CC:
        def get(self, k, d=None):
            return {"chance_correction": {"method": "normalized", "clip": True,
                                          "warn_on_missing_choices": False}}.get(k, d)

    tables = build_tables(cfg=_CC(), responses=toy_responses)
    splits = {
        "train": {"query_ids": ["routerbench:gsm8k:aaaa", "routerbench:mmlu:bbbb",
                                 "chatbot_arena:a:dddd"]},
        "validation": {"query_ids": []},
        "test": {"query_ids": ["routerbench:mystery:cccc"]},
        "cold_start_models": {"model_ids": ["llama-2-70b-chat"]},
    }
    return Phase1Data(load_config(), tables["responses"], tables["queries"],
                      tables["models"], splits)


def test_model_scopes(toy_phase1):
    assert "llama-2-70b-chat" in toy_phase1.cold_start_model_ids()
    assert "llama-2-70b-chat" not in toy_phase1.warm_model_ids()
    assert "gpt-4-1106-preview" in toy_phase1.warm_model_ids()


def test_correctness_excludes_pairwise(toy_phase1):
    c = toy_phase1.correctness(models="all")
    assert set(c["metric_type"]).issubset(schemas.MetricType.CORRECTNESS)
    assert schemas.MetricType.ARENA_PREFERENCE not in set(c["metric_type"])


def test_correctness_score_kinds(toy_phase1):
    c = toy_phase1.correctness(models="all", score_kind="raw")
    mc = c[c["metric_type"] == schemas.MetricType.MC_ACCURACY]
    # raw keeps 0.25; effective would rescale it to 0.0
    assert (mc["score_raw"] == 0.25).any()
    eff = toy_phase1.correctness(models="all", score_kind="effective")
    mc_eff = eff[(eff["metric_type"] == "mc_accuracy") & (eff["score_raw"] == 0.25)]
    assert mc_eff["score"].iloc[0] == pytest.approx(0.0)


def test_correctness_matrix_shape_and_split(toy_phase1):
    R = toy_phase1.correctness_matrix(split="train",
                                      combine_metrics=["accuracy", "mc_accuracy"],
                                      models="all")
    assert "gpt-4-1106-preview" in R.columns
    assert "routerbench:gsm8k:aaaa" in R.index
    # test split query must be absent
    assert "routerbench:mystery:cccc" not in R.index


def test_pairwise_reconstructs_triples(toy_phase1):
    p = toy_phase1.pairwise(models="all")
    assert len(p) == 1
    row = p.iloc[0]
    assert {row["model_a"], row["model_b"]} == {"gpt-4-1106-preview", "llama-2-70b-chat"}
    assert row["winner"] in ("a", "b", "tie")
    assert row["score_a"] in (0.0, 0.5, 1.0)


def test_pairwise_warm_filter_drops_cold_opponent(toy_phase1):
    # llama-2-70b-chat is cold-start -> warm-scoped pairwise excludes the battle
    assert toy_phase1.pairwise(models="warm").empty
    assert len(toy_phase1.pairwise(models="all")) == 1


def test_n_choices_map(toy_phase1):
    nc = toy_phase1.n_choices()
    assert nc.get("routerbench:mmlu:bbbb") == 4


def test_summary_keys(toy_phase1):
    s = toy_phase1.summary()
    for k in ("observations", "correctness_observations", "pairwise_observations",
              "models_warm", "models_cold_start", "train_queries"):
        assert k in s


# --- real data (skips if not built) --------------------------------------- #
def test_load_phase1_real():
    cfg = load_config()
    try:
        d = load_phase1(cfg)
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
