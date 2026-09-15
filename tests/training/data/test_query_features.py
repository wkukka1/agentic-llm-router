from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from router.config import Config
from training.data.query_features import (
    build_query_features,
    family_names,
    feature_names,
    load_query_features,
)


@pytest.fixture
def cfg(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    queries = pd.DataFrame({
        "query_id": ["q:gsm8k:1", "q:gsm8k:1:5shot", "q:mbpp:1", "q:other:1"],
        "query": ["what is 2+2 in this math problem", "what is 2+2 in this math problem",
                  "def add(a, b): pass", "hello"],
        "dataset": ["gsm8k", "gsm8k", "mbpp", "some-other-dataset"],
    })
    queries.to_parquet(proc / "queries.parquet", index=False)
    responses = pd.DataFrame({
        "query_id": ["q:gsm8k:1", "q:mbpp:1"],
        "n_choices": pd.array([None, None], dtype="Int64"),
    })
    responses.to_parquet(proc / "responses.parquet", index=False)
    data = {
        "paths": {"processed": "processed"},
        "embedding": {"cache_dir": "processed/embeddings"},
        "profiles": {"task_families": {"math": ["gsm8k"], "code": ["mbpp", "humaneval"]}},
    }
    return Config(data, root=tmp_path)


def test_feature_dim_and_names(cfg):
    fams = family_names(cfg)
    assert fams == ["code", "math"]
    assert feature_names(cfg) == ["code", "math", "is_5shot", "prompt_len", "prompt_words", "n_choices"]


def test_shot_indicator_and_family_onehot(cfg):
    store = build_query_features(cfg, save=False)
    assert store.dim == len(feature_names(cfg))
    row = {qid: store.get(qid) for qid in store.ids}

    fams = family_names(cfg)
    math_i, code_i, shot_i = fams.index("math"), fams.index("code"), len(fams)

    assert row["q:gsm8k:1"][math_i] == 1.0 and row["q:gsm8k:1"][code_i] == 0.0
    assert row["q:gsm8k:1"][shot_i] == 0.0
    assert row["q:gsm8k:1:5shot"][shot_i] == 1.0
    # the 5shot twin has identical family/text-derived features to its 0shot pair
    # (same underlying question) -- only the shot indicator differs
    np.testing.assert_array_equal(
        np.delete(row["q:gsm8k:1"], shot_i), np.delete(row["q:gsm8k:1:5shot"], shot_i)
    )
    assert row["q:mbpp:1"][code_i] == 1.0 and row["q:mbpp:1"][math_i] == 0.0
    # unmapped dataset -> all-zero family one-hot, no crash
    assert row["q:other:1"][:len(fams)].sum() == 0.0


def test_missing_n_choices_defaults_zero(cfg):
    store = build_query_features(cfg, save=False)
    n_choices_i = feature_names(cfg).index("n_choices")
    assert store.get("q:gsm8k:1")[n_choices_i] == 0.0


def test_query_ids_subset(cfg):
    store = build_query_features(cfg, query_ids=["q:gsm8k:1"], save=False)
    assert store.ids == ["q:gsm8k:1"]


def test_build_save_and_load_roundtrip(cfg):
    built = build_query_features(cfg, name="t1", save=True)
    loaded = load_query_features(cfg, name="t1")
    assert loaded is not None
    assert loaded.dim == built.dim and set(loaded.ids) == set(built.ids)
    np.testing.assert_array_almost_equal(
        loaded.get("q:gsm8k:1:5shot"), built.get("q:gsm8k:1:5shot")
    )


def test_load_missing_returns_none(cfg):
    assert load_query_features(cfg, name="never-built") is None
