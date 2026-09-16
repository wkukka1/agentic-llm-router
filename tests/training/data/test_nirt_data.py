from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import FakeTrainingData
from helpers import make_store as _fake_store

from training.data import schemas
from router.data.nirt import NIRTDataset
from training.data.nirt import (
    NIRT_OBS_COLUMNS,
    build_nirt_observations,
    observations,
    write_nirt_observations,
)


def test_observation_table_schema_and_no_embeddings(toy_training_data):
    df = build_nirt_observations(toy_training_data.cfg, data=toy_training_data, models="all")
    assert list(df.columns) == NIRT_OBS_COLUMNS
    # lightweight: no vector columns
    assert not any(c.endswith("embedding") for c in df.columns)
    assert df["split"].notna().all()
    assert set(df["metric_type"]).issubset(schemas.MetricType.CORRECTNESS)


def test_observations_exclude_cold_models_by_default(toy_training_data):
    df = build_nirt_observations(toy_training_data.cfg, data=toy_training_data, models="warm")
    assert "llama-2-70b-chat" not in set(df["model_id"])
    df_all = build_nirt_observations(toy_training_data.cfg, data=toy_training_data, models="all")
    assert "llama-2-70b-chat" in set(df_all["model_id"])


def test_target_score_kind(toy_training_data):
    raw = build_nirt_observations(toy_training_data.cfg, data=toy_training_data, models="all", score_kind="raw")
    eff = build_nirt_observations(toy_training_data.cfg, data=toy_training_data, models="all", score_kind="effective")
    mc_raw = raw[(raw.metric_type == "mc_accuracy") & (raw.score_raw == 0.25)]
    mc_eff = eff[(eff.metric_type == "mc_accuracy") & (eff.score_raw == 0.25)]
    assert mc_raw["target"].iloc[0] == pytest.approx(0.25)
    assert mc_eff["target"].iloc[0] == pytest.approx(0.0)   # chance-corrected


# --------------------------------------------------------------------------- #
# XD-05 / XA-11: the one training-side "read or build" accessor              #
# --------------------------------------------------------------------------- #
def test_observations_raises_when_missing_and_not_building(isolated_training_data):
    with pytest.raises(FileNotFoundError, match="nirt_observations"):
        observations(isolated_training_data.cfg, build=False)


def test_observations_builds_in_memory_without_writing(isolated_training_data):
    df = observations(isolated_training_data.cfg, data=isolated_training_data, build=True)
    assert len(df) > 0
    path = isolated_training_data.cfg.path("processed") / "nirt_observations.parquet"
    assert not path.exists()   # build=True never writes to disk


def test_observations_reads_existing_file_over_building(isolated_training_data):
    built = build_nirt_observations(isolated_training_data.cfg, data=isolated_training_data,
                                    models="all")
    write_nirt_observations(built, isolated_training_data.cfg)

    df = observations(isolated_training_data.cfg, build=False)   # no data=, so a build would crash
    pd.testing.assert_frame_equal(df.reset_index(drop=True), built.reset_index(drop=True),
                                  check_dtype=False)   # parquet round-trip widens object -> string


def test_observations_build_kw_forces_rebuild_even_when_file_exists(isolated_training_data):
    built_warm = build_nirt_observations(isolated_training_data.cfg, data=isolated_training_data,
                                         models="warm")
    write_nirt_observations(built_warm, isolated_training_data.cfg)
    assert "llama-2-70b-chat" not in set(built_warm["model_id"])

    df_all = observations(isolated_training_data.cfg, data=isolated_training_data, models="all")
    assert "llama-2-70b-chat" in set(df_all["model_id"])   # cold model only surfaces via a rebuild


def test_dataset_joins_by_id_without_duplicating_vectors(toy_training_data):
    obs = build_nirt_observations(toy_training_data.cfg, data=toy_training_data, models="all")
    q_store = _fake_store(sorted(obs["query_id"].unique()), 16, "query_id")
    m_store = _fake_store(sorted(obs["model_id"].unique()), 8, "model_id", seed=1)
    ds = NIRTDataset(obs, q_store, m_store, return_ids=True)

    assert len(ds) == len(obs)
    assert ds.query_dim == 16 and ds.model_dim == 8
    item = ds[0]
    assert item["query_embedding"].shape == (16,)
    assert item["model_embedding"].shape == (8,)
    assert np.isfinite(item["target"])
    # the returned query vector equals the store row for that id (no copy drift)
    np.testing.assert_array_equal(
        item["query_embedding"], q_store.get(item["query_id"])
    )
    # dataset holds references to the shared matrices, not per-row copies
    assert ds._qmat is q_store.matrix
    assert ds._mmat is m_store.matrix


def test_dataset_drops_observations_without_embeddings(toy_training_data):
    obs = build_nirt_observations(toy_training_data.cfg, data=toy_training_data, models="all")
    q_store = _fake_store(sorted(obs["query_id"].unique())[:1], 4, "query_id")
    m_store = _fake_store(sorted(obs["model_id"].unique()), 4, "model_id")
    ds = NIRTDataset(obs, q_store, m_store)
    assert ds.dropped > 0
    assert len(ds) < len(obs)
    assert set(ds.obs["query_id"]) <= set(q_store._index)


def test_gather_matches_iteration(toy_training_data):
    obs = build_nirt_observations(toy_training_data.cfg, data=toy_training_data, models="all")
    q_store = _fake_store(sorted(obs["query_id"].unique()), 6, "query_id")
    m_store = _fake_store(sorted(obs["model_id"].unique()), 5, "model_id")
    ds = NIRTDataset(obs, q_store, m_store)
    g = ds.gather()
    assert g["query_embedding"].shape == (len(ds), 6)
    np.testing.assert_array_equal(g["query_embedding"][3], ds[3]["query_embedding"])
    np.testing.assert_array_equal(g["target"], ds.targets)


def test_feature_store_default_none_unchanged(toy_training_data):
    """feature_store=None (the default) must not change query_dim / shapes --
    back-compat for every existing checkpoint."""
    obs = build_nirt_observations(toy_training_data.cfg, data=toy_training_data, models="all")
    q_store = _fake_store(sorted(obs["query_id"].unique()), 6, "query_id")
    m_store = _fake_store(sorted(obs["model_id"].unique()), 5, "model_id")
    ds = NIRTDataset(obs, q_store, m_store)
    assert ds.query_dim == 6
    assert ds.gather()["query_embedding"].shape == (len(ds), 6)


def test_feature_store_concatenated(toy_training_data):
    obs = build_nirt_observations(toy_training_data.cfg, data=toy_training_data, models="all")
    qids = sorted(obs["query_id"].unique())
    q_store = _fake_store(qids, 6, "query_id")
    m_store = _fake_store(sorted(obs["model_id"].unique()), 5, "model_id")
    feat_store = _fake_store(qids, 3, "query_id", seed=7)
    ds = NIRTDataset(obs, q_store, m_store, feature_store=feat_store, return_ids=True)

    assert ds.query_dim == 9   # 6 embedding + 3 feature dims
    item = ds[0]
    assert item["query_embedding"].shape == (9,)
    np.testing.assert_array_equal(item["query_embedding"][:6], q_store.get(item["query_id"]))
    np.testing.assert_array_equal(item["query_embedding"][6:], feat_store.get(item["query_id"]))

    g = ds.gather()
    assert g["query_embedding"].shape == (len(ds), 9)
    np.testing.assert_array_equal(g["query_embedding"][0], item["query_embedding"])
    # a subset gather() must index the feature columns consistently with the base ones
    sub = ds.gather(index=[2, 0])
    np.testing.assert_array_equal(sub["query_embedding"][1], item["query_embedding"])


def test_feature_store_missing_query_falls_back_to_zero(toy_training_data):
    obs = build_nirt_observations(toy_training_data.cfg, data=toy_training_data, models="all")
    qids = sorted(obs["query_id"].unique())
    q_store = _fake_store(qids, 4, "query_id")
    m_store = _fake_store(sorted(obs["model_id"].unique()), 4, "model_id")
    # feature store covers none of these queries -> every row should fall back to zeros
    feat_store = _fake_store(["some-other-query"], 2, "query_id")
    ds = NIRTDataset(obs, q_store, m_store, feature_store=feat_store)
    g = ds.gather()
    np.testing.assert_array_equal(g["query_embedding"][:, 4:], np.zeros((len(ds), 2), np.float32))


def test_from_config_query_features_wiring(toy_training_data):
    obs = build_nirt_observations(toy_training_data.cfg, data=toy_training_data, models="all")
    qids = sorted(obs["query_id"].unique())
    q_store = _fake_store(qids, 6, "query_id")
    m_store = _fake_store(sorted(obs["model_id"].unique()), 5, "model_id")
    feat_store = _fake_store(qids, 3, "query_id", seed=9)
    stub = FakeTrainingData(query_store=q_store, profile_store=m_store, feature_store=feat_store)
    ds = NIRTDataset.from_config(toy_training_data.cfg, data=stub, observations=obs, query_features="default")
    assert ds.query_dim == 9


def test_from_config_missing_query_features_raises(toy_training_data):
    obs = build_nirt_observations(toy_training_data.cfg, data=toy_training_data, models="all")
    qids = sorted(obs["query_id"].unique())
    q_store = _fake_store(qids, 6, "query_id")
    m_store = _fake_store(sorted(obs["model_id"].unique()), 5, "model_id")
    stub = FakeTrainingData(query_store=q_store, profile_store=m_store)
    with pytest.raises(FileNotFoundError, match="query features"):
        NIRTDataset.from_config(toy_training_data.cfg, data=stub, observations=obs, query_features="missing")


def test_from_config_default_query_features_none(toy_training_data):
    """query_features=None (the default) never touches data.query_features."""
    obs = build_nirt_observations(toy_training_data.cfg, data=toy_training_data, models="all")
    qids = sorted(obs["query_id"].unique())
    q_store = _fake_store(qids, 6, "query_id")
    m_store = _fake_store(sorted(obs["model_id"].unique()), 5, "model_id")
    stub = FakeTrainingData(query_store=q_store, profile_store=m_store)  # raises if from_config asks for features
    ds = NIRTDataset.from_config(toy_training_data.cfg, data=stub, observations=obs)
    assert ds.query_dim == 6


def test_collate_produces_tensors(toy_training_data):
    torch = pytest.importorskip("torch")
    obs = build_nirt_observations(toy_training_data.cfg, data=toy_training_data, models="all")
    q_store = _fake_store(sorted(obs["query_id"].unique()), 6, "query_id")
    m_store = _fake_store(sorted(obs["model_id"].unique()), 5, "model_id")
    ds = NIRTDataset(obs, q_store, m_store)
    batch = ds.collate([ds[0], ds[1]])
    assert batch["query_embedding"].shape == (2, 6)
    assert batch["target"].shape == (2,)
    assert batch["query_embedding"].dtype == torch.float32
