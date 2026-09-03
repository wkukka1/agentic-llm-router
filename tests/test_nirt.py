from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from router.data import schemas
from router.data.nirt import NIRT_OBS_COLUMNS, NIRTDataset, build_nirt_observations
from router.data.phase1 import Phase1Data
from router.data.response_matrix import build_tables
from router.embeddings.encoder import EmbeddingStore


class _ChanceCfg:
    def get(self, k, d=None):
        return {"chance_correction": {"method": "normalized", "clip": True,
                                      "warn_on_missing_choices": False}}.get(k, d)


def _fake_store(ids, dim, id_field, seed=0):
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((len(ids), dim)).astype(np.float32)
    manifest = {"dim": dim, "id_field": id_field, "count": len(ids), "complete": True}
    return EmbeddingStore(list(ids), mat, manifest, id_field=id_field)


@pytest.fixture
def phase1(toy_responses):
    tables = build_tables(cfg=_ChanceCfg(), responses=toy_responses)
    splits = {
        "train": {"query_ids": ["routerbench:gsm8k:aaaa", "routerbench:mmlu:bbbb"]},
        "validation": {"query_ids": []},
        "test": {"query_ids": ["routerbench:mystery:cccc"]},
        "cold_start_models": {"model_ids": ["llama-2-70b-chat"]},
    }
    from router.config import load_config

    return Phase1Data(load_config(), tables["responses"], tables["queries"],
                      tables["models"], splits)


def test_observation_table_schema_and_no_embeddings(phase1):
    df = build_nirt_observations(phase1.cfg, data=phase1, models="all")
    assert list(df.columns) == NIRT_OBS_COLUMNS
    # lightweight: no vector columns
    assert not any(c.endswith("embedding") for c in df.columns)
    assert df["split"].notna().all()
    assert set(df["metric_type"]).issubset(schemas.MetricType.CORRECTNESS)


def test_observations_exclude_cold_models_by_default(phase1):
    df = build_nirt_observations(phase1.cfg, data=phase1, models="warm")
    assert "llama-2-70b-chat" not in set(df["model_id"])
    df_all = build_nirt_observations(phase1.cfg, data=phase1, models="all")
    assert "llama-2-70b-chat" in set(df_all["model_id"])


def test_target_score_kind(phase1):
    raw = build_nirt_observations(phase1.cfg, data=phase1, models="all", score_kind="raw")
    eff = build_nirt_observations(phase1.cfg, data=phase1, models="all", score_kind="effective")
    mc_raw = raw[(raw.metric_type == "mc_accuracy") & (raw.score_raw == 0.25)]
    mc_eff = eff[(eff.metric_type == "mc_accuracy") & (eff.score_raw == 0.25)]
    assert mc_raw["target"].iloc[0] == pytest.approx(0.25)
    assert mc_eff["target"].iloc[0] == pytest.approx(0.0)   # chance-corrected


def test_dataset_joins_by_id_without_duplicating_vectors(phase1):
    obs = build_nirt_observations(phase1.cfg, data=phase1, models="all")
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


def test_dataset_drops_observations_without_embeddings(phase1):
    obs = build_nirt_observations(phase1.cfg, data=phase1, models="all")
    q_store = _fake_store(sorted(obs["query_id"].unique())[:1], 4, "query_id")
    m_store = _fake_store(sorted(obs["model_id"].unique()), 4, "model_id")
    ds = NIRTDataset(obs, q_store, m_store)
    assert ds.dropped > 0
    assert len(ds) < len(obs)
    assert set(ds.obs["query_id"]) <= set(q_store._index)


def test_gather_matches_iteration(phase1):
    obs = build_nirt_observations(phase1.cfg, data=phase1, models="all")
    q_store = _fake_store(sorted(obs["query_id"].unique()), 6, "query_id")
    m_store = _fake_store(sorted(obs["model_id"].unique()), 5, "model_id")
    ds = NIRTDataset(obs, q_store, m_store)
    g = ds.gather()
    assert g["query_embedding"].shape == (len(ds), 6)
    np.testing.assert_array_equal(g["query_embedding"][3], ds[3]["query_embedding"])
    np.testing.assert_array_equal(g["target"], ds.targets)


def test_collate_produces_tensors(phase1):
    torch = pytest.importorskip("torch")
    obs = build_nirt_observations(phase1.cfg, data=phase1, models="all")
    q_store = _fake_store(sorted(obs["query_id"].unique()), 6, "query_id")
    m_store = _fake_store(sorted(obs["model_id"].unique()), 5, "model_id")
    ds = NIRTDataset(obs, q_store, m_store)
    batch = ds.collate([ds[0], ds[1]])
    assert batch["query_embedding"].shape == (2, 6)
    assert batch["target"].shape == (2,)
    assert batch["query_embedding"].dtype == torch.float32
