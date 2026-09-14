"""IRT-Router benchmark loader + pre-partitioned split (arXiv 2506.01048)."""

from __future__ import annotations

import textwrap

import pandas as pd
import pytest

from router.config import Config
from training.data import schemas
from training.data.loaders import load_irt_router
from training.data.normalize import make_query_id
from training.data.response_matrix import build_tables
from training.data.splits import check_leakage, make_presplit

_COLS = "id,question,ground_truth,completion,input_tokens,output_tokens,cost,performance,task,llm"


def _csv(rows: list[str]) -> str:
    return _COLS + "\n" + "\n".join(rows) + "\n"


@pytest.fixture
def irt_config(tmp_path):
    root = tmp_path / "raw" / "irt_router"
    data = root / "data"
    data.mkdir(parents=True)
    # 2 tasks (mmlu = MC, gsm8k = free-form), 3 models -- one out of pool.
    (data / "train.csv").write_text(_csv([
        '1,"Q mmlu 1",A,,10,5,0.001,1,mmlu,gpt_4o',
        '1,"Q mmlu 1",A,,10,5,0.0001,0,mmlu,glm_4_flash',
        '1,"Q mmlu 1",A,,10,5,0.002,1,mmlu,claude35_haiku20241022',
        '2,"Q gsm8k 1",42,,20,30,0.003,0.5,gsm8k,gpt_4o',
        '2,"Q gsm8k 1",42,,20,30,0.0003,0.0,gsm8k,glm_4_flash',
        '3,"Q mmlu 2",C,,11,6,0.001,0,mmlu,gpt_4o',
        '3,"Q mmlu 2",C,,11,6,0.0001,1,mmlu,glm_4_flash',
    ]), encoding="utf-8")
    (data / "test1.csv").write_text(_csv([
        '1,"Q mmlu test",B,,10,5,0.001,1,mmlu,gpt_4o',
        '1,"Q mmlu test",B,,10,5,0.0001,0,mmlu,glm_4_flash',
    ]), encoding="utf-8")
    (data / "test2.csv").write_text(_csv([
        '1,"Q gsm8k ood",7,,10,5,0.001,1,gsm8k,gpt_4o',
        '1,"Q gsm8k ood",7,,10,5,0.0001,0,gsm8k,glm_4_flash',
    ]), encoding="utf-8")

    return Config({
        "seed": 42,
        "paths": {"processed": str(tmp_path / "proc"), "splits": str(tmp_path / "splits")},
        "sources": {"irt_router": {
            "local_dir": str(root),
            "pool": ["gpt_4o", "glm_4_flash"],
        }},
        "multiple_choice": {"by_task": {"mmlu": 4}, "by_prefix": {}},
        "chance_correction": {"method": "normalized", "clip": True},
        "split": {"presplit": True, "train_fraction": 0.7, "validation_fraction": 0.3},
    }, root=tmp_path)


def test_loader_emits_canonical_schema(irt_config):
    df = load_irt_router(irt_config)
    for col in schemas.RESPONSE_COLUMNS:
        assert col in df.columns
    assert set(df["source"].unique()) == {schemas.Source.IRT_ROUTER}
    # out-of-pool model dropped
    assert set(df["model_id"].unique()) == {"gpt_4o", "glm_4_flash"}
    # MC vs free-form
    mmlu = df[df["dataset"] == "mmlu"]
    gsm = df[df["dataset"] == "gsm8k"]
    assert (mmlu["metric_type"] == schemas.MetricType.MC_ACCURACY).all()
    assert (mmlu["n_choices"] == 4).all()
    assert (gsm["metric_type"] == schemas.MetricType.ACCURACY).all()
    assert gsm["n_choices"].isna().all()
    # cost passed through verbatim, never null
    assert df["cost"].notna().all()
    assert float(df[(df.model_id == "gpt_4o") & (df.dataset == "gsm8k")]["cost"].iloc[0]) == 0.003


def test_origin_recorded_for_presplit(irt_config):
    df = load_irt_router(irt_config)
    origins = {m.get("origin") for m in df["metadata"]}
    assert origins == {"train", "test1", "test2"}


def test_query_id_stable_across_files(irt_config):
    # same question text in train + test would share a query_id (content-addressed)
    qid = make_query_id(schemas.Source.IRT_ROUTER, "gsm8k", "Q gsm8k 1")
    df = load_irt_router(irt_config)
    assert qid in set(df["query_id"])


def test_make_presplit_partitions_by_origin(irt_config):
    tables = build_tables(irt_config, sources=[schemas.Source.IRT_ROUTER])
    splits = make_presplit(tables["responses"], irt_config)

    assert set(splits) >= {"train", "validation", "test", "ood", "cold_start_models"}
    assert not check_leakage(splits)
    assert splits["cold_start_models"]["model_ids"] == []

    resp = tables["responses"]
    qid_task = dict(zip(resp["query_id"], resp["dataset"]))
    # test1 questions -> test ; test2 questions -> ood
    test_q = make_query_id(schemas.Source.IRT_ROUTER, "mmlu", "Q mmlu test")
    ood_q = make_query_id(schemas.Source.IRT_ROUTER, "gsm8k", "Q gsm8k ood")
    assert test_q in splits["test"]["query_ids"]
    assert ood_q in splits["ood"]["query_ids"]
    assert ood_q not in splits["test"]["query_ids"]
    # every ood query is a test2 (gsm8k here) task, never a train-only task
    assert all(qid_task[q] == "gsm8k" for q in splits["ood"]["query_ids"])


def test_make_presplit_is_deterministic(irt_config):
    tables = build_tables(irt_config, sources=[schemas.Source.IRT_ROUTER])
    a = make_presplit(tables["responses"], irt_config)
    b = make_presplit(tables["responses"], irt_config)
    assert a["train"]["query_ids"] == b["train"]["query_ids"]
    assert a["validation"]["query_ids"] == b["validation"]["query_ids"]
