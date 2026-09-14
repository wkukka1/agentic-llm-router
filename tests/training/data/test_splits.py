from __future__ import annotations

import pandas as pd
import pytest

from router.config import load_config
from training.data.normalize import content_hash
from training.data.splits import check_leakage, make_splits


@pytest.fixture
def responses_for_split():
    rows = []
    for i in range(300):
        q = f"question number {i}"
        for m in ("gpt-4-1106-preview", "llama-2-70b-chat", "yi-34b-chat"):
            rows.append(
                dict(query_id=f"routerbench:t:{content_hash(q)[:16]}", model_id=m,
                     query=q, score=0.5, metric_type="accuracy", dataset="t",
                     split="test", source="routerbench", content_hash=content_hash(q))
            )
    # a semantically identical query from a different dataset -> must not leak
    q0 = "question number 0"
    rows.append(dict(query_id=f"lm_eval_harness:t2:{content_hash(q0)[:16]}",
                     model_id="gpt-4-1106-preview", query=q0, score=1.0,
                     metric_type="acc", dataset="t2", split=None,
                     source="lm_eval_harness", content_hash=content_hash(q0)))
    return pd.DataFrame(rows)


def test_splits_are_deterministic(responses_for_split, cfg):
    a = make_splits(responses_for_split, cfg)
    b = make_splits(responses_for_split, cfg)
    assert a["train"]["query_ids"] == b["train"]["query_ids"]
    assert a["test"]["query_ids"] == b["test"]["query_ids"]


def test_no_query_leakage(responses_for_split, cfg):
    splits = make_splits(responses_for_split, cfg)
    assert check_leakage(splits) == []


def test_content_identical_queries_share_split(responses_for_split, cfg):
    splits = make_splits(responses_for_split, cfg)
    h = content_hash("question number 0")[:16]
    rb_id = f"routerbench:t:{h}"
    lm_id = f"lm_eval_harness:t2:{h}"
    for name in ("train", "validation", "test"):
        ids = set(splits[name]["query_ids"])
        assert (rb_id in ids) == (lm_id in ids)


def test_split_fractions_roughly_respected(responses_for_split, cfg):
    splits = make_splits(responses_for_split, cfg)
    n = sum(len(splits[s]["query_ids"]) for s in ("train", "validation", "test"))
    assert 0.7 < len(splits["train"]["query_ids"]) / n < 0.9


# --------------------------------------------------------------------------- #
# force_warm / force_cold overrides (pool-expansion workstream)               #
# --------------------------------------------------------------------------- #
@pytest.fixture
def multi_model_responses():
    rows = []
    for i in range(120):
        q = f"q {i}"
        for m in ("m_a", "m_b", "m_c", "m_d", "m_e", "m_f"):
            rows.append(dict(query_id=f"routerbench:t:{content_hash(q)[:16]}", model_id=m,
                             query=q, score=0.5, metric_type="accuracy", dataset="t",
                             split=None, source="routerbench", content_hash=content_hash(q)))
    return pd.DataFrame(rows)


def _cfg_with(**split_over):
    cfg = load_config()
    s = cfg.split.to_dict()
    s.update(dict(cold_start_min_observations=0, cold_start_model_fraction=0.34))
    s.update(split_over)
    return cfg


def test_force_warm_and_force_cold_honoured(multi_model_responses):
    sp = make_splits(multi_model_responses,
                     _cfg_with(force_warm=["m_a"], force_cold=["m_f"]))
    cold = set(sp["cold_start_models"]["model_ids"])
    warm = set(sp["cold_start_models"]["warm_model_ids"])
    assert "m_a" in warm and "m_a" not in cold
    assert "m_f" in cold and "m_f" not in warm
    assert sp["train"]["params"]["force_warm"] == ["m_a"]
    assert sp["train"]["params"]["force_cold"] == ["m_f"]


def test_forced_warm_id_never_in_cold_lottery(multi_model_responses):
    # force m_a warm even though it would otherwise be lottery-eligible
    forced = make_splits(multi_model_responses, _cfg_with(force_warm=["m_a"]))
    assert "m_a" not in forced["cold_start_models"]["model_ids"]
    assert check_leakage(forced) == []


def test_force_cold_overrides_min_observations(multi_model_responses):
    # m_f has plenty of rows here; force it cold with a high threshold anyway
    sp = make_splits(multi_model_responses,
                     _cfg_with(cold_start_min_observations=10_000, force_cold=["m_f"]))
    assert sp["cold_start_models"]["model_ids"] == ["m_f"]


def test_force_warm_and_force_cold_conflict_raises(multi_model_responses):
    with pytest.raises(ValueError):
        make_splits(multi_model_responses, _cfg_with(force_warm=["m_a"], force_cold=["m_a"]))
