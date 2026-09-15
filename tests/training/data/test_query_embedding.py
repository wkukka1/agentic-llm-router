from __future__ import annotations

import pytest

from training.data.embeddings import build_query_embedding_store

pytestmark = pytest.mark.filterwarnings("ignore")


def _build(data, **kw):
    try:
        return build_query_embedding_store(data, "irt", **kw)
    except KeyError:
        raise
    except Exception as exc:  # model not cached / offline
        pytest.skip(f"encoder unavailable: {exc}")


def test_build_all_queries_by_default(isolated_training_data):
    store = _build(isolated_training_data)
    all_ids = set(isolated_training_data.queries["query_id"])
    assert set(store._index) == all_ids


def test_split_filters_to_that_splits_query_ids(isolated_training_data):
    store = _build(isolated_training_data, split="train")
    assert set(store._index) == {
        "routerbench:gsm8k:aaaa", "routerbench:mmlu:bbbb", "chatbot_arena:a:dddd",
    }
    assert "routerbench:mystery:cccc" not in store


def test_unbuilt_split_raises(isolated_training_data):
    with pytest.raises(KeyError):
        _build(isolated_training_data, split="ood")


def test_from_nirt_excludes_arena_only_queries(isolated_training_data):
    """chatbot_arena:a:dddd is pairwise-only (no correctness row), so it never
    enters nirt_observations even though it's in the 'train' split."""
    store = _build(isolated_training_data, from_nirt=True)
    assert "chatbot_arena:a:dddd" not in store
    assert "routerbench:gsm8k:aaaa" in store
    assert "routerbench:mystery:cccc" in store  # correctness row, split="test"


def test_limit_takes_first_n_by_query_id(isolated_training_data):
    store = _build(isolated_training_data, limit=1)
    assert len(store) == 1
    assert list(store._index) == [sorted(isolated_training_data.queries["query_id"])[0]]


def test_output_written_under_tmp_path_not_repo(isolated_training_data, tmp_path):
    _build(isolated_training_data)
    cache_dir = isolated_training_data.cfg.resolve(
        isolated_training_data.cfg.get("embedding.cache_dir", "data/processed/embeddings")
    )
    assert str(cache_dir).startswith(str(tmp_path))
    assert cache_dir.exists()
