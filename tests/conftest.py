from __future__ import annotations

import pandas as pd
import pytest

from router.config import load_config
from router.data import schemas


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def toy_responses() -> pd.DataFrame:
    """A tiny hand-built canonical response frame covering the edge cases the
    quality / correction / split code must handle."""
    rows = [
        # free-form accuracy, both models, one query
        dict(query_id="routerbench:gsm8k:aaaa", model_id="gpt-4-1106-preview",
             query="What is 2+2?", score=1.0, metric_type=schemas.MetricType.ACCURACY,
             dataset="gsm8k", split="test", source=schemas.Source.ROUTERBENCH,
             is_multiple_choice=False, n_choices=None, metadata={}),
        dict(query_id="routerbench:gsm8k:aaaa", model_id="llama-2-70b-chat",
             query="What is 2+2?", score=0.0, metric_type=schemas.MetricType.ACCURACY,
             dataset="gsm8k", split="test", source=schemas.Source.ROUTERBENCH,
             is_multiple_choice=False, n_choices=None, metadata={}),
        # 4-choice MC
        dict(query_id="routerbench:mmlu:bbbb", model_id="gpt-4-1106-preview",
             query="Pick A/B/C/D", score=1.0, metric_type=schemas.MetricType.MC_ACCURACY,
             dataset="mmlu-x", split="test", source=schemas.Source.ROUTERBENCH,
             is_multiple_choice=True, n_choices=4, metadata={}),
        dict(query_id="routerbench:mmlu:bbbb", model_id="llama-2-70b-chat",
             query="Pick A/B/C/D", score=0.25, metric_type=schemas.MetricType.MC_ACCURACY,
             dataset="mmlu-x", split="test", source=schemas.Source.ROUTERBENCH,
             is_multiple_choice=True, n_choices=4, metadata={}),
        # MC with unknown n_choices -> must be left uncorrected + warn
        dict(query_id="routerbench:mystery:cccc", model_id="gpt-4-1106-preview",
             query="mystery mc", score=0.5, metric_type=schemas.MetricType.MC_ACCURACY,
             dataset="mystery", split=None, source=schemas.Source.ROUTERBENCH,
             is_multiple_choice=True, n_choices=None, metadata={}),
        # arena pair
        dict(query_id="chatbot_arena:a:dddd", model_id="gpt-4-1106-preview",
             query="opinion?", score=1.0, metric_type=schemas.MetricType.ARENA_PREFERENCE,
             dataset="a", split=None, source=schemas.Source.ARENA,
             is_multiple_choice=False, n_choices=None,
             metadata={"pair_id": "p1", "role": "a", "opponent_model_id": "llama-2-70b-chat"}),
        dict(query_id="chatbot_arena:a:dddd", model_id="llama-2-70b-chat",
             query="opinion?", score=0.0, metric_type=schemas.MetricType.ARENA_PREFERENCE,
             dataset="a", split=None, source=schemas.Source.ARENA,
             is_multiple_choice=False, n_choices=None,
             metadata={"pair_id": "p1", "role": "b", "opponent_model_id": "gpt-4-1106-preview"}),
    ]
    return schemas.coerce_response_frame(pd.DataFrame(rows))
