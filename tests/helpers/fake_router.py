"""A constant-score text-capable ``RouterModel`` for pipeline tests."""

from __future__ import annotations

import pandas as pd

from router.routing import RouterModel

POOL = ["fast-small", "big-strong", "coder"]


class FakeTextRouter(RouterModel):
    """Constant per-model scores, 'big-strong' always best -- enough to prove
    Router actually calls through to route_text and interprets the
    result, not a stand-in for real routing quality."""

    kind = "fake_text"
    can_route_text = True

    def __init__(self, model_ids=POOL):
        super().__init__(model_ids, name="fake")

    def predict_scores(self, query_ids):
        ids = [str(q) for q in query_ids]
        return pd.DataFrame([[0.5] * len(self._model_ids) for _ in ids],
                            index=ids, columns=self._model_ids)

    def predict_scores_text(self, texts, *, encoder=None):
        row = {"fast-small": 0.4, "big-strong": 0.9, "coder": 0.6}
        return pd.DataFrame([row for _ in texts],
                            index=list(range(len(texts))), columns=self._model_ids)
