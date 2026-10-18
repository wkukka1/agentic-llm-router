"""Run state belongs to a run: a public helper used outside run() must not leave any behind."""

from __future__ import annotations

import pandas as pd

from router.agentic import AgenticRouter, ClientRegistry, LLMTriage
from router.agentic.router import _current_run
from router.routing import RouterModel

POOL = ["small", "big"]


class _Router(RouterModel):
    kind = "fake"
    can_route_text = True

    def __init__(self):
        super().__init__(POOL, name="fake")

    def predict_scores(self, query_ids):
        return pd.DataFrame([[0.5, 0.5] for _ in query_ids], index=[str(q) for q in query_ids], columns=POOL)

    def predict_scores_text(self, texts, *, encoder=None):
        return pd.DataFrame([[0.9, 0.9] for _ in texts], index=list(range(len(texts))), columns=POOL)


class _Chat:
    def invoke(self, _prompt):
        return "SINGLE"


def test_standalone_calls_outside_run_do_not_share_a_budget_or_leave_state_behind():
    agent = AgenticRouter(_Router(), ClientRegistry.echo(POOL), triage=LLMTriage(_Chat()), max_calls=1)
    agent.triage_prompt("What is the capital of France?")
    agent.triage_prompt("What is the capital of Spain?")  # a leaked state would refuse this one
    assert _current_run.get() is None
