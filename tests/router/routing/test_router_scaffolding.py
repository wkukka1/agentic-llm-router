"""Real logic added to router.routing.base.Router (filter_candidates,
UnsupportedCandidatePolicy) plus router.execution.budget.BudgetLedger --
the other pieces of the class-diagram scaffolding with actual behavior to
test, not just importability."""

from __future__ import annotations

import pandas as pd
import pytest

from router.execution.budget import BudgetLedger, ExecutionLimits
from router.execution.orchestrator import Orchestrator, OrchestratorConfig
from router.execution.task import AgentTask
from router.routing.base import Router, UnsupportedCandidatePolicy


class ConstRouter(Router):
    kind = "const"

    def predict_scores(self, query_ids):
        ids = [str(q) for q in query_ids]
        return pd.DataFrame([[0.5] * len(self._model_ids) for _ in ids],
                            index=ids, columns=self._model_ids)


class _Profile:
    def __init__(self, model_id, capabilities=()):
        self.model_id = model_id
        self.capabilities = list(capabilities)


def test_filter_candidates_drops_unsupported_by_default():
    r = ConstRouter(["a", "b"])
    kept = r.filter_candidates([_Profile("a"), _Profile("c")], constraints=_NoConstraints())
    assert [c.model_id for c in kept] == ["a"]


def test_filter_candidates_error_policy_raises():
    r = ConstRouter(["a", "b"], unsupported_policy=UnsupportedCandidatePolicy.ERROR)
    with pytest.raises(ValueError, match="not a supported candidate"):
        r.filter_candidates([_Profile("c")], constraints=_NoConstraints())


def test_filter_candidates_score_with_prior_keeps_unsupported():
    r = ConstRouter(["a", "b"], unsupported_policy=UnsupportedCandidatePolicy.SCORE_WITH_PRIOR)
    kept = r.filter_candidates([_Profile("a"), _Profile("c")], constraints=_NoConstraints())
    assert {c.model_id for c in kept} == {"a", "c"}


def test_filter_candidates_requires_capability():
    r = ConstRouter(["a", "b"])
    kept = r.filter_candidates(
        [_Profile("a", capabilities=["vision"]), _Profile("b")],
        constraints=_Constraints(required_capabilities=["vision"]),
    )
    assert [c.model_id for c in kept] == ["a"]


def test_save_load_are_unimplemented_by_default():
    r = ConstRouter(["a", "b"])
    with pytest.raises(NotImplementedError):
        r.save("/tmp/whatever")
    with pytest.raises(NotImplementedError):
        r.load(object())


class _NoConstraints:
    required_capabilities: list[str] = []


class _Constraints:
    def __init__(self, required_capabilities):
        self.required_capabilities = required_capabilities


# --------------------------------------------------------------------------- #
# BudgetLedger                                                                #
# --------------------------------------------------------------------------- #
def test_budget_ledger_reserve_commit_release():
    ledger = BudgetLedger("root", total=10.0)
    assert ledger.remaining() == 10.0

    assert ledger.reserve("t1", 4.0) is True
    assert ledger.reserved == 4.0
    assert ledger.remaining() == 6.0

    ledger.commit("t1", 3.5)
    assert ledger.spent == 3.5
    assert ledger.reserved == 0.0
    assert ledger.remaining() == 6.5

    assert ledger.reserve("t2", 2.0) is True
    ledger.release("t2")
    assert ledger.reserved == 0.0
    assert ledger.remaining() == 6.5


def test_budget_ledger_refuses_overcommitment():
    ledger = BudgetLedger("root", total=5.0)
    assert ledger.reserve("t1", 4.0) is True
    assert ledger.reserve("t2", 2.0) is False   # only 1.0 left
    assert ledger.reserved == 4.0


def test_execution_limits_defaults_allow_shallow_recursion():
    limits = ExecutionLimits()
    assert limits.max_depth == 2
    assert limits.max_fanout > 0


class _ConstOrchestrator(Orchestrator):
    """Minimal concrete Orchestrator -- only can_spawn is under test here."""

    def plan(self, task):
        raise NotImplementedError

    def execute(self, plan):
        raise NotImplementedError

    def replan(self, plan):
        raise NotImplementedError


def _orchestrator(max_depth=2):
    config = OrchestratorConfig(
        name="o", version="0", llm=None,
        ledger=BudgetLedger("root", total=100.0),
        limits=ExecutionLimits(max_depth=max_depth),
    )
    return _ConstOrchestrator(config)


def test_can_spawn_allows_child_at_max_depth_not_beyond():
    """XA-08: max_depth means "the depth of the deepest node allowed to
    exist" here, the same convention AgenticRouter._solve uses (nodes exist
    through depth == max_depth; recursion just stops there) -- not "the
    deepest depth a child may be spawned at" (which would cap real nodes one
    level shallower than the live agentic path allows)."""
    orch = _orchestrator(max_depth=2)

    orch.task = AgentTask(task_id="parent1", parent_task_id=None, root_task_id="root",
                          prompt="p", depth=1)
    at_max = AgentTask(task_id="c1", parent_task_id=None, root_task_id="root",
                       prompt="p", depth=2)
    assert orch.can_spawn(at_max) is True

    orch.task = AgentTask(task_id="parent2", parent_task_id=None, root_task_id="root",
                          prompt="p", depth=2)
    past_max = AgentTask(task_id="c2", parent_task_id=None, root_task_id="root",
                         prompt="p", depth=3)
    assert orch.can_spawn(past_max) is False
