"""Repro: Orchestrator.run drops its spawn-time budget hold before re-reserving. Run from repo root."""
import sys
sys.path.insert(0, "src")
from router.execution.budget import BudgetLedger, ExecutionLimits
from router.execution.orchestrator import Orchestrator, OrchestratorConfig
from router.execution.plan import Plan, PlanStep
from router.execution.results import ExecutionResult
from router.execution.task import AgentTask

class O(Orchestrator):
    def plan(self, task):    return Plan("p", task.task_id, steps=[PlanStep("s", "x", None, estimated_cost=5.0)])
    def execute(self, plan): return ExecutionResult(actual_cost=5.0)
    def replan(self, plan):  return plan

ledger = BudgetLedger("root", 10.0)
orch = O(OrchestratorConfig("o", "0", llm=None, ledger=ledger, limits=ExecutionLimits(max_cost=10.0)))
orch.task = AgentTask("root", None, "root", "parent prompt")
child = AgentTask("child", "root", "root", "child prompt", depth=1, allocated_budget=6.0)
child_orch = orch.spawn(child)
print("after spawn: child holds 6.0, remaining =", ledger.remaining())

real_release = ledger.release
def release_then_sibling_reserves(task_id):
    real_release(task_id)
    if task_id == "child":       # the window between release() and reserve() in run()
        print("  window: sibling reserve(8.0) ->", ledger.reserve("sibling", 8.0))
ledger.release = release_then_sibling_reserves

res = child_orch.run(child)
print("child needs 5.0 of its 6.0 allocation ->", res.error_code, "| status:", child.status.value)
