"""Repros: cost semantics on the serving side. Run from repo root."""
import sys, warnings
sys.path.insert(0, "src")
import pandas as pd
from router.routing.routers import MatrixRouter
from router.agentic.router import AgenticRouter
from router.agentic.llm_clients import ClientRegistry
from router.llm.client import LLMClient
from router.llm.profile import LLMProfile
from router.llm.adapters.fakes import CallableAdapter
from router.execution.budget import BudgetLedger

print("== AGENTICRESULT-001: a call whose cost is unknown is reported as cost 0.0 ==")
scores = pd.DataFrame([[0.9]], index=["q"], columns=["m"])
reg = ClientRegistry({"m": LLMClient(CallableAdapter("m", lambda p, **k: "an answer"))})   # no profile, no price
res = AgenticRouter(MatrixRouter(scores), reg, query_resolver=lambda p: "q").run("hello")
print("SubCall.cost =", res.steps[0].cost, "| AgenticResult.cost =", res.cost)

print("== LLMCLIENT-001: a profile with no price set prices every call at 0.0 ==")
client = LLMClient(CallableAdapter("m", lambda p, **k: "x"), LLMProfile(name="m", provider="p", model_id="m"))
print("price(1000, 1000) =", client.price(1000, 1000), " complete().cost =", client.complete("hi").cost)

print("== BUDGETLEDGER-001: commit can push spent past total ==")
led = BudgetLedger("root", 10.0)
led.reserve("t", 4.0); led.commit("t", 15.0)
print(f"total={led.total} spent={led.spent} remaining={led.remaining()}")
