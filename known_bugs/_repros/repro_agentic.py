"""Repros for the serving side: router.agentic + router.llm. Run from repo root."""
import sys, types, warnings
sys.path.insert(0, "src")
import pandas as pd
from router.routing.routers import MatrixRouter
from router.agentic.router import AgenticRouter
from router.agentic.llm_clients import ClientRegistry
from router.agentic.decompose import NaiveDecomposer, LLMDecomposer
from router.llm.client import LLMClient, LLMClientFactory
from router.llm.profile import LLMProfile
from router.llm.adapters.fakes import EchoAdapter

print("== AGENTICROUTER-001: result reports pool-best quality next to the model that answered ==")
scores = pd.DataFrame([[0.90, 0.80]], index=["q"], columns=["big", "cheap"])
ar = AgenticRouter(MatrixRouter(scores, model_costs=[1.0, 0.0]), query_resolver=lambda p: "q", lam=0.5)
res = ar.run("hello")
print(f"selected_model_id={res.selected_model_id}  result.predicted_quality={res.predicted_quality}  "
      f"steps[0].predicted_quality={res.steps[0].predicted_quality}")
print("summary:", res.summary())

print("== CLIENTREGISTRY-001: echo-fallback warning depends on construction path ==")
def n_warn(reg):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always"); reg.get("unknown")
    return len(w)
real = LLMClient(EchoAdapter("a"))
r1 = ClientRegistry({"a": real})
r2 = ClientRegistry(); r2.register("a", real)
print("clients passed to constructor -> warnings:", n_warn(r1))
print("clients added via register()  -> warnings:", n_warn(r2))

print("== NAIVEDECOMPOSER-001: sub-tasks silently dropped ==")
d = NaiveDecomposer()
print(d("Fix bug. Add tests for the module. Update the docs."))
eight = "Do these:\n" + "\n".join(f"{i}. Handle item number {i} carefully" for i in range(1, 9))
print("8 numbered items ->", len(d(eight)), "sub-tasks")
print("== NAIVEDECOMPOSER-002: the 'then,' marker never matches when followed by a space ==")
print(d("Write the parser then, write the tests"))
print(d("Write the parser then,write the tests"))

print("== LLMDECOMPOSER-001: line-prefix stripping corrupts a decimal-number task ==")
print(LLMDecomposer(None)._parse("3.14 times 2 equals what?\n2. summarise the result"))

print("== LLMCLIENTFACTORY-001: cached client keeps the old profile after a re-price ==")
f = LLMClientFactory(adapters={"echo": EchoAdapter})
old = LLMProfile(name="m", provider="echo", model_id="m", output_cost_per_token=1e-6)
new = LLMProfile(name="m", provider="echo", model_id="m", output_cost_per_token=5e-6)
f.client_for(old); c2 = f.client_for(new)
print("new profile price/token: 5e-06 | client_for(new).price(0, 1000) =", c2.price(0, 1000), "(expected 0.005)")

print("== AGENTICORCHESTRATOR-001: handle_tool_error=True vs a non-ToolException ==")
try:
    from router.agentic.orchestrator import build_router_tools
    class FakeAgent:
        router = types.SimpleNamespace(model_ids=["a"])
        def route_decision(self, q): raise LookupError("no usable score for this prompt")
    tools = build_router_tools(FakeAgent(), 0, [])
    for t in tools[:2]:
        try:
            print(t.name, "->", repr(t.invoke({"query": "x"} if t.name == "route_query" else {"query": "x", "model_id": "zzz"})))
        except Exception as e:
            print(t.name, "-> RAISED", type(e).__name__, e)
except ImportError as e:
    print("langchain_core not installed:", e)
