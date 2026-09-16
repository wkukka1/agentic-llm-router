"""Agentic routing: a trained :class:`~router.routing.base.Router` wrapped in an
LLM orchestrator.

``AgenticRouter`` takes a live prompt, routes it, then **triages** on the
router's own predicted-quality signal:

* the best model can handle it -> call that model's client and return the answer;
* it is compound / the best model is predicted to struggle -> hand it to an
  orchestrator that decomposes the request and routes each sub-task recursively
  (the router is exposed to the orchestrator as a tool), then synthesizes.

```python
from router.routing import NIRTRouter
from router.agentic import AgenticRouter, ClientRegistry

ar = AgenticRouter(NIRTRouter.from_run("nirt-2d-projected"))            # echo clients
print(ar.run("What is 2+2?").summary())                                # single
print(ar.run("Summarise transformers, then implement attention in NumPy "
             "and analyse its complexity.").summary())                 # orchestrated

# real models + an LLM-driven orchestrator
from langchain.chat_models import init_chat_model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
ar = AgenticRouter(NIRTRouter.from_run("nirt-2d-projected"),
                   clients=ClientRegistry.langchain(["gpt-4o", "claude-3-5-sonnet-latest"]),
                   ).with_langchain_orchestrator(llm)
```

See ``docs/agentic_router.md``. LangChain is an optional dependency (the
``agentic`` extra); the defaults (echo clients, `RecursiveOrchestrator`,
`HeuristicTriage`, `NaiveDecomposer`) need nothing beyond the core install.
"""

from __future__ import annotations

from .llm_clients import (
    CallableClient,
    ClientRegistry,
    EchoClient,
    LangChainClient,
    LLMClient,
    LLMResponse,
    guess_provider,
)
from .decompose import (
    LLMDecomposer,
    LLMSynthesizer,
    NaiveDecomposer,
    concat_synthesizer,
)
from .orchestrator import (
    LangChainToolOrchestrator,
    RecursiveOrchestrator,
    build_router_tools,
)
from .result import AgenticResult, SubCall
from .router import AgenticRouter
from .triage import HeuristicTriage, LLMTriage, TriageDecision

__all__ = [
    "AgenticRouter",
    "AgenticResult",
    "SubCall",
    # LLM clients
    "ClientRegistry",
    "LLMResponse",
    "LLMClient",
    "EchoClient",
    "CallableClient",
    "LangChainClient",
    "guess_provider",
    # triage
    "TriageDecision",
    "HeuristicTriage",
    "LLMTriage",
    # decomposition
    "NaiveDecomposer",
    "LLMDecomposer",
    "concat_synthesizer",
    "LLMSynthesizer",
    # orchestrators
    "RecursiveOrchestrator",
    "LangChainToolOrchestrator",
    "build_router_tools",
]
