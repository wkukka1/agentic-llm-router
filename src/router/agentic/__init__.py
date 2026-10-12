"""Agentic routing: a trained :class:`~router.routing.base.RouterModel` wrapped in an
LLM orchestrator.

``AgenticRouter`` takes a live prompt, routes it, then **triages** on the
router's own predicted-quality signal:

* the best model can handle it -> call that model's client and return the answer;
* it is compound / the best model is predicted to struggle -> hand it to the
  plan orchestrator, which decomposes the request via an LLM, routes and runs
  each sub-task (the router is exposed to it as a tool), judges the result,
  repairs/replans on a failed check, then synthesizes one answer.

```python
from router.routing import NIRTRouter
from router.agentic import AgenticRouter, ClientRegistry

ar = AgenticRouter(NIRTRouter.from_run("nirt-2d-projected"))            # echo clients
print(ar.run("What is 2+2?").summary())                                # single
print(ar.run("Summarise transformers, then implement attention in NumPy "
             "and analyse its complexity.").summary())                 # orchestrated

# real models, routed through the plan orchestrator (decompose -> execute -> judge)
from router.execution.routed import plan_orchestrator
ar = AgenticRouter(NIRTRouter.from_run("nirt-2d-projected"),
                   clients=ClientRegistry.langchain(["gpt-4o", "claude-3-5-sonnet-latest"]),
                   orchestrator=plan_orchestrator(budget=0.05))
```

See ``docs/agentic_router.md``. The defaults (echo clients, the plan
orchestrator with ``HeuristicTriage``/``NaiveDecomposer``) need nothing beyond
the core install for a request simple enough that triage answers it directly;
an orchestrated request calls through to the LLM configured as ``plan.model``
(:meth:`AgenticRouter.from_settings`), which needs LangChain (the ``agentic``
extra). Clients are :class:`~router.llm.client.LLMClient`\\ s wrapping a
:class:`~router.llm.adapters.base.ProviderAdapter` (``EchoAdapter``/
``CallableAdapter``/``LangChainAdapter`` in :mod:`router.llm.adapters`) --
build one directly and pass it to :meth:`ClientRegistry.register` for a
custom client.
"""

from __future__ import annotations

from ..execution.routed import PlanOrchestratorBridge, plan_orchestrator
from .llm_clients import (
    ClientRegistry,
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
    # orchestrator
    "plan_orchestrator",
    "PlanOrchestratorBridge",
]
