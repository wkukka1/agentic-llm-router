"""LLM infrastructure: profiles, a registry, a client interface, and provider
adapters. Scaffolding for the production router design -- see
``docs/architecture.md``.

Distinct from :mod:`router.agentic.llm_clients` (``LLMClient``/``LLMResponse``/
``ClientRegistry``), which is the simpler, already-working client layer the
live agentic flow actually dispatches through today (``invoke(prompt)``, no
``complete``/``stream`` split, no ``ProviderAdapter`` indirection). The two
are not yet unified -- see each module's docstring for the mapping.

Also distinct from :mod:`training.models.profiles`, which builds *text
descriptions* of models for the NIRT cold-start embedding, not the
registry/cost/capability records here.
"""
