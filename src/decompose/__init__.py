"""Prompt understanding: classifiers + an embedder produce :class:`PromptSignals`
for a request. Scaffolding for the production router design -- see
``docs/architecture.md``. Import submodules directly (``decomposer``,
``signals``, ``conversation``, ``classifiers``, ``embedders``).

A standalone top-level package (no dependency on ``router``/``training``/
``evaluation``) -- ``router.router.RoutingPipeline`` imports from here, not
the other way around. Not to be confused with
:mod:`router.agentic.decompose` (``NaiveDecomposer``/``LLMDecomposer``),
a different, unrelated job: splitting one *compound request* into several
*sub-prompts* for the live orchestrator, versus this package's job of
extracting *signals* from one prompt.
"""
