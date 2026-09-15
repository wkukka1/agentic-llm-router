"""Tool-calling interface for an orchestrator: :class:`~router.tools.base.Tool`,
:class:`~router.tools.mcp.MCPTool`, :class:`~router.tools.router_tool.RouterTool`.

Distinct from :func:`router.agentic.orchestrator.build_router_tools`, which
already builds real LangChain ``StructuredTool``s (``route_query``,
``answer_with_model``, ``route_and_answer``) for
:class:`~router.agentic.orchestrator.LangChainToolOrchestrator` -- that's
working code, LangChain-specific. This package is the framework-agnostic
``Tool`` interface from the production design, not yet used by anything."""
