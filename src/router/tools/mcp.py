"""An MCP-backed tool. No MCP client integration exists in this codebase yet
(:func:`router.agentic.orchestrator.build_router_tools` builds LangChain tools
directly, not via MCP) -- this is a structural placeholder for wiring one in."""

from __future__ import annotations

from typing import Any

from .base import Tool


class MCPTool(Tool):
    def __init__(self, name: str, server: str, *, version: str = "0"):
        self.name = name
        self.version = version
        self.server = server

    def execute(self, input: Any) -> Any:
        raise NotImplementedError(
            f"MCPTool({self.name!r}) has no MCP client wired up -- this is scaffolding."
        )
