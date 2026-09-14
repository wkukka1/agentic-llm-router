"""``ExecutionResult``: what running a step (or a whole task tree) actually
cost and produced -- the ``actual`` counterpart to
:class:`~router.execution.forecast.CostForecast`'s prediction.

The live agentic flow's closest equivalent is
:class:`router.agentic.result.SubCall` / ``AgenticResult`` (prompt, answer,
cost, children) -- informally shaped, no ``errorCode``/token counts. Nothing
constructs this dataclass yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    output: str = ""
    actual_cost: float = 0.0
    actual_latency: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = True
    error_code: Optional[str] = None
