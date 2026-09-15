"""Orchestrators: how a request that triage flagged as *compound* gets solved.

Both implement ``run(prompt, agent, depth) -> (answer, list[SubCall])`` where
``agent`` is the :class:`~router.agentic.router.AgenticRouter` (called back for
per-sub-task routing, so decomposition is recursive).

* :class:`RecursiveOrchestrator` -- plain control flow: decompose, solve each
  sub-task through the agent, synthesize. Deterministic, no agent framework.
* :class:`LangChainToolOrchestrator` -- exposes the router + clients as
  LangChain tools and lets a tool-calling agent drive: it decides how to split
  the work and calls ``route_and_answer`` (which re-enters routing) itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .result import SubCall

if TYPE_CHECKING:
    from .router import AgenticRouter

__all__ = ["RecursiveOrchestrator", "LangChainToolOrchestrator", "build_router_tools"]

# AgentExecutor's output when max_iterations / max_execution_time is exhausted
_STOPPED_PREFIX = "Agent stopped due to"


# --------------------------------------------------------------------------- #
# plain recursive orchestrator                                                #
# --------------------------------------------------------------------------- #
class RecursiveOrchestrator:
    name = "recursive"

    def run(self, prompt: str, agent: "AgenticRouter", depth: int) -> tuple[str, list[SubCall]]:
        subs = agent.decomposer(prompt)
        if len(subs) <= 1:
            # reuses the route triage already computed for this prompt
            sc = agent._answer_directly(prompt, depth)
            return sc.answer, [sc]
        children = [agent._solve(s, depth + 1) for s in subs]
        answer = agent.synthesizer(prompt, [(c.prompt, c.answer) for c in children])
        return answer, children


# --------------------------------------------------------------------------- #
# LangChain tool-calling orchestrator                                         #
# --------------------------------------------------------------------------- #
def build_router_tools(agent: "AgenticRouter", depth: int, steps: list[SubCall]):
    """LangChain ``StructuredTool``s bound to ``agent``; every model call is
    appended to ``steps``. Returns ``[route_query, answer_with_model,
    route_and_answer]``. Tool exceptions are returned to the agent as
    observations (``handle_tool_error``) instead of aborting the executor."""
    from langchain_core.tools import StructuredTool

    def route_query(query: str) -> str:
        """Ask the trained router which model should answer QUERY. Returns the
        chosen model id, its predicted quality (0-1), and the runner-up."""
        rr = agent.route_decision(query)
        ranked = sorted(zip(rr.model_ids, rr.scores[0]), key=lambda kv: -kv[1])
        top = ", ".join(f"{m}={q:.2f}" for m, q in ranked[:3])
        return (f"best={rr.selected_model_ids[0]} "
                f"predicted_quality={float(rr.selected_scores[0]):.2f} | top: {top}")

    def answer_with_model(query: str, model_id: str) -> str:
        """Send QUERY to a specific MODEL_ID (one of the router's candidate
        models) and return its answer."""
        pool = agent.router.model_ids
        if model_id not in pool:
            return (f"ERROR: {model_id!r} is not a candidate model. "
                    f"Choose one of: {', '.join(pool)}")
        sc = agent._call_model(query, model_id, depth + 1)
        steps.append(sc)
        return sc.answer

    def route_and_answer(query: str) -> str:
        """Route QUERY to the best model (recursively decomposing if the router
        says it is still too complex) and return the answer."""
        sc = agent._solve(query, depth + 1)
        steps.append(sc)
        return sc.answer

    return [
        StructuredTool.from_function(route_query, handle_tool_error=True),
        StructuredTool.from_function(answer_with_model, handle_tool_error=True),
        StructuredTool.from_function(route_and_answer, handle_tool_error=True),
    ]


def _agent_executor_api():
    """``(AgentExecutor, create_tool_calling_agent)``: ``langchain.agents`` in
    0.3, ``langchain_classic.agents`` once langchain 1.x moved the legacy
    executor out."""
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
    except ImportError:
        try:
            from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError(
                "LangChainToolOrchestrator needs AgentExecutor: install `langchain>=0.3,<1` "
                "or, with langchain 1.x, `langchain-classic`"
            ) from exc
    return AgentExecutor, create_tool_calling_agent


class LangChainToolOrchestrator:
    name = "langchain-agent"

    def __init__(self, llm, *, system_prompt: str | None = None, max_iterations: int = 8,
                 verbose: bool = False):
        self._llm = llm
        self._max_iterations = max_iterations
        self._verbose = verbose
        self._system = system_prompt or (
            "You are an orchestrator for an LLM router. Break the user's request "
            "into sub-tasks and solve each with the tools. Prefer `route_and_answer` "
            "so the router picks the model for each sub-task; use `route_query` first "
            "if you want to inspect the choice. Combine the results into one final "
            "answer. Do not answer from your own knowledge."
        )

    def run(self, prompt: str, agent: "AgenticRouter", depth: int) -> tuple[str, list[SubCall]]:
        from langchain_core.prompts import ChatPromptTemplate

        AgentExecutor, create_tool_calling_agent = _agent_executor_api()
        steps: list[SubCall] = []
        tools = build_router_tools(agent, depth, steps)
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self._system),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        executor = AgentExecutor(
            agent=create_tool_calling_agent(self._llm, tools, chat_prompt),
            tools=tools, max_iterations=self._max_iterations, verbose=self._verbose,
            return_intermediate_steps=False,
        )
        out = executor.invoke({"input": prompt})
        answer = out.get("output", "") if isinstance(out, dict) else str(out)
        if not steps:  # agent answered without a tool call -> fall back to direct routing
            sc = agent._answer_directly(prompt, depth)
            return sc.answer, [sc]
        if str(answer).startswith(_STOPPED_PREFIX):
            # iteration / time limit hit: the stop notice is not an answer, but
            # the sub-answers gathered so far are
            answer = agent.synthesizer(prompt, [(s.prompt, s.answer) for s in steps])
        return answer, steps
