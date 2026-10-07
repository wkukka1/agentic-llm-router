# LangChainToolOrchestrator tool-error handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `LANGCHAINTOOLORCHESTRATOR-001`: `build_router_tools`'s `route_query`/`route_and_answer` tools are built with `handle_tool_error=True`, but LangChain's `handle_tool_error=True` only catches `langchain_core.tools.ToolException` — the real failure modes (`agent.route_decision` raising `LookupError`/`TypeError`) propagate straight through `AgentExecutor.invoke`, aborting the whole orchestrated run and discarding sub-answers already paid for. The module docstring promises exactly the behavior this bug breaks: "Tool exceptions are returned to the agent as observations instead of aborting the executor."

**Architecture:** One function, `build_router_tools` (`src/router/agentic/orchestrator.py:49-92`), gets two tool bodies (`route_query`, `route_and_answer`) wrapped to convert the specific exceptions `agent.route_decision`/`agent._solve` are documented to raise into `ToolException`, matching the pattern `answer_with_model` (the third tool, already correct) already uses of returning a clear error instead of crashing.

**Tech Stack:** Python 3.12, pytest, `langchain_core` (already an installed dependency).

**Spec:** `fixed_bugs/LangChainToolOrchestrator/LANGCHAINTOOLORCHESTRATOR-001-tool-errors-abort-the-executor/bug.md`.

## Global Constraints

- `answer_with_model`'s existing behavior (returns an `"ERROR: ..."` string for a bad `model_id`, never raises) is already correct per the bug file and must not change.
- The existing test `tests/router/agentic/test_agentic_router.py::test_langchain_orchestrator_tools_build_and_recurse` (a real, successful `route_query`/`route_and_answer` call against a working agent) must keep passing unchanged — this fix only changes behavior on the *failure* path.
- Only wrap the exception types the bug file identifies as the real failure modes: `LookupError` (no usable score) and `TypeError` (router can't score raw text) — from `agent.route_decision`, reached by both `route_query` directly and `route_and_answer` via `agent._solve`. Do not blanket-catch `Exception`, which would also swallow real bugs (e.g. an `AttributeError` from a programming mistake) and make them silently disappear as tool output instead of surfacing.

## Review Focus

- `route_query` on a working agent must still return its normal `"best=... predicted_quality=... | top: ..."` string — the `try`/`except` wrapper must not change the success path's return value or type. (Task 1)
- A `TypeError` from `route_decision` (the router-can't-score-raw-text case) must be caught the same way `LookupError` is — the bug file lists both, not just `LookupError`. (Task 1)
- `route_and_answer`'s wrapped call goes through `agent._solve`, not `agent.route_decision` directly — the fix must wrap the right call in each tool body, not just copy-paste the same `try` around the wrong line. (Task 1)
- An exception from a genuinely unexpected type (e.g. `AttributeError`) must still propagate (not be silently swallowed as some generic `ToolException`) — the Global Constraints' "only the documented failure modes" rule is a correctness requirement, not just a style preference; add a test pinning this. (Task 1)

---

## Task 1: Wrap `route_query`/`route_and_answer` failures as `ToolException`

**Files:**
- Modify: `src/router/agentic/orchestrator.py` (`build_router_tools`, ~L49-92)
- Test: `tests/router/agentic/test_agentic_router.py`

**Interfaces:**
- Consumes: `langchain_core.tools.ToolException` (external API — already importable alongside `StructuredTool` from `langchain_core.tools`).
- Produces: `build_router_tools(agent, depth, steps)` (unchanged signature/return — still `[route_query, answer_with_model, route_and_answer]` as `StructuredTool`s) — `route_query`/`route_and_answer` now raise `ToolException` (caught by each tool's existing `handle_tool_error=True`) instead of letting `LookupError`/`TypeError` propagate out of `AgentExecutor.invoke`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/router/agentic/test_agentic_router.py`, near `test_langchain_orchestrator_tools_build_and_recurse` (check the top of the file for an existing `from types import SimpleNamespace` import; add one if it's missing):

```python
def test_route_query_tool_error_is_returned_not_raised():
    """LANGCHAINTOOLORCHESTRATOR-001: handle_tool_error=True only catches
    ToolException -- route_query's LookupError must be converted so a bad
    prompt becomes an observation, not an executor-aborting crash."""
    pytest.importorskip("langchain_core")
    from router.agentic.orchestrator import build_router_tools

    class _FailingAgent:
        router = SimpleNamespace(model_ids=["a"])

        def route_decision(self, q):
            raise LookupError("no usable score for this prompt")

    tools = build_router_tools(_FailingAgent(), 0, [])
    route_query = next(t for t in tools if t.name == "route_query")

    result = route_query.invoke({"query": "x"})

    assert "no usable score for this prompt" in result


def test_route_and_answer_tool_error_is_returned_not_raised():
    """Same bug, via agent._solve (what route_and_answer calls)."""
    pytest.importorskip("langchain_core")
    from router.agentic.orchestrator import build_router_tools

    class _FailingAgent:
        router = SimpleNamespace(model_ids=["a"])

        def _solve(self, prompt, depth):
            raise TypeError("router can't score raw text")

    steps: list = []
    tools = build_router_tools(_FailingAgent(), 0, steps)
    route_and_answer = next(t for t in tools if t.name == "route_and_answer")

    result = route_and_answer.invoke({"query": "x"})

    assert "router can't score raw text" in result
    assert steps == []  # the failed call must not be recorded as a completed SubCall


def test_route_query_reraises_an_undocumented_exception_type():
    """An exception outside the documented LookupError/TypeError failure
    modes (e.g. an AttributeError from a real bug) must not be silently
    swallowed as tool output."""
    pytest.importorskip("langchain_core")
    from router.agentic.orchestrator import build_router_tools

    class _BrokenAgent:
        router = SimpleNamespace(model_ids=["a"])

        def route_decision(self, q):
            raise AttributeError("something else broke")

    tools = build_router_tools(_BrokenAgent(), 0, [])
    route_query = next(t for t in tools if t.name == "route_query")

    with pytest.raises(AttributeError, match="something else broke"):
        route_query.invoke({"query": "x"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/router/agentic/test_agentic_router.py::test_route_query_tool_error_is_returned_not_raised tests/router/agentic/test_agentic_router.py::test_route_and_answer_tool_error_is_returned_not_raised -v`
Expected: FAIL — both raise `LookupError`/`TypeError` out of `.invoke(...)` instead of returning a string.
Run: `.venv/Scripts/python.exe -m pytest tests/router/agentic/test_agentic_router.py::test_route_query_reraises_an_undocumented_exception_type -v`
Expected: PASS already (nothing catches anything yet, so it already propagates) — kept as a pin for after Step 3.

- [ ] **Step 3: Implement**

In `src/router/agentic/orchestrator.py`, `build_router_tools`:

```python
def build_router_tools(agent: "AgenticRouter", depth: int, steps: list[SubCall]):
    """LangChain ``StructuredTool``s bound to ``agent``; every model call is
    appended to ``steps``. Returns ``[route_query, answer_with_model,
    route_and_answer]``. Tool exceptions are returned to the agent as
    observations (``handle_tool_error``) instead of aborting the executor --
    ``route_query``/``route_and_answer`` wrap the ``LookupError``/``TypeError``
    ``agent.route_decision`` documents raising into ``ToolException``, which
    ``handle_tool_error=True`` actually catches (LANGCHAINTOOLORCHESTRATOR-001;
    any other exception type still propagates as a real crash, not a
    silently-swallowed tool result)."""
    from langchain_core.tools import StructuredTool, ToolException

    def route_query(query: str) -> str:
        """Ask the trained router which model should answer QUERY. Returns the
        chosen model id, its predicted quality (0-1), and the runner-up."""
        try:
            rr = agent.route_decision(query)
        except (LookupError, TypeError) as exc:
            raise ToolException(str(exc)) from exc
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
        says it is still too complex) and return the answer.

        This re-enters full triage + dispatch via ``agent._solve`` -- the only
        thing bounding it is ``agent.max_depth`` (a bare depth counter). See
        ``router.tools.router_tool``/``router.context.RoutingMode.MODEL_SELECTION``
        for the (unimplemented) design intent of a stricter tool-only guard (XA-07)."""
        try:
            sc = agent._solve(query, depth + 1)
        except (LookupError, TypeError) as exc:
            raise ToolException(str(exc)) from exc
        steps.append(sc)
        return sc.answer

    return [
        StructuredTool.from_function(route_query, handle_tool_error=True),
        StructuredTool.from_function(answer_with_model, handle_tool_error=True),
        StructuredTool.from_function(route_and_answer, handle_tool_error=True),
    ]
```

(Only the two `try`/`except` blocks and the `ToolException` import are new; `answer_with_model` is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/agentic/test_agentic_router.py -k "route_query or route_and_answer or langchain_orchestrator" -v`
Expected: PASS, all of them, including the pre-existing `test_langchain_orchestrator_tools_build_and_recurse`.

- [ ] **Step 5: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/router/agentic/orchestrator.py tests/router/agentic/test_agentic_router.py
git commit -m "fix(agentic): route_query/route_and_answer convert LookupError/TypeError to ToolException (LANGCHAINTOOLORCHESTRATOR-001)"
```

---

## Final step: update `known_bugs/`

Update status to `fixed` in `fixed_bugs/LangChainToolOrchestrator/LANGCHAINTOOLORCHESTRATOR-001-tool-errors-abort-the-executor/bug.md` and its row in `known_bugs/README.md`'s `## Bugs` table.
