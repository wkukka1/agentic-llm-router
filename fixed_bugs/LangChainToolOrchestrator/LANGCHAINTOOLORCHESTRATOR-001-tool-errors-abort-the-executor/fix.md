# LANGCHAINTOOLORCHESTRATOR-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-25, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`route_query` / `route_and_answer` convert `LookupError`/`TypeError` into `ToolException`, which `handle_tool_error=True` catches; other exception types still propagate. `answer_with_model` is not wrapped (a `LookupError` from `_call_model` still aborts). Tests: `test_route_query_tool_error_is_returned_not_raised`, `test_route_and_answer_tool_error_is_returned_not_raised`, `test_route_query_reraises_an_undocumented_exception_type`.

## Fix suggested when the bug was filed

Wrap the tool bodies and convert `LookupError`/`TypeError`/`ValueError` into
`langchain_core.tools.ToolException` (or return an `"ERROR: ..."` string as
`answer_with_model` does).

## Regression tests

`tests/router/agentic/test_agentic_router.py:288` `test_route_query_tool_error_is_returned_not_raised`

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
```

`tests/router/agentic/test_agentic_router.py:309` `test_route_and_answer_tool_error_is_returned_not_raised`

```python
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
```

`tests/router/agentic/test_agentic_router.py:330` `test_route_query_reraises_an_undocumented_exception_type`

```python
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

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/agentic/orchestrator.py`

```diff
--- a/src/router/agentic/orchestrator.py
+++ b/src/router/agentic/orchestrator.py
@@ -50,13 +50,21 @@ def build_router_tools(agent: "AgenticRouter", depth: int, steps: list[SubCall])
     """LangChain ``StructuredTool``s bound to ``agent``; every model call is
     appended to ``steps``. Returns ``[route_query, answer_with_model,
     route_and_answer]``. Tool exceptions are returned to the agent as
-    observations (``handle_tool_error``) instead of aborting the executor."""
-    from langchain_core.tools import StructuredTool
+    observations (``handle_tool_error``) instead of aborting the executor --
+    ``route_query``/``route_and_answer`` wrap the ``LookupError``/``TypeError``
+    ``agent.route_decision`` documents raising into ``ToolException``, which
+    ``handle_tool_error=True`` actually catches (LANGCHAINTOOLORCHESTRATOR-001;
+    any other exception type still propagates as a real crash, not a
+    silently-swallowed tool result)."""
+    from langchain_core.tools import StructuredTool, ToolException
 
     def route_query(query: str) -> str:
         """Ask the trained router which model should answer QUERY. Returns the
         chosen model id, its predicted quality (0-1), and the runner-up."""
-        rr = agent.route_decision(query)
+        try:
+            rr = agent.route_decision(query)
+        except (LookupError, TypeError) as exc:
+            raise ToolException(str(exc)) from exc
         ranked = sorted(zip(rr.model_ids, rr.scores[0]), key=lambda kv: -kv[1])
         top = ", ".join(f"{m}={q:.2f}" for m, q in ranked[:3])
         return (f"best={rr.selected_model_ids[0]} "
@@ -81,7 +89,10 @@ def build_router_tools(agent: "AgenticRouter", depth: int, steps: list[SubCall])
         thing bounding it is ``agent.max_depth`` (a bare depth counter). See
         ``router.tools.router_tool``/``router.context.RoutingMode.MODEL_SELECTION``
         for the (unimplemented) design intent of a stricter tool-only guard (XA-07)."""
-        sc = agent._solve(query, depth + 1)
+        try:
+            sc = agent._solve(query, depth + 1)
+        except (LookupError, TypeError) as exc:
+            raise ToolException(str(exc)) from exc
         steps.append(sc)
         return sc.answer
 
```
