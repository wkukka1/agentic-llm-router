# AGENTICROUTER-003: `max_calls` and `cost` only cover routed-model calls, not the helper LLMs

- **Status:** fixed 2026-09-23 (uncommitted, partial -- see caveat below;
  [triage.py](../../src/router/agentic/triage.py), [decompose.py](../../src/router/agentic/decompose.py),
  [orchestrator.py](../../src/router/agentic/orchestrator.py), [router.py](../../src/router/agentic/router.py);
  regression tests `test_llm_triage_counts_against_max_calls`,
  `test_llm_decomposer_and_synthesizer_count_against_max_calls` in `tests/router/agentic/test_agentic_router.py`)
- **Severity:** medium (a spend cap that does not cap the spend)
- **Class:** `AgenticRouter._call_model` ([router.py:205-208](../../src/router/agentic/router.py#L205-L208))
- **Found:** 2026-09-19

## What

The module docstring says `max_calls` "caps the model calls one `run()` may make across the
whole recursion tree". `self._calls` is incremented in exactly one place, `_call_model`,
i.e. calls to the *routed* models through `ClientRegistry`. Everything else that calls a
model bypasses both the counter and the cost total:

- `LLMTriage`: `self._chat.invoke(...)` ([triage.py:130](../../src/router/agentic/triage.py#L130)), once per triaged prompt, including every sub-prompt.
- `LLMDecomposer` ([decompose.py:98](../../src/router/agentic/decompose.py#L98)) and `LLMSynthesizer` ([decompose.py:158](../../src/router/agentic/decompose.py#L158)).
- `LangChainToolOrchestrator`: the tool-calling agent's own LLM, up to `max_iterations=8`
  steps per orchestrated node ([orchestrator.py:139-144](../../src/router/agentic/orchestrator.py#L139-L144)).

## Failure scenario

`max_calls=3` on an orchestrated request with LLM triage, decomposer and synthesizer, and the
LangChain orchestrator: three routed calls are allowed, plus a triage call per node, a
decompose, a synthesize, and up to 8 orchestrator-LLM steps per node, none counted or
priced. Recursion to `max_depth=2` multiplies this. `AgenticResult.cost` reports only the
routed calls.

## Suggested fix

Route the helper LLMs through the same counter (a `Budget` object passed to them, or wrap
their `invoke`), or rename the limit to `max_routed_calls` and document that helper calls
are uncapped and unpriced.

## Fix applied

All routed and helper LLM calls now share one per-run budget (`router.py`'s `_RunState`, also
the AGENTICROUTER-004 fix). `LLMTriage`, `LLMDecomposer`, `LLMSynthesizer` each grew an
`on_call: Optional[Callable[[], None]]` hook, called immediately before their `_chat.invoke(...)`
and *outside* the `try` that catches provider errors (so a budget-exhausted `RuntimeError`
propagates instead of being swallowed as "provider call failed, fall back to heuristic").
`AgenticRouter.__init__` wires `helper.on_call = self._use_call` onto any of `self.triage`/
`self.decomposer`/`self.synthesizer` that exposes the attribute (`hasattr` duck-typing, so a
plain heuristic/naive/user callable with no `on_call` is a harmless no-op).

**Caveat — `LangChainToolOrchestrator` is only partially fixed.** Its tool-calling agent can
make several internal reasoning-step LLM calls per `run()`, with no single `_chat.invoke()`
call site to hook and no `SubCall` to attach a mid-reasoning failure to. It now gets a
LangChain callback handler (`on_chat_model_start`/`on_llm_start`) that *counts* each of those
calls against the shared budget, so a routed call that follows (`answer_with_model`/
`route_and_answer`, still enforced by `agent._call_model`) correctly sees the reduced
remaining budget — but the callback itself does not raise, so a tool-agent that burns through
`max_calls` purely on its own reasoning, with zero routed tool calls, is still not stopped
mid-reasoning. This is the bug's own documented fallback ("or ... document that helper calls
are uncapped"), narrowed to just this one call site instead of applied to all of them.
