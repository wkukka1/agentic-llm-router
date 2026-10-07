# Agentic execution layer: two-person plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the classes in the agentic / production-router path that are still scaffolding, split into two tracks that two people can work on at the same time with almost no file overlap.

**Architecture:** `router.agentic` (`AgenticRouter`, `RecursiveOrchestrator`, `LangChainToolOrchestrator`, triage, decomposition, clients) is fully implemented and tested. What is missing is the production execution layer in `router/execution/` (plan, forecast, orchestrate against a shared `BudgetLedger`) and the provider/tool layer it calls (direct-SDK adapters, `RouterTool`, `MCPTool`). Track A builds the execution engine; Track B builds providers and tools. They meet at interfaces that already exist.

**Tech Stack:** Python 3.12, pytest, the `openai` / `anthropic` SDKs (Track B), LangChain (optional `agentic` extra).

## What is not implemented

| Class | Location | Gap |
|---|---|---|
| `Orchestrator` | [execution/orchestrator.py](../../../src/router/execution/orchestrator.py) | `run()`, `spawn()` and `can_spawn()` work; `plan()`, `execute()` and `replan()` are abstract and no concrete subclass exists. Nothing creates one. Open bug ORCHESTRATOR-001. |
| `PlanForecaster` | [execution/forecast.py](../../../src/router/execution/forecast.py) | `score_plan()` works; `forecast_cost()`, `forecast_quality()` and `forecast_latency()` are abstract with no concrete subclass. |
| `OrchestratorFactory` | same file | The concrete class must be passed in; there is no default or registry. |
| `RouterTool.route_for_model_selection` | [tools/router_tool.py](../../../src/router/tools/router_tool.py) | Raises `NotImplementedError`. `RoutingMode.MODEL_SELECTION`, meant to stop the tool re-entering full agentic dispatch, is never checked anywhere. |
| `MCPTool.execute` | [tools/mcp.py](../../../src/router/tools/mcp.py) | Placeholder with no MCP client. |
| `OpenAIAdapter` / `AnthropicAdapter` / `GoogleAdapter` | [llm/adapters/](../../../src/router/llm/adapters/) | Stubs: `complete()` and `stream()` both raise. Only `LangChainAdapter` makes real calls. |
| `Router` (top level) | [router.py](../../../src/router/router.py) | `orchestrator_factory` is never used. There is no latency signal, so `ModelScore.expected_latency` is always 0 (RC-01 warnings). |
| `DefaultRoutingPolicy` | [policy.py](../../../src/router/policy.py) | Always sends the result back to the user; never escalates to `PARENT_ORCHESTRATOR`, even though `escalation_threshold` exists. |

Out of scope: `LocalArtifactStore.save` for `BaselineArtifactPayload` (minor, not on the agentic path).

## Day 0: both people, about 1 hour

- [ ] **Agree the interfaces.** They already exist, so nothing new is needed:
  - `Plan` / `PlanStep` ([execution/plan.py](../../../src/router/execution/plan.py))
  - `ProviderAdapter.complete()` / `stream()` returning `LLMResponse` with `latency` filled in
  - `RouterTool.execute(RoutingRequest) -> RoutingDecision`
- [ ] **Get approval for new classes.** Per the repo's `gating-new-classes` rule, each new class needs sign-off before it is written, and a proposed edit to the classDiagram in [docs/architecture.md](../../architecture.md) afterwards. Get sign-off up front for:
  - Person A: a concrete orchestrator (working name `RoutedOrchestrator`) and a concrete forecaster (working name `RouterPlanForecaster`).
  - Person B: none. B only fills in classes that already exist.
- [ ] **Branches.** Each person works in their own git worktree: `feat/execution-engine` (A) and `feat/providers-tools` (B). The only shared file expected to conflict is `docs/architecture.md`; whoever merges second rebases it.

## Track A: execution engine

**Owns:** `src/router/execution/*`, `src/router/router.py`, `src/router/policy.py`, `src/router/agentic/orchestrator.py`.

- [ ] **A1. Fix ORCHESTRATOR-001**: `run()` releases a child's spawn-time budget hold before re-reserving it, leaving a window for a sibling to take it. Do this first; the new orchestrator depends on it.
  - Done when: the repro in `known_bugs/_repros` passes, a regression test is added, and the entry is moved to `fixed_bugs/Orchestrator/ORCHESTRATOR-001-.../{bug,fix}.md`.
- [ ] **A2. Concrete `PlanForecaster`.**
  - Cost: from `router.llm.cost` and the step's estimated input/output tokens.
  - Quality: from `RouterModel` predicted scores for each step's model.
  - Latency: from a per-model table (stub now; B4 supplies real numbers).
  - Done when: unit tests on a hand-built `Plan` cover each forecast and `score_plan()` with and without a ledger.
- [ ] **A3. Concrete `Orchestrator`.**
  - `plan()`: decomposer + router pick a model per step and produce a `Plan`.
  - `execute()`: runs steps in dependency order through `LLMClient`, returns `ExecutionResult` with actual cost.
  - `replan()`: sends failed steps to the next-best model.
  - Done when: tests pass with `EchoAdapter` only, no network.
- [ ] **A4. Wire it in.**
  - Add a default/registry to `OrchestratorFactory`.
  - Use `Router.orchestrator_factory` in `Router.route`.
  - Add escalation to `DefaultRoutingPolicy`: `PARENT_ORCHESTRATOR` when every candidate scores below `escalation_threshold`.
  - Done when: an end-to-end test shows `Router.route` escalating and the orchestrator running.
- [ ] **A5. Fix RECURSIVEORCHESTRATOR-001**: solve sub-tasks concurrently (thread pool; per-run state is already context-local).
  - Done when: the existing 52 tests in `tests/router/agentic` and `tests/router/llm` still pass, plus a test showing sub-tasks overlap in time.

## Track B: providers and tools

**Owns:** `src/router/llm/adapters/*`, `src/router/tools/*`, `src/router/context.py`, `src/router/agentic/decompose.py`, `src/router/agentic/triage.py`.

- [ ] **B1. `OpenAIAdapter` and `AnthropicAdapter`.** `complete()` and `stream()` on the official SDKs, filling in input/output tokens, cost, latency and `raw`/`meta`.
  - Done when: unit tests pass against a mocked SDK, plus one opt-in live test skipped unless an API-key env var is set.
- [ ] **B2. `RouterTool.route_for_model_selection`.** Route a *copy* of the request (`dataclasses.replace(request, mode=RoutingMode.MODEL_SELECTION)`) through `DefaultRoutingPolicy`. Change `build_router_tools` to use it so a tool call can never re-enter full agentic dispatch.
  - Done when: a test shows a tool call never triggers a second orchestration, and the caller's request is not mutated.
- [ ] **B3. Close the two agentic bugs.**
  - AGENTICROUTER-003 caveat: make the `LangChainToolOrchestrator` callback stop the agent when `max_calls` is exhausted by its own reasoning steps.
  - NAIVEDECOMPOSER-002: fix the `\bthen,\b` marker in `_SPLIT_RE`.
  - Done when: both have regression tests and are moved to `fixed_bugs/`.
- [ ] **B4. Latency signal.** Adapters record `latency`; a small rolling per-model store feeds `ModelScore.expected_latency`; remove the RC-01 warnings in `Router.route`.
  - Done when: a non-default `max_latency` actually filters candidates in a test.
- [ ] **B5. Optional: `GoogleAdapter` and `MCPTool`** (the latter on the `mcp` SDK).
  - Done when: tests pass against mocked clients.

## Handoffs and integration

- **A3 does not wait for B.** It builds and tests against `EchoAdapter`, then switches to the real adapters once B1 merges.
- **A2 depends on B4** only for real latency numbers; until then it uses the stub table.
- **Merge order:** B1 → A1 → A2 / A3 → B2 → A4 → B4. B2 lands before A4 so the orchestrator uses the guarded `RouterTool`.
- [ ] **Final milestone, done together: real-model evaluation.** A script that runs about 50 prompts through both `AgenticRouter` and `Router` + `Orchestrator` with two real providers, and reports quality, cost and latency against single-model routing. Neither path has been run end to end with real models yet.
