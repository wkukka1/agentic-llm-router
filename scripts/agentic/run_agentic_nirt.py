"""Run the agentic router (NIRT-backed) on two prompts: one that should stay
single-shot, one that should trip the orchestrator. Then route the same
prompt at two different IRT-Router-style alpha blends to show the per-request
cost-preference override picking a different model.

Uses configs/agentic.yaml as-is (run: nirt-2d-projected, clients: echo --
no network calls, no API keys needed) so this is safe to run standalone.

    python scripts/agentic/run_agentic_nirt.py
"""

from __future__ import annotations

from router.agentic import AgenticRouter

# Short, single-topic factual question: not multipart (no "?"x3, no sequencing
# markers, no enumerated list, well under max_chars_single=600) -- HeuristicTriage
# only sends this to orchestrate if the router's best predicted quality q* is
# itself below the 0.55 threshold.
SINGLE_PROMPT = "What is the capital of France?"

# Multi-part by construction: "First," + "after that" are in triage.py's
# _MULTIPART_MARKERS, so HeuristicTriage.is_multipart() is True regardless of
# q* -- this always orchestrates (see triage.py:85-91).
ORCHESTRATE_PROMPT = (
    "First, explain how quicksort works. Then implement it in Rust. "
    "After that, analyse its time and space complexity, and finally compare "
    "it to mergesort."
)


def show(label: str, prompt: str, agent: AgenticRouter, *, alpha: float | None = None) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"prompt: {prompt!r}")
    result = agent.run(prompt, alpha=alpha)
    print(f"mode:               {result.mode}")
    print(f"triage_reason:      {result.triage_reason}")
    print(f"selected_model_id:  {result.selected_model_id}")
    print(f"predicted_quality:  {result.predicted_quality:.3f}")
    print(f"models_used:        {result.models_used()}")
    print(f"cost:               {result.cost}")
    print(f"answer:             {result.answer[:300]}")
    print(f"\nsummary():\n{result.summary()}")


def main() -> None:
    # configs/agentic.yaml: run=nirt-2d-projected, clients=echo, orchestrator=recursive
    agent = AgenticRouter.from_settings()
    print(repr(agent))

    show("SINGLE-SHOT (expected)", SINGLE_PROMPT, agent)
    show("ORCHESTRATED (expected)", ORCHESTRATE_PROMPT, agent)

    # Per-request IRT-Router-style reward blend (RouterModel.route(alpha=...),
    # see router/nirt/routing_decision.py:alpha_to_lam): alpha in (0, 1],
    # argmax[alpha*quality - (1-alpha)*linear_cost(C)]. 1.0 == quality-only
    # (same picks as the default lam=0.0 above); near 0 is cost-averse and
    # should pull the pick toward a cheaper model in the pool.
    show("ALPHA=1.0 (quality-first)", ORCHESTRATE_PROMPT, agent, alpha=1.0)
    show("ALPHA=0.05 (cost-averse)", ORCHESTRATE_PROMPT, agent, alpha=0.0099)


if __name__ == "__main__":
    main()
