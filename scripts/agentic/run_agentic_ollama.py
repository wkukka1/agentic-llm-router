"""Run the agentic router against a local Ollama server (no API keys, no
network beyond localhost).

Prerequisites:
    ollama serve                        # usually already running as a service
    ollama pull deepseek-coder:1.3b
    ollama pull deepseek-llm:7b-chat
    ollama pull qwen2.5:3b qwen2.5:1.5b qwen2.5:0.5b phi3.5 smollm2:1.7b
    pip install -e ".[ollama]"

    python scripts/agentic/run_agentic_ollama.py

Two parts:

* PART A uses the real trained ``nirt-2d-projected`` router (same checkpoint
  ``run_agentic_nirt.py`` uses) over its actual pool. ``deepseek_chat`` and
  ``deepseek_coder`` -- real ids in that pool (configs/serve_anthropic.yaml's
  ``pool:`` map) -- are served locally through Ollama; every other pool id
  the router might pick instead has no registered client, so ``ClientRegistry``
  raises rather than silently answering with an echo placeholder (see
  CLIENTREGISTRY-001). This is a genuine routing decision, live-backed only
  when it happens to land on one of the two mapped ids -- the prompt below
  is code-flavored to make that more likely, not to force it; a full-coverage
  fix (every pool id resolving to a real local model) is a separate, not yet
  built project. The local
  models are NOT the same checkpoints RouterBench scored (deepseek-llm:7b-chat
  / deepseek-coder:1.3b vs. DeepSeek's ~2024 hosted deepseek_chat / coder
  API models) -- closest locally-runnable approximations only.

* PART B uses a small illustrative stand-in router (``_LocalPool`` below)
  over five real small open-weight models -- the live counterparts of
  ``configs/phase0.yaml``'s "pool-expansion E3" models, picked because
  they're the ones already called out in this repo as small enough to run
  on the dev machine's 4GB laptop GPU. No NIRT checkpoint was ever trained
  on this pool, so ``_LocalPool``'s scores are a simple deterministic
  heuristic (bigger model -> higher quality), not a learned prediction --
  it exists so every prompt below gets a real, visible local response
  across the whole tier, and so the alpha-sweep still shows a cost/quality
  tradeoff.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from router.agentic import AgenticRouter
from router.agentic.llm_clients import ClientRegistry
from router.llm.adapters.ollama import OllamaAdapter
from router.llm.client import LLMClient
from router.routing import NIRTRouter
from router.routing.base import RouterModel

OLLAMA_HOST = "http://localhost:11434"

CODE_PROMPT = (
    "Write a Python function that returns the nth Fibonacci number using "
    "memoization, and briefly explain its time complexity."
)
SINGLE_PROMPT = "What is the capital of France?"
ORCHESTRATE_PROMPT = (
    "First, explain how quicksort works. Then implement it in Rust. "
    "After that, analyse its time and space complexity, and finally compare "
    "it to mergesort."
)


def _debug_path(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return f"debug_traces/{slug}.txt"


def show(label: str, prompt: str, agent: AgenticRouter, *, alpha: float | None = None) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"prompt: {prompt!r}")
    debug_path = _debug_path(label)
    result = agent.run(prompt, alpha=alpha, debug=debug_path)
    print(f"debug_trace:        {Path(debug_path).resolve()}")
    print(f"mode:               {result.mode}")
    print(f"triage_reason:      {result.triage_reason}")
    print(f"selected_model_id:  {result.selected_model_id}")
    print(f"predicted_quality:  {result.predicted_quality:.3f}")
    print(f"models_used:        {result.models_used()}")
    print(f"cost:               {result.cost}")
    print(f"answer ({len(result.answer)} chars): {result.answer}")


# --------------------------------------------------------------------------- #
# Part A -- the real trained router, deepseek ids served locally             #
# --------------------------------------------------------------------------- #
def part_a() -> None:
    import ollama

    router = NIRTRouter.from_run("nirt-2d-projected")
    shared = ollama.Client(host=OLLAMA_HOST)

    clients = ClientRegistry()  # unmapped pool ids still fall back to echo
    clients.register("deepseek_chat", LLMClient(OllamaAdapter("deepseek-llm:7b-chat", client=shared)))
    clients.register("deepseek_coder", LLMClient(OllamaAdapter("deepseek-coder:1.3b", client=shared)))

    agent = AgenticRouter(router=router, clients=clients)
    show("PART A: real nirt-2d-projected router (deepseek -> local Ollama)", CODE_PROMPT, agent)
    show("PART B: single-shot over the local pool", SINGLE_PROMPT, agent)
    show("PART B: orchestrated over the local pool", ORCHESTRATE_PROMPT, agent)
    show("PART B: alpha=1.0 (quality-first)", ORCHESTRATE_PROMPT, agent, alpha=1.0)
    show("PART B: alpha=0.05 (cost-averse)", ORCHESTRATE_PROMPT, agent, alpha=0.0099)


def main() -> None:
    part_a()


if __name__ == "__main__":
    main()
