"""Every LLMRouterBench performance-cost model must be a registered canonical id --
otherwise training.data.quality's uncanonicalized_model check flags it on every run."""

from __future__ import annotations

from training.data.model_registry import canonical_model_id, is_canonical

_LLMROUTERBENCH_MODELS = [
    "claude-sonnet-4", "deepseek-v3-0324", "deepseek-v3.1-terminus", "deepseek-r1-0528",
    "gemini-2.5-flash", "gemini-2.5-pro", "gpt-5-chat", "gpt-5",
    "qwen3-235b-a22b-2507", "qwen3-235b-a22b-thinking-2507", "glm-4.6",
    "kimi-k2-0905", "intern-s1",
]


def test_every_llmrouterbench_model_is_a_registered_canonical_id():
    for native in _LLMROUTERBENCH_MODELS:
        cid = canonical_model_id(native)
        assert is_canonical(cid), f"{native!r} -> {cid!r} not registered as canonical"
