"""LLM-judge client for the anchor-topology judge pipeline (docs/anchor_judge.md).

Greenfield: no other code in this repo calls an external LLM API. Kept
deliberately small and separate from ``router.data``/``router.nirt`` so it's
easy to delete wholesale if the anchor-judge line of work closes.
"""

from .client import JudgeClient, JudgeVerdict

__all__ = ["JudgeClient", "JudgeVerdict"]
