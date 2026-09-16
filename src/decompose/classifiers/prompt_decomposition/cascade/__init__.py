"""The escalation side: reading the weak model's own answer.

Separate from the heads because it answers a different question. They predict
what will happen from the prompt; this reads what did happen. Four independent
measurements put the prompt-side version of this decision at chance.
"""

from decompose.classifiers.prompt_decomposition.cascade.signals import (
    RESPONSE_FEATURES,
    response_matrix,
    response_signals,
)

__all__ = ["RESPONSE_FEATURES", "response_matrix", "response_signals"]
