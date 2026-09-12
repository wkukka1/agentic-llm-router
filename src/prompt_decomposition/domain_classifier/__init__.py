"""Domain classifier: what the prompt is about.

Self-contained -- the label space in `taxonomy`, the serving head in `head`.

0.923 top-1 / 0.980 top-2 on 402 prompts labelled outside this project. Eight
classes served, predicted as ten and merged post hoc: training directly on the
coarse labels reaches 0.738 where predicting fine and summing reaches 0.763.
"""

from prompt_decomposition.domain_classifier.head import DomainHead, DomainPrediction
from prompt_decomposition.domain_classifier.taxonomy import (
    DOMAIN_DESCRIPTIONS,
    DOMAIN_LABELS,
    Domain,
    apply_domain_merges,
)

__all__ = [
    "DOMAIN_DESCRIPTIONS",
    "DOMAIN_LABELS",
    "Domain",
    "DomainHead",
    "DomainPrediction",
    "apply_domain_merges",
]
