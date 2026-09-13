"""Prompt attributes the arena labels for free: gates, and judged criteria.

Eleven binary heads over one encoder pass. The four *gates* -- code, maths,
creative writing, constrained instruction -- are preconditions worth acting on.
The seven *criteria* are the arena's judged difficulty dimensions, served as
inputs to whatever combines them downstream, never as a difficulty prediction.
"""

from prompt_decomposition.attribute_classifier.head import AttributeHead, AttributePrediction
from prompt_decomposition.attribute_classifier.model import AttributeModel
from prompt_decomposition.attribute_classifier.taxonomy import (
    ATTRIBUTE_NAMES,
    ATTRIBUTES,
    CRITERIA,
    GATES,
    Attribute,
    describe,
)

__all__ = [
    "ATTRIBUTES",
    "ATTRIBUTE_NAMES",
    "CRITERIA",
    "GATES",
    "Attribute",
    "AttributeHead",
    "AttributeModel",
    "AttributePrediction",
    "describe",
]
