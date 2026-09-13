"""Prompt decomposition, serving side: prompt in, signals out.

A peer of the other serving subsystems, not the whole of `router`. Everything
here runs at inference: the five heads, the shared feature builders, and the
composite that returns them as one named vector.

    heads/domain        what the prompt is about        0.923 top-1
    heads/task          what it asks to be done         0.844 top-1
    heads/length        how big the job is              rho 0.582
    heads/attributes    11 gates and judged criteria    code 0.905 P@R50
    signals/surface     23 free deterministic features
    cascade/signals     what the weak model's own answer says
    composite           all of it, as 57 named columns

Difficulty is deliberately absent: four feature families failed to predict it
from prompt text, so it belongs downstream of this vector, not inside it.
"""

from router.prompt_decomposition.composite import RouterHead, RouterPrediction
from router.prompt_decomposition.heads.attributes import AttributeHead, AttributePrediction
from router.prompt_decomposition.heads.base import CalibratedHead
from router.prompt_decomposition.heads.domain import DomainHead, DomainPrediction
from router.prompt_decomposition.heads.length import LengthHead, LengthPrediction
from router.prompt_decomposition.heads.task import TaskHead, TaskPrediction

__all__ = [
    "AttributeHead", "AttributePrediction", "CalibratedHead", "DomainHead",
    "DomainPrediction", "LengthHead", "LengthPrediction", "RouterHead",
    "RouterPrediction", "TaskHead", "TaskPrediction",
]
